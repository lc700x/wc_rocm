import os
import ctypes
import torch
import threading
import time
from typing import Optional
from ._wc_cuda import WcCapture as _NativeWcCapture

__all__ = ["WindowsCapture", "Frame", "InternalCaptureControl", "get_luid", "list_windows"]


# --- Utilities ---
def list_windows():
    """Returns a list of (hwnd, title) for all visible windows."""
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    titles = []

    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowText(hwnd, buff, length + 1)
                titles.append((hwnd, buff.value))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)
    return titles


# --- CUDA Management (Internal) ---
_cudart = None


def _get_cudart():
    global _cudart
    if _cudart is None:
        torch_dir = os.path.dirname(torch.__file__)
        site_packages = os.path.dirname(torch_dir)

        candidates = [
            os.path.join(torch_dir, "lib"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
        ]

        cudart_path = None
        for lib_dir in candidates:
            if not os.path.exists(lib_dir):
                continue
            for f in os.listdir(lib_dir):
                if f.startswith("cudart64") and f.endswith(".dll"):
                    cudart_path = os.path.join(lib_dir, f)
                    break
            if cudart_path:
                break

        if not cudart_path:
            raise RuntimeError("Could not find cudart64_*.dll in torch/lib or nvidia/cuda_runtime/bin")

        _cudart = ctypes.WinDLL(cudart_path)

        # API Definitions
        _cudart.cudaGraphicsD3D11RegisterResource.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        _cudart.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        _cudart.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _cudart.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _cudart.cudaGraphicsSubResourceGetMappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        _cudart.cudaMemcpy2DFromArray.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        _cudart.cudaMemcpy2DFromArrayAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    return _cudart


def get_luid(device_id=0):
    if ctypes.windll.nvcuda.cuInit(0) != 0:
        raise RuntimeError("Failed to initialize CUDA Driver API")
    device = ctypes.c_int()
    ctypes.windll.nvcuda.cuDeviceGet(ctypes.byref(device), device_id)
    luid_buf = ctypes.create_string_buffer(8)
    mask = ctypes.c_uint()
    ctypes.windll.nvcuda.cuDeviceGetLuid(luid_buf, ctypes.byref(mask), device)
    luid_bytes = luid_buf.raw
    low = int.from_bytes(luid_bytes[:4], "little")
    high = int.from_bytes(luid_bytes[4:], "little", signed=True)
    return low, high


# --- Interface Classes ---


class Frame:
    def __init__(self, tensor: torch.Tensor, width: int, height: int):
        self.frame_buffer = tensor  # This is the PyTorch CUDA Tensor
        self.width = width
        self.height = height


class InternalCaptureControl:
    def __init__(self, capture_instance):
        self._capture = capture_instance

    def stop(self):
        self._capture.stop()


class _DX11ToPyTorchBridge:
    def __init__(self, texture_ptr, width, height, device_id=0):
        self.width = width
        self.height = height
        self.resource = ctypes.c_void_p()
        self.cudart = _get_cudart()
        self.cudart.cudaSetDevice(device_id)
        res = self.cudart.cudaGraphicsD3D11RegisterResource(ctypes.byref(self.resource), texture_ptr, 0)
        if res != 0:
            raise RuntimeError(f"cudaGraphicsD3D11RegisterResource failed: {res}")

    def update(self, stream, device_id):
        """Copies from DX11 texture to a new PyTorch tensor using a dedicated stream."""
        target_tensor = torch.empty((self.height, self.width, 4), dtype=torch.uint8, device=f"cuda:{device_id}")

        h_stream = ctypes.c_void_p(stream.cuda_stream)
        res = self.cudart.cudaGraphicsMapResources(1, ctypes.byref(self.resource), h_stream)
        if res != 0:
            return None
        try:
            cu_array = ctypes.c_void_p()
            self.cudart.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(cu_array), self.resource, 0, 0)
            # Use Async version to respect the dedicated stream
            self.cudart.cudaMemcpy2DFromArrayAsync(
                target_tensor.data_ptr(),
                self.width * 4,
                cu_array,
                0,
                0,
                self.width * 4,
                self.height,
                3,  # cudaMemcpyDeviceToDevice
                h_stream,
            )
            # Synchronize ONLY the dedicated stream to ensure the copy is finished before returning
            stream.synchronize()
        finally:
            self.cudart.cudaGraphicsUnmapResources(1, ctypes.byref(self.resource), h_stream)

        return target_tensor

    def __del__(self):
        if hasattr(self, "resource") and self.resource:
            try:
                self.cudart.cudaGraphicsUnregisterResource(self.resource)
            except Exception:
                pass


class WindowsCapture:
    def __init__(
        self,
        cursor_capture: Optional[bool] = True,
        draw_border: Optional[bool] = None,
        secondary_window: Optional[bool] = None,
        minimum_update_interval: Optional[int] = None,
        dirty_region: Optional[bool] = None,
        monitor_index: Optional[int] = None,
        window_name: Optional[str] = None,
        window_hwnd: Optional[int] = None,
        device_id: int = 0,  # Custom addition for CUDA
    ):
        # Resolve LUID if possible, else fallback to default
        try:
            low, high = get_luid(device_id)
            luid = (low, high)
        except Exception:
            luid = None

        self.device_id = device_id
        self._inner = _NativeWcCapture(
            luid=luid, monitor_index=monitor_index, window_hwnd=window_hwnd, window_title=window_name
        )
        self.frame_handler = None
        self.closed_handler = None
        self._bridge = None
        self._running = False
        self._loop_thread = None
        self._control = InternalCaptureControl(self)
        self._last_id = 0
        self._stream = None

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.frame_handler = handler
        elif handler.__name__ == "on_closed":
            self.closed_handler = handler
        return handler

    def start(self):
        """Starts the capture and blocks (emulating windows-capture behavior)."""
        if not self.frame_handler:
            raise Exception("on_frame_arrived handler not set")

        # Initialize a dedicated stream for this capture session
        self._stream = torch.cuda.Stream(device=self.device_id)

        self._inner.start()
        self._running = True
        self._last_id = 0

        # Wait for thread to initialize (up to 1 second)
        start_wait = time.time()
        while not self._inner.is_alive() and time.time() - start_wait < 1.0:
            err = self._inner.get_last_error()
            if err:
                raise RuntimeError(err)
            time.sleep(0.01)

        try:
            while self._running:
                # Use a small timeout to keep Python responsive (Ctrl+C)
                res = self._inner.get_frame(self._last_id, timeout=0.1)
                if res:
                    gpu_frame, new_id = res
                    self._last_id = new_id

                    w, h = gpu_frame.width, gpu_frame.height
                    ow, oh = gpu_frame.original_width, gpu_frame.original_height
                    if self._bridge is None or self._bridge.width != w or self._bridge.height != h:
                        self._bridge = _DX11ToPyTorchBridge(gpu_frame.texture_ptr, w, h, self.device_id)

                    # Update returns a fresh, synchronized tensor
                    tensor = self._bridge.update(self._stream, self.device_id)
                    if tensor is not None:
                        # Slice the tensor to original size to remove alignment padding
                        sliced_tensor = tensor[:oh, :ow]
                        frame = Frame(sliced_tensor, ow, oh)
                        self.frame_handler(frame, self._control)
                else:
                    # If no frame arrived, check if the capture thread is still alive
                    if not self._inner.is_alive():
                        err = self._inner.get_last_error()
                        if err:
                            raise RuntimeError(err)
                        break
        finally:
            self.stop()
            if self.closed_handler:
                self.closed_handler()

    def start_free_threaded(self):
        """Starts the capture on a background thread."""
        self._loop_thread = threading.Thread(target=self.start, daemon=True)
        self._loop_thread.start()
        return self

    def stop(self):
        self._running = False
        self._inner.stop()

    def wait(self):
        """Waits for the background thread to finish."""
        if self._loop_thread:
            self._loop_thread.join()
