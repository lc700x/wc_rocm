import ctypes
import threading
import time
from typing import Optional

import torch

from ._wc_rocm import WcCapture as _NativeWcCapture

__all__ = ["WindowsCapture", "Frame", "InternalCaptureControl", "get_luid", "list_windows"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_windows():
    """Returns a list of (hwnd, title) for all visible windows."""
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

    ctypes.windll.user32.EnumWindows(EnumWindowsProc(foreach_window), 0)
    return titles


# ---------------------------------------------------------------------------
# DXGI adapter enumeration — used to find the AMD GPU LUID
# ---------------------------------------------------------------------------

# IID_IDXGIFactory1 = {770AAE78-F26F-4DBA-A829-253C83D1B387}
_IID_DXGI_FACTORY1 = (ctypes.c_byte * 16)(
    0x78, 0xAE, 0x0A, 0x77, 0x6F, 0xF2, 0xBA, 0x4D,
    0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87,
)


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description",           ctypes.c_wchar * 128),
        ("VendorId",              ctypes.c_uint),
        ("DeviceId",              ctypes.c_uint),
        ("SubSysId",              ctypes.c_uint),
        ("Revision",              ctypes.c_uint),
        ("DedicatedVideoMemory",  ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory",    ctypes.c_size_t),
        ("AdapterLuid_Low",       ctypes.c_uint),
        ("AdapterLuid_High",      ctypes.c_int),
        ("Flags",                 ctypes.c_uint),
    ]


def _vtable(com_ptr):
    vt = ctypes.cast(com_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    return ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))


def _com_release(com_ptr):
    if com_ptr and com_ptr.value:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(_vtable(com_ptr)[2])(com_ptr)


def _enum_adapters():
    """Yield (index, desc, adapter_ptr); caller must release adapter_ptr."""
    factory = ctypes.c_void_p()
    if ctypes.WinDLL("dxgi.dll").CreateDXGIFactory1(
        ctypes.byref(_IID_DXGI_FACTORY1), ctypes.byref(factory)
    ) != 0:
        return
    EnumAdapters1 = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
    )(_vtable(factory)[12])
    i = 0
    try:
        while True:
            adapter = ctypes.c_void_p()
            if EnumAdapters1(factory, i, ctypes.byref(adapter)) != 0:
                break
            GetDesc1 = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DxgiAdapterDesc1)
            )(_vtable(adapter)[10])
            desc = _DxgiAdapterDesc1()
            GetDesc1(adapter, ctypes.byref(desc))
            yield i, desc, adapter
            i += 1
    finally:
        _com_release(factory)


def get_luid(device_id: int = 0):
    """Return (luid_low, luid_high) for the hardware GPU at device_id via DXGI."""
    hw = []
    for _, desc, adapter in _enum_adapters():
        if desc.VendorId != 0x1414:  # skip WARP software adapter
            hw.append((desc.AdapterLuid_Low, desc.AdapterLuid_High))
        _com_release(adapter)
    idx = min(device_id, len(hw) - 1) if hw else -1
    return hw[idx] if idx >= 0 else (0, 0)


# ---------------------------------------------------------------------------
# amdhip64 loader
# ---------------------------------------------------------------------------

_hip = None


def _find_hip_dll() -> str:
    import importlib.util, os
    spec = importlib.util.find_spec("_rocm_sdk_core")
    candidates = []
    if spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
        candidates += [
            os.path.join(pkg_dir, "bin", "amdhip64_7.dll"),
            os.path.join(pkg_dir, "bin", "amdhip64.dll"),
        ]
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
        candidates.append(os.path.join(sys32, name))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("Could not find amdhip64*.dll — install rocm-sdk-core or AMD ROCm.")


def _get_hip():
    global _hip
    if _hip is not None:
        return _hip
    lib = ctypes.WinDLL(_find_hip_dll())
    lib.hipSetDevice.restype                              = ctypes.c_int
    lib.hipSetDevice.argtypes                             = [ctypes.c_int]
    lib.hipImportExternalMemory.restype                   = ctypes.c_int
    lib.hipImportExternalMemory.argtypes                  = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipExternalMemoryGetMappedMipmappedArray.restype  = ctypes.c_int
    lib.hipExternalMemoryGetMappedMipmappedArray.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.hipDestroyExternalMemory.restype                  = ctypes.c_int
    lib.hipDestroyExternalMemory.argtypes                 = [ctypes.c_void_p]
    lib.hipGetMipmappedArrayLevel.restype                 = ctypes.c_int
    lib.hipGetMipmappedArrayLevel.argtypes                = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]
    lib.hipFreeMipmappedArray.restype                     = ctypes.c_int
    lib.hipFreeMipmappedArray.argtypes                    = [ctypes.c_void_p]
    lib.hipMemcpy2DFromArrayAsync.restype                 = ctypes.c_int
    lib.hipMemcpy2DFromArrayAsync.argtypes                = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_void_p,
    ]
    # External-semaphore interop (GPU-to-GPU fence). Present on ROCm 7+.
    for _name in ("hipImportExternalSemaphore", "hipWaitExternalSemaphoresAsync",
                  "hipDestroyExternalSemaphore"):
        if not hasattr(lib, _name):
            break
    else:
        lib.hipImportExternalSemaphore.restype            = ctypes.c_int
        lib.hipImportExternalSemaphore.argtypes           = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        lib.hipWaitExternalSemaphoresAsync.restype        = ctypes.c_int
        lib.hipWaitExternalSemaphoresAsync.argtypes       = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p,
        ]
        lib.hipDestroyExternalSemaphore.restype           = ctypes.c_int
        lib.hipDestroyExternalSemaphore.argtypes          = [ctypes.c_void_p]
    if hasattr(lib, "hipGetLastError"):
        lib.hipGetLastError.restype                       = ctypes.c_int
        lib.hipGetLastError.argtypes                      = []
    _hip = lib
    return _hip


# ---------------------------------------------------------------------------
# HIP structs
# ---------------------------------------------------------------------------

class _HipChannelFormatDesc(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                ("z", ctypes.c_int), ("w", ctypes.c_int), ("f", ctypes.c_int)]

class _HipExtent(ctypes.Structure):
    _fields_ = [("width", ctypes.c_size_t), ("height", ctypes.c_size_t), ("depth", ctypes.c_size_t)]

class _HipWin32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]

class _HipHandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _HipWin32Handle), ("nvSciBufObject", ctypes.c_void_p)]

class _HipExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type",     ctypes.c_int),
        ("handle",   _HipHandleUnion),
        ("size",     ctypes.c_ulonglong),
        ("flags",    ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]

class _HipExternalMemoryMipmappedArrayDesc(ctypes.Structure):
    _fields_ = [
        ("offset",     ctypes.c_ulonglong),
        ("formatDesc", _HipChannelFormatDesc),
        ("extent",     _HipExtent),
        ("flags",      ctypes.c_uint),
        ("numLevels",  ctypes.c_uint),
    ]


# --- External semaphore interop (GPU-to-GPU D3D11 fence) -------------------

class _HipExternalSemaphoreHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type",     ctypes.c_int),
        ("handle",   _HipHandleUnion),
        ("flags",    ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _HipExtSemWaitParams(ctypes.Structure):
    class _Params(ctypes.Structure):
        class _Fence(ctypes.Structure):
            _fields_ = [("value", ctypes.c_ulonglong)]

        class _NvSciSync(ctypes.Union):
            _fields_ = [("fence", ctypes.c_void_p), ("reserved", ctypes.c_ulonglong)]

        class _KeyedMutex(ctypes.Structure):
            _fields_ = [("key", ctypes.c_ulonglong)]

        _fields_ = [
            ("fence",     _Fence),
            ("nvSciSync", _NvSciSync),
            ("keyedMutex", _KeyedMutex),
            ("reserved",  ctypes.c_uint * 10),
        ]

    _fields_ = [
        ("params",   _Params),
        ("flags",    ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


# HIP external-semaphore handle type for a shared D3D11/D3D12 fence NT handle.
# AMD's ROCm runtime imports the D3D11 fence's shared NT handle under the
# D3D12Fence type (4); the dedicated D3D11Fence type (5) is rejected. The two
# fence objects are interop-compatible (same monotonic timeline), so type 4 is
# the correct, working choice on AMD.
_HIP_EXT_SEM_D3D_FENCE = 4

# BGRA8 = 4 × 8-bit unsigned channels
_BGRA8_FORMAT = _HipChannelFormatDesc(8, 8, 8, 8, 1)  # kind=1 → unsigned
# GetSharedHandle on AMD returns a KMT handle → type 7
_HIP_EXT_MEM_D3D11_KMT = 7
_HIP_MEMCPY_D2D         = 3  # hipMemcpyDeviceToDevice


# ---------------------------------------------------------------------------
# HIP D3D11 interop bridge
# ---------------------------------------------------------------------------

class _HipD3D11Bridge:
    """
    Imports a D3D11 shared texture into HIP once, then DMAs into a pair of
    pre-allocated ROCm tensors (double-buffer) so update() never needs clone().
    """

    def __init__(self, shared_handle_int: int, width: int, height: int, device_id: int):
        self.width   = width
        self.height  = height
        self._hip    = _get_hip()
        self._hip.hipSetDevice(device_id)

        if not shared_handle_int:
            raise RuntimeError("shared_handle is NULL — texture lacks D3D11_RESOURCE_MISC_SHARED")

        # Import external memory once for the lifetime of this bridge
        hdesc = _HipExternalMemoryHandleDesc()
        hdesc.type                = _HIP_EXT_MEM_D3D11_KMT
        hdesc.handle.win32.handle = ctypes.c_void_p(shared_handle_int)
        hdesc.handle.win32.name   = None
        hdesc.size                = width * height * 4
        hdesc.flags               = 0

        self._ext_mem = ctypes.c_void_p()
        err = self._hip.hipImportExternalMemory(ctypes.byref(self._ext_mem), ctypes.byref(hdesc))
        if err != 0:
            raise RuntimeError(f"hipImportExternalMemory failed: {err}")

        mip_desc = _HipExternalMemoryMipmappedArrayDesc()
        mip_desc.formatDesc    = _BGRA8_FORMAT
        mip_desc.extent.width  = width
        mip_desc.extent.height = height
        mip_desc.extent.depth  = 0
        mip_desc.numLevels     = 1

        self._mip_array = ctypes.c_void_p()
        err = self._hip.hipExternalMemoryGetMappedMipmappedArray(
            ctypes.byref(self._mip_array), self._ext_mem, ctypes.byref(mip_desc)
        )
        if err != 0:
            self._hip.hipDestroyExternalMemory(self._ext_mem)
            raise RuntimeError(f"hipExternalMemoryGetMappedMipmappedArray failed: {err}")

        self._level0 = ctypes.c_void_p()
        err = self._hip.hipGetMipmappedArrayLevel(ctypes.byref(self._level0), self._mip_array, 0)
        if err != 0:
            self._hip.hipFreeMipmappedArray(self._mip_array)
            self._hip.hipDestroyExternalMemory(self._ext_mem)
            raise RuntimeError(f"hipGetMipmappedArrayLevel failed: {err}")

        # Double-buffer: two tensors ping-pong so update() never needs .clone()
        self._tensors = [
            torch.empty((height, width, 4), dtype=torch.uint8, device=f"cuda:{device_id}"),
            torch.empty((height, width, 4), dtype=torch.uint8, device=f"cuda:{device_id}"),
        ]
        self._bufs = [ctypes.c_void_p(t.data_ptr()) for t in self._tensors]
        self._idx  = 0

    def update(self, stream, synchronize: bool = False) -> torch.Tensor:
        """Async DMA into the current buffer and optionally synchronize stream."""
        err = self._hip.hipMemcpy2DFromArrayAsync(
            self._bufs[self._idx], self.width * 4,
            self._level0, 0, 0, self.width * 4, self.height,
            _HIP_MEMCPY_D2D, ctypes.c_void_p(stream.cuda_stream),
        )
        if err != 0:
            raise RuntimeError(f"hipMemcpy2DFromArrayAsync failed: {err}")
        if synchronize:
            stream.synchronize()
        tensor = self._tensors[self._idx]
        self._idx ^= 1  # swap to the other buffer for next frame
        return tensor

    def __del__(self):
        try:
            if getattr(self, "_mip_array", None) and self._mip_array.value:
                self._hip.hipFreeMipmappedArray(self._mip_array)
            if getattr(self, "_ext_mem", None) and self._ext_mem.value:
                self._hip.hipDestroyExternalMemory(self._ext_mem)
        except Exception:
            pass


class _GpuFenceUnsupported(Exception):
    """Raised when the HIP runtime cannot import the D3D11 shared fence."""


class _HipExternalSemaphore:
    """Imports a shared D3D11 fence (NT handle) as a HIP external semaphore so
    the HIP stream can wait on GPU copy completion without any CPU busy-wait."""

    def __init__(self, fence_handle_int: int, device_id: int):
        self._hip = _get_hip()
        self._hip.hipSetDevice(device_id)

        if not hasattr(self._hip, "hipImportExternalSemaphore"):
            raise _GpuFenceUnsupported("HIP runtime lacks external-semaphore support")
        if not fence_handle_int:
            raise _GpuFenceUnsupported("fence handle is NULL")

        desc = _HipExternalSemaphoreHandleDesc()
        desc.type                = _HIP_EXT_SEM_D3D_FENCE
        desc.handle.win32.handle = ctypes.c_void_p(fence_handle_int)
        desc.handle.win32.name   = None
        desc.flags               = 0

        self._sem = ctypes.c_void_p()
        err = self._hip.hipImportExternalSemaphore(ctypes.byref(self._sem), ctypes.byref(desc))
        if err != 0:
            self._sem = ctypes.c_void_p()
            # Clear the sticky HIP error so a subsequent fallback (or torch
            # allocation) doesn't inherit it.
            if hasattr(self._hip, "hipGetLastError"):
                self._hip.hipGetLastError()
            raise _GpuFenceUnsupported(f"hipImportExternalSemaphore failed: {err}")

        # Reusable wait-params; only the fence value changes per frame.
        self._wait = _HipExtSemWaitParams()

    def wait(self, fence_value: int, stream) -> None:
        """Enqueue a GPU-side wait until the fence reaches fence_value."""
        self._wait.params.fence.value = fence_value
        err = self._hip.hipWaitExternalSemaphoresAsync(
            ctypes.byref(self._sem), ctypes.byref(self._wait), 1,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if err != 0:
            raise RuntimeError(f"hipWaitExternalSemaphoresAsync failed: {err}")

    def __del__(self):
        try:
            if getattr(self, "_sem", None) and self._sem.value:
                self._hip.hipDestroyExternalSemaphore(self._sem)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class Frame:
    def __init__(self, tensor: torch.Tensor, width: int, height: int):
        self.frame_buffer = tensor  # uint8 BGRA tensor on ROCm device
        self.width        = width
        self.height       = height


class InternalCaptureControl:
    def __init__(self, capture_instance):
        self._capture = capture_instance

    def stop(self):
        self._capture.stop()


class WindowsCapture:
    """
    Windows screen capture delivering frames as PyTorch ROCm tensors.

    Uses HIP external-memory interop to import D3D11 shared textures directly
    onto the AMD GPU with no CPU copies.

        capture = WindowsCapture(monitor_index=1)

        @capture.event
        def on_frame_arrived(frame, control):
            # frame.frame_buffer: uint8 BGRA tensor on cuda:0 (ROCm)
            # NOTE: tensor is reused next frame — clone if you need to keep it.
            ...
            control.stop()

        @capture.event
        def on_closed():
            pass

        capture.start()
    """

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
        device_id: int = 0,
        synchronize_copy: bool = False,
        use_gpu_fence: bool = True,
    ):
        self.device_id = device_id
        self.synchronize_copy = synchronize_copy
        self.use_gpu_fence = use_gpu_fence
        self._monitor_index = monitor_index
        self._window_name = window_name
        self._window_hwnd = window_hwnd
        try:
            luid = get_luid(device_id)
        except Exception:
            luid = None

        self._luid = luid
        self._inner = self._make_inner(use_gpu_fence)
        self.frame_handler  = None
        self.closed_handler = None
        self._bridges: list  = [None, None]
        self._last_wh: tuple = (0, 0)
        self._running        = False
        self._loop_thread   = None
        self._control       = InternalCaptureControl(self)
        self._last_id       = 0
        self._stream        = None
        self._semaphore     = None

    def _make_inner(self, use_gpu_fence: bool):
        return _NativeWcCapture(
            luid=self._luid, monitor_index=self._monitor_index,
            window_hwnd=self._window_hwnd, window_title=self._window_name,
            use_gpu_fence=use_gpu_fence,
        )

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.frame_handler = handler
        elif handler.__name__ == "on_closed":
            self.closed_handler = handler
        return handler

    def start(self):
        """Start capture and block until stopped."""
        if not self.frame_handler:
            raise RuntimeError("on_frame_arrived handler not set")

        self._stream   = torch.cuda.current_stream(self.device_id)
        try:
            self._capture_loop()
        except _GpuFenceUnsupported:
            # The HIP runtime can't import the D3D11 shared fence on this
            # system. Transparently fall back to the legacy CPU-wait path and
            # retry once so capture still works correctly.
            if not self.use_gpu_fence:
                raise
            self.use_gpu_fence = False
            self._semaphore = None
            self._inner = self._make_inner(False)
            self._capture_loop()
        finally:
            if self.closed_handler:
                self.closed_handler()

    def _capture_loop(self):
        self._inner.start()
        self._running  = True
        self._last_id  = 0
        self._bridges  = [None, None]
        self._last_wh  = (0, 0)
        self._semaphore = None

        t0 = time.time()
        while not self._inner.is_alive() and time.time() - t0 < 1.0:
            err = self._inner.get_last_error()
            if err:
                raise RuntimeError(err)
            time.sleep(0.01)

        try:
            while self._running:
                res = self._inner.get_frame(self._last_id, timeout=0.1)
                if not res:
                    if not self._inner.is_alive():
                        err = self._inner.get_last_error()
                        if err:
                            raise RuntimeError(err)
                        break
                    continue

                gpu_frame, new_id = res
                self._last_id = new_id
                w, h    = gpu_frame.width,         gpu_frame.height
                ow, oh  = gpu_frame.original_width, gpu_frame.original_height
                buf_idx = gpu_frame.buf_idx  # 0 or 1, set by Rust ping-pong

                # Drop both bridges on resolution change; Rust recreated both textures
                # with new KMT handles so the old imported handles are stale.
                if (w, h) != self._last_wh:
                    self._bridges = [None, None]
                    self._last_wh = (w, h)

                if self._bridges[buf_idx] is None:
                    self._bridges[buf_idx] = _HipD3D11Bridge(
                        gpu_frame.shared_handle, w, h, self.device_id
                    )

                # GPU-to-GPU fence: enqueue a stream wait for the D3D11 copy to
                # finish before the DMA reads the shared texture — no CPU spin.
                if self.use_gpu_fence and gpu_frame.fence_value:
                    if self._semaphore is None:
                        self._semaphore = _HipExternalSemaphore(
                            gpu_frame.fence_handle, self.device_id
                        )
                    self._semaphore.wait(gpu_frame.fence_value, self._stream)

                tensor = self._bridges[buf_idx].update(
                    self._stream, synchronize=self.synchronize_copy
                )
                self.frame_handler(Frame(tensor[:oh, :ow], ow, oh), self._control)

        finally:
            self.stop()

    def start_free_threaded(self):
        """Start capture on a background daemon thread."""
        self._loop_thread = threading.Thread(target=self.start, daemon=True)
        self._loop_thread.start()
        return self

    def stop(self):
        self._running = False
        self._inner.stop()

    def wait(self):
        """Wait for the background thread to finish."""
        if self._loop_thread:
            self._loop_thread.join()