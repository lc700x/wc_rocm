import os
import ctypes
import threading
import time
from typing import Optional

import torch

from ._wc_rocm import WcCapture as _NativeWcCapture

__all__ = ["WindowsCapture", "Frame", "InternalCaptureControl", "get_luid", "list_windows"]


# Utilities

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


# GPU vendor detection via DXGI

_NVIDIA_VENDOR_IDS = {0x10DE}
_AMD_VENDOR_IDS    = {0x1002, 0x1022}

# IID_IDXGIFactory1 = {770AAE78-F26F-4DBA-A829-253C83D1B387}
_IID_DXGI_FACTORY1 = (ctypes.c_byte * 16)(
    0x78, 0xAE, 0x0A, 0x77, 0x6F, 0xF2, 0xBA, 0x4D,
    0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87,
)


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description",          ctypes.c_wchar * 128),
        ("VendorId",             ctypes.c_uint),
        ("DeviceId",             ctypes.c_uint),
        ("SubSysId",             ctypes.c_uint),
        ("Revision",             ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory",ctypes.c_size_t),
        ("SharedSystemMemory",   ctypes.c_size_t),
        ("AdapterLuid_Low",      ctypes.c_uint),
        ("AdapterLuid_High",     ctypes.c_int),
        ("Flags",                ctypes.c_uint),
    ]


def _vtable(com_ptr):
    vt_ptr = ctypes.cast(com_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    return ctypes.cast(vt_ptr, ctypes.POINTER(ctypes.c_void_p))


def _com_release(com_ptr):
    if com_ptr and com_ptr.value:
        Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(_vtable(com_ptr)[2])
        Release(com_ptr)


def _enum_adapters():
    """Yield (index, _DxgiAdapterDesc1, adapter_ptr); caller must release adapter_ptr."""
    dxgi = ctypes.WinDLL("dxgi.dll")
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_IID_DXGI_FACTORY1), ctypes.byref(factory))
    if hr != 0:
        return
    vt_f = _vtable(factory)
    EnumAdapters1 = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
    )(vt_f[12])
    i = 0
    try:
        while True:
            adapter = ctypes.c_void_p()
            if EnumAdapters1(factory, i, ctypes.byref(adapter)) != 0:
                break
            vt_a = _vtable(adapter)
            GetDesc1 = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DxgiAdapterDesc1)
            )(vt_a[10])
            desc = _DxgiAdapterDesc1()
            GetDesc1(adapter, ctypes.byref(desc))
            yield i, desc, adapter
            i += 1
    finally:
        _com_release(factory)


def _get_gpu_vendor_luid(device_index: int = 0):
    """
    Return (vendor_id, luid_low, luid_high) for the GPU at device_index.
    Skips software/WARP adapters (VendorId == 0x1414).
    """
    hw_adapters = []
    for _, desc, adapter in _enum_adapters():
        if desc.VendorId not in (0x1414,):  # skip WARP
            hw_adapters.append((desc.VendorId, desc.AdapterLuid_Low, desc.AdapterLuid_High))
        _com_release(adapter)
    if device_index >= len(hw_adapters):
        device_index = 0
    if not hw_adapters:
        return 0, 0, 0
    v, lo, hi = hw_adapters[device_index]
    return v, lo, hi


# LUID helpers (public)

def get_luid(device_id: int = 0):
    """
    Return (luid_low, luid_high) for the GPU at device_id.

    For NVIDIA GPUs this delegates to the CUDA driver (nvcuda.dll) for
    exact CUDA-device → LUID mapping.  For AMD / unknown vendors it uses
    DXGI adapter enumeration.
    """
    vendor, luid_low, luid_high = _get_gpu_vendor_luid(device_id)
    if vendor in _NVIDIA_VENDOR_IDS:
        # Original NVIDIA path via CUDA driver API
        try:
            if ctypes.windll.nvcuda.cuInit(0) != 0:
                raise RuntimeError("cuInit failed")
            dev = ctypes.c_int()
            ctypes.windll.nvcuda.cuDeviceGet(ctypes.byref(dev), device_id)
            buf  = ctypes.create_string_buffer(8)
            mask = ctypes.c_uint()
            ctypes.windll.nvcuda.cuDeviceGetLuid(buf, ctypes.byref(mask), dev)
            raw  = buf.raw
            return int.from_bytes(raw[:4], "little"), int.from_bytes(raw[4:], "little", signed=True)
        except Exception:
            pass  # fall through to DXGI path
    return luid_low, luid_high


# NVIDIA CUDA interop bridge

_cudart = None


def _get_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart

    torch_dir       = os.path.dirname(torch.__file__)
    site_packages   = os.path.dirname(torch_dir)
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

    lib = ctypes.WinDLL(cudart_path)
    lib.cudaGraphicsD3D11RegisterResource.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]
    lib.cudaGraphicsUnregisterResource.argtypes   = [ctypes.c_void_p]
    lib.cudaGraphicsMapResources.argtypes         = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cudaGraphicsUnmapResources.argtypes       = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cudaGraphicsSubResourceGetMappedArray.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
    ]
    lib.cudaMemcpy2DFromArrayAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    _cudart = lib
    return _cudart


class _DX11ToPyTorchBridge:
    """NVIDIA CUDA interop: D3D11 texture → CUDA tensor (zero-copy on GPU)."""

    def __init__(self, texture_ptr, width, height, device_id=0):
        self.width    = width
        self.height   = height
        self.resource = ctypes.c_void_p()
        self.cudart   = _get_cudart()
        self.cudart.cudaSetDevice(device_id)
        res = self.cudart.cudaGraphicsD3D11RegisterResource(
            ctypes.byref(self.resource), texture_ptr, 0
        )
        if res != 0:
            raise RuntimeError(f"cudaGraphicsD3D11RegisterResource failed: {res}")

    def update(self, stream, device_id):
        target_tensor = torch.empty(
            (self.height, self.width, 4), dtype=torch.uint8, device=f"cuda:{device_id}"
        )
        h_stream = ctypes.c_void_p(stream.cuda_stream)
        res = self.cudart.cudaGraphicsMapResources(1, ctypes.byref(self.resource), h_stream)
        if res != 0:
            return None
        try:
            cu_array = ctypes.c_void_p()
            self.cudart.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(cu_array), self.resource, 0, 0
            )
            self.cudart.cudaMemcpy2DFromArrayAsync(
                target_tensor.data_ptr(), self.width * 4,
                cu_array, 0, 0, self.width * 4, self.height,
                3,  # cudaMemcpyDeviceToDevice
                h_stream,
            )
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


# AMD HIP interop bridge


# GetSharedHandle returns a KMT-style handle on AMD; use type 7 (D3D11ResourceKmt)
_HIP_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 7
_HIP_CHANNEL_FORMAT_KIND_UNSIGNED               = 1
_HIP_MEMCPY_DEVICE_TO_DEVICE                    = 3

_hip = None


def _find_hip_dll() -> str:
    import importlib.util
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
    raise RuntimeError("Could not find amdhip64*.dll. Install rocm-sdk-core or AMD ROCm.")


def _get_hip():
    global _hip
    if _hip is not None:
        return _hip
    lib = ctypes.WinDLL(_find_hip_dll())
    lib.hipSetDevice.restype                               = ctypes.c_int
    lib.hipSetDevice.argtypes                              = [ctypes.c_int]
    lib.hipImportExternalMemory.restype                    = ctypes.c_int
    lib.hipImportExternalMemory.argtypes                   = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipExternalMemoryGetMappedMipmappedArray.restype   = ctypes.c_int
    lib.hipExternalMemoryGetMappedMipmappedArray.argtypes  = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.hipDestroyExternalMemory.restype                   = ctypes.c_int
    lib.hipDestroyExternalMemory.argtypes                  = [ctypes.c_void_p]
    lib.hipGetMipmappedArrayLevel.restype                  = ctypes.c_int
    lib.hipGetMipmappedArrayLevel.argtypes                 = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]
    lib.hipFreeMipmappedArray.restype                      = ctypes.c_int
    lib.hipFreeMipmappedArray.argtypes                     = [ctypes.c_void_p]
    lib.hipMemcpy2DFromArrayAsync.restype                  = ctypes.c_int
    lib.hipMemcpy2DFromArrayAsync.argtypes                 = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.hipStreamSynchronize.restype                       = ctypes.c_int
    lib.hipStreamSynchronize.argtypes                      = [ctypes.c_void_p]
    _hip = lib
    return _hip


# HIP structs -----------------------------------------------------------

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

_BGRA8_FORMAT = _HipChannelFormatDesc(8, 8, 8, 8, _HIP_CHANNEL_FORMAT_KIND_UNSIGNED)




class _HipD3D11Bridge:
    """AMD HIP interop: D3D11 shared texture → HIP tensor (zero-copy on GPU)."""

    def __init__(self, width: int, height: int, device_id: int):
        self.width    = width
        self.height   = height
        self._device  = device_id
        self._hip     = _get_hip()
        self._hip.hipSetDevice(device_id)
        # Allocate a persistent PyTorch ROCm tensor; its data_ptr() is the HIP device pointer
        self._tensor  = torch.empty(
            (height, width, 4), dtype=torch.uint8, device=f"cuda:{device_id}"
        )
        self._buf     = ctypes.c_void_p(self._tensor.data_ptr())

    def update(self, shared_handle_int: int, stream, device_id: int):
        """Import D3D11 shared texture handle via HIP external memory and DMA to self._buf."""
        shared_handle = ctypes.c_void_p(shared_handle_int)
        if not shared_handle.value:
            raise RuntimeError("shared_handle is NULL; texture may lack D3D11_RESOURCE_MISC_SHARED")

        hdesc = _HipExternalMemoryHandleDesc()
        hdesc.type                  = _HIP_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
        hdesc.handle.win32.handle   = shared_handle
        hdesc.handle.win32.name     = None
        hdesc.size                  = self.width * self.height * 4
        hdesc.flags                 = 0

        ext_mem = ctypes.c_void_p()
        err = self._hip.hipImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(hdesc))
        if err != 0:
            raise RuntimeError(f"hipImportExternalMemory failed: {err}")

        mip_desc = _HipExternalMemoryMipmappedArrayDesc()
        mip_desc.offset            = 0
        mip_desc.formatDesc        = _BGRA8_FORMAT
        mip_desc.extent.width      = self.width
        mip_desc.extent.height     = self.height
        mip_desc.extent.depth      = 0
        mip_desc.flags             = 0
        mip_desc.numLevels         = 1

        mip_array = ctypes.c_void_p()
        err = self._hip.hipExternalMemoryGetMappedMipmappedArray(
            ctypes.byref(mip_array), ext_mem, ctypes.byref(mip_desc)
        )
        if err != 0:
            self._hip.hipDestroyExternalMemory(ext_mem)
            raise RuntimeError(f"hipExternalMemoryGetMappedMipmappedArray failed: {err}")

        level0 = ctypes.c_void_p()
        err = self._hip.hipGetMipmappedArrayLevel(ctypes.byref(level0), mip_array, 0)
        if err != 0:
            self._hip.hipFreeMipmappedArray(mip_array)
            self._hip.hipDestroyExternalMemory(ext_mem)
            raise RuntimeError(f"hipGetMipmappedArrayLevel failed: {err}")

        h_stream = ctypes.c_void_p(stream.cuda_stream)
        err = self._hip.hipMemcpy2DFromArrayAsync(
            self._buf, self.width * 4,
            level0, 0, 0, self.width * 4, self.height,
            _HIP_MEMCPY_DEVICE_TO_DEVICE, h_stream,
        )
        self._hip.hipFreeMipmappedArray(mip_array)
        self._hip.hipDestroyExternalMemory(ext_mem)
        if err != 0:
            raise RuntimeError(f"hipMemcpy2DFromArrayAsync failed: {err}")

        stream.synchronize()
        # Return a clone so the caller owns a distinct tensor
        return self._tensor.clone()



# Interface classes

class Frame:
    def __init__(self, tensor: torch.Tensor, width: int, height: int):
        self.frame_buffer = tensor  # PyTorch tensor on the GPU (BGRA uint8)
        self.width        = width
        self.height       = height


class InternalCaptureControl:
    def __init__(self, capture_instance):
        self._capture = capture_instance

    def stop(self):
        self._capture.stop()


# WindowsCapture — auto-selects CUDA (NVIDIA) or HIP (AMD) backend

class WindowsCapture:
    """
    Screen capture that delivers frames as PyTorch GPU tensors.

    Automatically uses:
      - NVIDIA GPUs: CUDA interop via cudaGraphicsD3D11RegisterResource
      - AMD GPUs:    HIP external-memory interop via hipImportExternalMemory

    Usage is identical regardless of GPU vendor:

        capture = WindowsCapture(monitor_index=1)

        @capture.event
        def on_frame_arrived(frame, control):
            # frame.frame_buffer: uint8 BGRA tensor on cuda:0
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
    ):
        # Detect GPU vendor and resolve LUID
        vendor, luid_low, luid_high = _get_gpu_vendor_luid(device_id)
        self._vendor    = vendor
        self._is_amd    = vendor in _AMD_VENDOR_IDS
        self.device_id  = device_id

        try:
            luid = get_luid(device_id)
        except Exception:
            luid = (luid_low, luid_high) if (luid_low or luid_high) else None

        self._inner         = _NativeWcCapture(
            luid=luid, monitor_index=monitor_index,
            window_hwnd=window_hwnd, window_title=window_name,
        )
        self.frame_handler  = None
        self.closed_handler = None
        self._bridge        = None
        self._running       = False
        self._loop_thread   = None
        self._control       = InternalCaptureControl(self)
        self._last_id       = 0
        self._stream        = None

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.frame_handler = handler
        elif handler.__name__ == "on_closed":
            self.closed_handler = handler
        return handler

    def start(self):
        """Starts the capture and blocks until stopped."""
        if not self.frame_handler:
            raise RuntimeError("on_frame_arrived handler not set")

        self._stream  = torch.cuda.Stream(device=self.device_id)
        self._inner.start()
        self._running = True
        self._last_id = 0

        start_wait = time.time()
        while not self._inner.is_alive() and time.time() - start_wait < 1.0:
            err = self._inner.get_last_error()
            if err:
                raise RuntimeError(err)
            time.sleep(0.01)

        try:
            while self._running:
                res = self._inner.get_frame(self._last_id, timeout=0.1)
                if res:
                    gpu_frame, new_id = res
                    self._last_id = new_id

                    w,  h  = gpu_frame.width,          gpu_frame.height
                    ow, oh = gpu_frame.original_width,  gpu_frame.original_height

                    if self._is_amd:
                        # AMD: HIP external-memory path — handle from Rust via IDXGIResource
                        if self._bridge is None or self._bridge.width != w or self._bridge.height != h:
                            self._bridge = _HipD3D11Bridge(w, h, self.device_id)
                        tensor = self._bridge.update(gpu_frame.shared_handle, self._stream, self.device_id)
                    else:
                        # NVIDIA: CUDA graphics interop path
                        if self._bridge is None or self._bridge.width != w or self._bridge.height != h:
                            self._bridge = _DX11ToPyTorchBridge(gpu_frame.texture_ptr, w, h, self.device_id)
                        tensor = self._bridge.update(self._stream, self.device_id)

                    if tensor is not None:
                        sliced_tensor = tensor[:oh, :ow]
                        frame = Frame(sliced_tensor, ow, oh)
                        self.frame_handler(frame, self._control)
                else:
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
