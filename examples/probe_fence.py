"""Probe which HIP external-semaphore handle type works for a D3D11 shared fence."""
import ctypes
import threading
import time

import wc_rocm
from wc_rocm import get_luid, _get_hip
from wc_rocm import _HipExternalSemaphoreHandleDesc, _HipHandleUnion
from wc_rocm._wc_rocm import WcCapture as _NativeWcCapture


def _activity(stop):
    u = ctypes.windll.user32
    def run():
        x = 0
        while not stop.is_set():
            x ^= 1
            u.SetCursorPos(10 + x, 10)
            time.sleep(0.002)
    threading.Thread(target=run, daemon=True).start()


def main():
    stop = threading.Event()
    _activity(stop)

    luid = get_luid(0)
    inner = _NativeWcCapture(luid=luid, monitor_index=1, use_gpu_fence=True)
    inner.start()
    time.sleep(0.3)

    fence_handle = 0
    fence_value = 0
    last = 0
    t0 = time.time()
    while time.time() - t0 < 3.0:
        res = inner.get_frame(last, 0.1)
        if res:
            gf, last = res
            fence_handle = gf.fence_handle
            fence_value = gf.fence_value
            if fence_handle:
                break
    inner.stop()
    stop.set()

    print(f"fence_handle={fence_handle:#x} fence_value={fence_value}")
    if not fence_handle:
        print("No fence handle obtained — aborting probe.")
        return

    hip = _get_hip()
    hip.hipSetDevice(0)
    # ensure import funcs exist
    for name in ("hipImportExternalSemaphore", "hipGetLastError"):
        print(name, hasattr(hip, name))
    if hasattr(hip, "hipGetLastError"):
        hip.hipGetLastError.restype = ctypes.c_int
        hip.hipGetLastError.argtypes = []

    for type_id, label in [(4, "D3D12Fence"), (5, "D3D11Fence"),
                           (2, "OpaqueWin32"), (3, "OpaqueWin32Kmt")]:
        desc = _HipExternalSemaphoreHandleDesc()
        desc.type = type_id
        desc.handle.win32.handle = ctypes.c_void_p(fence_handle)
        desc.handle.win32.name = None
        desc.flags = 0
        sem = ctypes.c_void_p()
        err = hip.hipImportExternalSemaphore(ctypes.byref(sem), ctypes.byref(desc))
        print(f"type={type_id:<2} ({label:<14}) -> err={err}")
        if hasattr(hip, "hipGetLastError"):
            hip.hipGetLastError()  # clear sticky error
        if err == 0 and hasattr(hip, "hipDestroyExternalSemaphore"):
            hip.hipDestroyExternalSemaphore.restype = ctypes.c_int
            hip.hipDestroyExternalSemaphore.argtypes = [ctypes.c_void_p]
            hip.hipDestroyExternalSemaphore(sem)


if __name__ == "__main__":
    main()
