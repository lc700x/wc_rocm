import ctypes
import numpy as np
import torch
import threading
import time
from typing import Optional
from ._wc_xpu import WcCapture as _NativeWcCapture

__all__ = ["WindowsCapture", "Frame", "InternalCaptureControl", "list_windows"]


# --- Windows API helpers ---
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


# --- Interface Classes ---


class Frame:
    def __init__(self, tensor, width: int, height: int):
        self.frame_buffer = tensor  # PyTorch XPU Tensor
        self.width = width
        self.height = height


class InternalCaptureControl:
    def __init__(self, capture_instance):
        self._capture = capture_instance

    def stop(self):
        self._capture.stop()


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
        reuse_output_buffer: bool = False,
        output_buffer_count: int = 3,
    ):
        self._inner = _NativeWcCapture(
            luid=None, monitor_index=monitor_index, window_hwnd=window_hwnd, window_title=window_name
        )
        self.frame_handler = None
        self.closed_handler = None
        self._control = InternalCaptureControl(self)
        self._reuse_output_buffer = bool(reuse_output_buffer)
        self._output_buffer_count = max(2, int(output_buffer_count))
        self._running = False
        self._loop_thread = None
        self._last_id = 0

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
                res = self._inner.get_frame(self._last_id, timeout=0.1)
                if res:
                    gpu_frame, new_id = res
                    self._last_id = new_id

                    w, h = gpu_frame.width, gpu_frame.height
                    ow, oh = gpu_frame.original_width, gpu_frame.original_height

                    # Pixel data is pre-computed on the capture thread (avoids D3D11 thread-affinity issues)
                    raw_bytes = gpu_frame.pixel_data
                    if raw_bytes:
                        # Convert raw BGRA8 bytes to numpy -> torch XPU
                        data_len = ow * oh * 4
                        arr = np.frombuffer(raw_bytes[:data_len], dtype=np.uint8).copy()
                        arr = arr.reshape((oh, ow, 4))
                        tensor = torch.from_numpy(arr).to("xpu")
                        frame = Frame(tensor, ow, oh)
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
