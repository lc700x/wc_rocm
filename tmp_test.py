from wc_xpu import WindowsCapture
import time, sys
capture = WindowsCapture(monitor_index=1)
capture._inner.start()
sys.stdout.flush()
time.sleep(5)
print(f"alive: {capture._inner.is_alive()}")
sys.stdout.flush()
print(f"error: {repr(capture._inner.get_last_error())}")
sys.stdout.flush()
capture._inner.stop()
