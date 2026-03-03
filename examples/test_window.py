from wc_cuda import WindowsCapture, list_windows
from PIL import Image


def test_window_capture():
    print("\n--- Testing Window Capture ---")
    windows = list_windows()
    target_title = next((t for _, t in windows if any(x in t for x in ["PowerShell", "Prompt", "Code"])), None)
    if not target_title and windows:
        target_title = windows[0][1]
    if not target_title:
        print("  [SKIP] No visible windows found.")
        return
    print(f"  Targeting window: '{target_title}'")
    capture = WindowsCapture(window_name=target_title)
    frame_count = 0
    saved = False

    @capture.event
    def on_frame_arrived(frame, control):
        nonlocal frame_count, saved
        frame_count += 1
        if not saved:
            rgba = frame.frame_buffer[..., [2, 1, 0, 3]]
            Image.fromarray(rgba.cpu().numpy()).save("captured_window.png")
            print(f"  [OK] Saved captured_window.png ({frame.width}x{frame.height})")
            saved = True
        if frame_count >= 10:
            control.stop()

    capture.start()


if __name__ == "__main__":
    test_window_capture()
