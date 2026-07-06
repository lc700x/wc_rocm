import time
from wc_xpu import WindowsCapture


def test_stress_restart():
    print("\n--- Starting Stress Test: Restart (5 cycles) ---")
    for i in range(5):
        print(f"  Cycle {i + 1}/5...")
        capture = WindowsCapture(monitor_index=1)
        frame_received = False

        @capture.event
        def on_frame_arrived(frame, control):
            nonlocal frame_received
            frame_received = True
            control.stop()

        capture.start()
        if frame_received:
            print("    Frame received.")
        else:
            print("    [ERROR] No frame received.")
            break
        time.sleep(0.1)
    print("  [OK] Stress test completed.")


if __name__ == "__main__":
    test_stress_restart()
