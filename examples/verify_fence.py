"""Verify the GPU-to-GPU fence path: correctness + whether it stayed active."""
import argparse
import ctypes
import threading
import time

import torch

from wc_rocm import WindowsCapture


def _activity(stop_event):
    user32 = ctypes.windll.user32
    def run():
        x = 0
        while not stop_event.is_set():
            x ^= 1
            user32.SetCursorPos(10 + x, 10)
            time.sleep(0.002)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--legacy", action="store_true", help="Force use_gpu_fence=False")
    args = parser.parse_args()

    stop = threading.Event()
    _activity(stop)

    cap = WindowsCapture(
        monitor_index=args.monitor_index,
        device_id=args.device_id,
        use_gpu_fence=not args.legacy,
        synchronize_copy=False,
    )

    state = {"n": 0, "mean": None, "shape": None, "nonzero": False}

    @cap.event
    def on_frame_arrived(frame, control):
        state["n"] += 1
        if state["n"] == 5:
            buf = frame.frame_buffer
            state["shape"] = (frame.width, frame.height)
            m = buf.float().mean().item()
            state["mean"] = m
            state["nonzero"] = m > 0.0
        if state["n"] >= args.frames:
            control.stop()

    @cap.event
    def on_closed():
        pass

    cap.start()
    stop.set()

    print("=== GPU fence verification ===")
    print(f"requested mode    : {'legacy CPU-wait' if args.legacy else 'GPU fence'}")
    print(f"final use_gpu_fence: {cap.use_gpu_fence}  (False here means it fell back)")
    print(f"frames received   : {state['n']}")
    print(f"sample resolution : {state['shape']}")
    print(f"sample mean value : {state['mean']}")
    print(f"frame non-zero    : {state['nonzero']}")
    if not args.legacy and cap.use_gpu_fence and state["nonzero"]:
        print("RESULT: GPU-to-GPU fence ACTIVE and producing valid frames. ✔")
    elif not args.legacy and not cap.use_gpu_fence:
        print("RESULT: GPU fence unsupported on this system; fell back to CPU-wait.")
    else:
        print("RESULT: legacy CPU-wait path validated.")


if __name__ == "__main__":
    main()
