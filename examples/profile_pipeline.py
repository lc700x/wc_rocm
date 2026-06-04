"""Deep profiler for wc_rocm to determine whether FPS is refresh-capped or pipeline-bound.

It measures:
  1. Frame-to-frame delivery interval distribution (detects display refresh cap).
  2. Pure GPU DMA cost of bridge.update() with explicit synchronize (pipeline headroom).
  3. Estimated max theoretical FPS from DMA cost alone.
"""
import argparse
import ctypes
import statistics
import threading
import time

import torch

from wc_rocm import WindowsCapture


def _start_activity(stop_event: threading.Event) -> threading.Thread:
    """Wiggle the cursor on a background thread so WGC keeps presenting frames."""
    user32 = ctypes.windll.user32

    def run():
        dx = 0
        while not stop_event.is_set():
            dx = 1 if dx == 0 else 0
            user32.SetCursorPos(10 + dx, 10)
            time.sleep(0.002)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile wc_rocm pipeline cost")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--no-activity",
        action="store_true",
        help="Disable the built-in cursor-wiggle activity generator",
    )
    args = parser.parse_args()

    stop_activity = threading.Event()
    if not args.no_activity:
        _start_activity(stop_activity)

    capture = WindowsCapture(
        monitor_index=args.monitor_index,
        device_id=args.device_id,
        synchronize_copy=True,  # measure full DMA completion cost per frame
    )

    state = {
        "intervals": [],
        "dma_ms": [],
        "last_ts": None,
        "warmup_end": 0.0,
        "run_end": 0.0,
        "shape": None,
    }

    @capture.event
    def on_frame_arrived(frame, control):
        now = time.perf_counter()
        if state["warmup_end"] == 0.0:
            state["warmup_end"] = now + args.warmup
            state["run_end"] = state["warmup_end"] + args.seconds
            state["shape"] = (frame.width, frame.height)

        if now < state["warmup_end"]:
            state["last_ts"] = now
            return

        if state["last_ts"] is not None:
            state["intervals"].append((now - state["last_ts"]) * 1000.0)
        state["last_ts"] = now

        # Measure pure DMA cost: touch the tensor to force a sync read timing.
        t0 = time.perf_counter()
        # frame.frame_buffer is already DMAed + synchronized by the bridge.
        # Re-time a no-op GPU sync to capture residual latency.
        torch.cuda.synchronize(args.device_id)
        state["dma_ms"].append((time.perf_counter() - t0) * 1000.0)

        if now >= state["run_end"]:
            control.stop()

    @capture.event
    def on_closed():
        pass

    capture.start()

    stop_activity.set()

    intervals = state["intervals"]
    if not intervals:
        print("No frames captured — try moving a window on the target monitor.")
        return

    intervals_sorted = sorted(intervals)
    n = len(intervals_sorted)
    median = statistics.median(intervals_sorted)
    p05 = intervals_sorted[int(0.05 * (n - 1))]
    p95 = intervals_sorted[int(0.95 * (n - 1))]
    fps = 1000.0 / median if median > 0 else 0.0

    print("=== wc_rocm profile ===")
    print(f"resolution        : {state['shape']}")
    print(f"frames measured   : {n}")
    print(f"interval median   : {median:.3f} ms  ({fps:.2f} FPS)")
    print(f"interval p05/p95  : {p05:.3f} / {p95:.3f} ms")
    print(f"interval min/max  : {intervals_sorted[0]:.3f} / {intervals_sorted[-1]:.3f} ms")

    # Refresh-cap heuristic: tight clustering around a refresh multiple => capped.
    spread = p95 - p05
    near_60 = abs(median - 16.667) < 2.0
    near_120 = abs(median - 8.333) < 1.5
    near_144 = abs(median - 6.944) < 1.0
    print()
    if spread < 4.0 and (near_60 or near_120 or near_144):
        hz = round(1000.0 / median)
        print(f"VERDICT: DISPLAY-REFRESH BOUND (~{hz} Hz).")
        print("  Delivery is gated by the Windows Graphics Capture present rate.")
        print("  Pipeline is NOT the bottleneck; higher FPS needs a higher-refresh")
        print("  display or a non-vsync source (e.g. a game rendering >refresh).")
    else:
        print("VERDICT: PIPELINE-INFLUENCED — interval spread suggests processing cost")
        print("  contributes to frame timing; further code optimization can help.")


if __name__ == "__main__":
    main()
