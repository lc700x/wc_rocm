import argparse
import time

from wc_rocm import WindowsCapture


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure wc_rocm capture FPS")
    parser.add_argument("--seconds", type=float, default=8.0, help="Benchmark duration")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup duration")
    parser.add_argument("--monitor-index", type=int, default=1, help="Monitor index for capture")
    parser.add_argument("--device-id", type=int, default=0, help="ROCm device id")
    parser.add_argument(
        "--synchronize-copy",
        action="store_true",
        help="Synchronize stream every frame (lower FPS, stronger readiness guarantee)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Disable GPU-to-GPU fence; use the legacy Rust CPU busy-wait path",
    )
    args = parser.parse_args()

    capture = WindowsCapture(
        monitor_index=args.monitor_index,
        device_id=args.device_id,
        synchronize_copy=args.synchronize_copy,
        use_gpu_fence=not args.legacy,
    )

    stats = {
        "frames": 0,
        "warmup_end": 0.0,
        "run_end": 0.0,
        "first_shape": None,
        "cpu0": 0.0,
    }

    @capture.event
    def on_frame_arrived(frame, control):
        now = time.perf_counter()
        if stats["warmup_end"] == 0.0:
            stats["warmup_end"] = now + args.warmup
            stats["run_end"] = stats["warmup_end"] + args.seconds
            stats["first_shape"] = (frame.width, frame.height)

        if now < stats["warmup_end"]:
            return

        if stats["frames"] == 0:
            stats["cpu0"] = time.process_time()

        stats["frames"] += 1

        if now >= stats["run_end"]:
            stats["cpu1"] = time.process_time()
            control.stop()

    @capture.event
    def on_closed():
        pass

    t0 = time.perf_counter()
    capture.start()
    elapsed = time.perf_counter() - t0

    measured = max(args.seconds, 1e-6)
    fps = stats["frames"] / measured
    cpu_used = stats.get("cpu1", stats.get("cpu0", 0.0)) - stats.get("cpu0", 0.0)
    cpu_ms_per_frame = (cpu_used / stats["frames"] * 1000.0) if stats["frames"] else 0.0
    print("Benchmark complete")
    print(f"mode={'legacy CPU-wait' if args.legacy else 'GPU fence'}  synchronize_copy={args.synchronize_copy}")
    print(f"resolution={stats['first_shape']}")
    print(f"warmup_s={args.warmup:.3f} test_s={args.seconds:.3f} wall_s={elapsed:.3f}")
    print(f"frames={stats['frames']} fps={fps:.2f}")
    print(f"process_cpu_s={cpu_used:.3f}  cpu_ms_per_frame={cpu_ms_per_frame:.3f}")


if __name__ == "__main__":
    main()
