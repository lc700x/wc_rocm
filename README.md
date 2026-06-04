# wc_cuda

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers.

To avoid reinstalling torch, it's not listed as a dependency, but a CUDA-enabled PyTorch installation is required. It only works on x64 Windows with an NVIDIA GPU.

## Performance notes

Recent optimizations focused on improving frame throughput:

- Replaced the per-frame CPU-blocking CUDA stream sync (`stream.synchronize()`)
  with a non-blocking cross-stream dependency (`wait_stream`), so the capture
  loop no longer idles the CPU waiting for each device-to-device copy.
- Added optional CUDA output-buffer reuse (a small ring buffer) to remove a
  `cudaMalloc`/`cudaFree` per frame.
- Reduced capture-handoff overhead: single-waiter `notify_one` and relaxed
  atomic ordering on the stop flag.

Correctness note: the Direct3D `Flush()` after the shared-texture copy is kept
intentionally. It is a non-blocking GPU submit (not a stall), and it is required
so the CUDA interop map observes the copied frame.

## High-FPS mode (optional)

`WindowsCapture` now accepts:

- `reuse_output_buffer` (default: `False`)
- `output_buffer_count` (default: `3`)

Example:

```python
from wc_cuda import WindowsCapture

capture = WindowsCapture(
    monitor_index=1,
    device_id=0,
    reuse_output_buffer=True,
    output_buffer_count=4,
)
```

When buffer reuse is enabled, tensors may be overwritten after several frames (ring-buffer behavior), so clone tensors in your callback if you need to keep them long-term.

## AI generated

This was written by gemini-cli and claude code.
