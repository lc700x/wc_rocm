# Windows Capture GPU Wrapper for ROCm and CUDA

## 1. wc_rocm

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers.

To avoid reinstalling torch, it's not listed as a dependency, but a ROCm-enabled PyTorch installation is required. It only works on x64 Windows with an AMD GPU.


### Usage  
Example:

```python
from wc_rocm import WindowsCapture

capture = WindowsCapture(
    monitor_index=1,
    device_id=0,
    reuse_output_buffer=True,
    output_buffer_count=4,
)
```

When buffer reuse is enabled, tensors may be overwritten after several frames (ring-buffer behavior), so clone tensors in your callback if you need to keep them long-term.

## 2. wc_cuda

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers.

To avoid reinstalling torch, it's not listed as a dependency, but a CUDA-enabled PyTorch installation is required. It only works on x64 Windows with an NVIDIA GPU.

### High-FPS mode (optional)

`WindowsCapture` now accepts:

- `reuse_output_buffer` (default: `False`)
- `output_buffer_count` (default: `3`)

### Usage
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

This was written by **gemini-cli** and **Claude Code**. Originally written by NiiightmareXD [windows-capture-python](https://github.com/NiiightmareXD/windows-capture/tree/main/windows-capture-python) and forked from Nagadomi [wc_cuda](https://github.com/nagadomi/wc_cuda), then further optimized by **Claude Code**. Please report any issues or inaccuracies.
