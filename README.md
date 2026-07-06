# wc_xpu

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch XPU tensors on Intel GPUs, avoiding CPU memory copies and inter-device memory transfers.

Requires a PyTorch XPU build (`pip install torch --extra-index-url https://download.pytorch.org/whl/xpu`). Only works on x64 Windows with Intel Arc or Intel Iris Xe graphics.

## Usage

```python
from wc_xpu import WindowsCapture

capture = WindowsCapture(
    monitor_index=1,
    reuse_output_buffer=True,
    output_buffer_count=4,
)
```

When buffer reuse is enabled, tensors may be overwritten after several frames (ring-buffer behavior), so clone tensors in your callback if you need to keep them long-term.

## Credits

This was written by **gemini-cli** and **Claude Code**. Originally written by NiiightmareXD [windows-capture-python](https://github.com/NiiightmareXD/windows-capture/tree/main/windows-capture-python) and forked from Nagadomi [wc_cuda](https://github.com/nagadomi/wc_cuda), then further optimized by **Claude Code**.
