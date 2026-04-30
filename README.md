# wc_rocm

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers leveraged with DXGI and HIP_RUNTIME from ROCm7.

To avoid reinstalling torch, it's not listed as a dependency, but a ROCm7-enabled PyTorch installation is required. It only works on x64 Windows with an AMD GPU.

# AI generated

This was written by claude code.