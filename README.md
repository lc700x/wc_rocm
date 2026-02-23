# wc_cuda

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers.

To avoid reinstalling torch, it's not listed as a dependency, but a CUDA-enabled PyTorch installation is required. It only works on x64 Windows with an NVIDIA GPU.

# AI generated

This was written by gemini-cli.
