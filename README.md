# WindowsCaptureGPU wrapper for Python

A wrapper around [windows-capture](https://github.com/NiiightmareXD/windows-capture) that captures the screen directly into PyTorch CUDA tensors, avoiding CPU memory copies and inter-device memory transfers.
[rocm] leverages DXGI and HIP_RUNTIME from ROCm7.
[cuda] leverages D3D11 and CUDA_RUNTIME from CUDA. 

To avoid reinstalling torch, it's not listed as a dependency, but a ROCm7-enabled PyTorch installation is required. It only works on x64 Windows with an AMD GPU (RDNA1 and later) or an NVIDIA GPU (10 Series and later).

# AI generated

This was written by **Claude Code**. 
