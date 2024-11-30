import torch

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("Is CUDA available:", cuda_available)

# Print CUDA version used by PyTorch
if cuda_available:
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("CUDA devices available:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available in PyTorch")
