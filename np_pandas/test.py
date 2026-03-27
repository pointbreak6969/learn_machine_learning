import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {device}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version with CUDA: {torch.version.cuda}")
else:
    print("CUDA is not available. Using CPU.")

# Example of creating a tensor on the GPU if available
if torch.cuda.is_available():
    x = torch.randn(2, 3, device=device)
    print("\nExample tensor allocated on GPU:")
    print(x)
    print(f"Is tensor on CUDA? {x.is_cuda}")
