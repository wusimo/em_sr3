import torch

print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)         # Check the CUDA version PyTorch is built with
print(torch.backends.cudnn.is_available())  # Check if cuDNN is available
