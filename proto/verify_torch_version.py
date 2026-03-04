import torch
# Check if PyTorch was built with CUDA support
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

# If CUDA is available, print more details
if torch.cuda.is_available():
    print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}") 
else:
    print("Torch has not been installed with cuda support.")
    print("Please follow NOTE 2 in the README to fix this issue.")