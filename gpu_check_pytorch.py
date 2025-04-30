import sys
import torch
import platform
import subprocess
import os

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Error executing {command}: {e.output}"

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"CUDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check NVIDIA driver and CUDA
if platform.system() == "Windows":
    print("\nNVIDIA SMI output:")
    print(run_command("nvidia-smi"))
    print("\nNVCC version:")
    print(run_command("nvcc --version"))
else:
    print("\nNVIDIA SMI output:")
    print(run_command("nvidia-smi"))
    print("\nNVCC version:")
    print(run_command("nvcc --version"))

# Check environment variables
cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
print(f"\nCUDA_PATH/CUDA_HOME: {cuda_path}")

# Try to import a CUDA tensor
try:
    x = torch.cuda.FloatTensor(1)
    print("\nSuccessfully created a CUDA tensor!")
except Exception as e:
    print(f"\nFailed to create a CUDA tensor: {e}")