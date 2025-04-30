import torch

# Create test CUDA tensors and operations that require compilation
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test operation that requires CUDA compilation
    try:
        # This operation forces compilation of CUDA code
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = torch.matmul(x, y)
        print("CUDA compilation SUCCESS - operation completed")
        print(f"Result shape: {z.shape}")
    except Exception as e:
        print(f"CUDA compilation FAILED: {e}")