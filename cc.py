import torch
import os

def diagnose_gpu_issues():
    print("\n===== GPU DIAGNOSTICS =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Max memory cached: {torch.cuda.max_memory_cached() / 1e9:.2f} GB")
    
    print("\n===== DATASET CHECK =====")
    dataset_paths = ['Brain_Stroke_CT_SCAN_image', 'Dataset']
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"Dataset path exists: {path}")
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            print(f"Subdirectories: {subdirs}")
            
            # Check for Train/Test/Val
            for subdir in ['Train', 'Test', 'Validation']:
                subpath = os.path.join(path, subdir)
                if os.path.exists(subpath):
                    print(f"  - {subdir} exists with {len(os.listdir(subpath))} items")
                    class_dirs = [d for d in os.listdir(subpath) if os.path.isdir(os.path.join(subpath, d))]
                    print(f"    Classes: {class_dirs}")
                else:
                    print(f"  - {subdir} does not exist")
        else:
            print(f"Dataset path does not exist: {path}")
    
    print("\n===== MEMORY TEST =====")
    try:
        # Try to allocate a large tensor to see if memory is working
        test_tensor = torch.zeros(1000, 1000, 50, device='cuda')
        print(f"Successfully allocated test tensor of shape {test_tensor.shape}")
        print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        del test_tensor
        torch.cuda.empty_cache()
        print("Test tensor deleted and cache cleared")
    except RuntimeError as e:
        print(f"Failed to allocate test tensor: {e}")
    
    print("\n===== SUGGESTED FIXES =====")
    print("1. Try reducing batch size: BATCH_SIZE = 16 or 8")
    print("2. Enable mixed precision: use torch.cuda.amp.autocast")
    print("3. Reduce image size: IMG_SIZE = (192, 192)")
    print("4. Check power settings and ensure laptop is plugged in")
    print("5. Update NVIDIA drivers")
    print("6. Set environment variable: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")

# Run diagnostics
diagnose_gpu_issues()