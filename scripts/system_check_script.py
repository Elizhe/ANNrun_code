#!/usr/bin/env python3
"""
System Check Script for Windows
Check system capabilities and recommend optimal settings
"""

import platform
import multiprocessing as mp

def check_system():
    print("=== System Check for ANNrun_code (Windows) ===\n")
    
    # System information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # CPU information
    cpu_count = mp.cpu_count()
    print(f"\nCPU Cores: {cpu_count}")
    
    # Memory information (simple version for Windows)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Memory Usage: {memory.percent:.1f}%")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent:.1f}%")
        
    except ImportError:
        print("WARNING: psutil not installed - install with: pip install psutil")
        print("RAM info: Not available without psutil")
    
    # Framework availability
    print("\n=== Framework Availability ===")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"SUCCESS: TensorFlow: {tf.__version__}")
        
        # GPU check
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   GPUs detected: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"     GPU {i}: {gpu.name}")
            else:
                print("   No GPUs detected")
        except Exception as e:
            print(f"   GPU check failed: {e}")
            
    except ImportError:
        print("MISSING: TensorFlow not available")
        print("   Install with: pip install tensorflow")
    
    # PyTorch
    try:
        import torch
        print(f"SUCCESS: PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA available: Yes")
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   CUDA available: No")
    except ImportError:
        print("MISSING: PyTorch not available")
        print("   Install with: pip install torch")
    
    # Parallel processing libraries
    try:
        from joblib import Parallel
        import joblib
        print(f"SUCCESS: Joblib: {joblib.__version__}")
    except ImportError:
        print("MISSING: Joblib not available")
        print("   Install with: pip install joblib")
    
    try:
        import ray
        print(f"SUCCESS: Ray: {ray.__version__}")
    except ImportError:
        print("OPTIONAL: Ray not available")
        print("   Install with: pip install ray")
    
    # Core scientific libraries
    print("\n=== Core Libraries ===")
    
    libraries = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    for module_name, display_name in libraries:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"SUCCESS: {display_name}: {version}")
        except ImportError:
            print(f"MISSING: {display_name} not available")
    
    # Windows-specific recommendations
    print("\n=== Windows Optimization Recommendations ===")
    
    if cpu_count >= 8:
        recommended_jobs = min(cpu_count - 1, 8)
        print(f"SUCCESS: Good CPU count for parallel processing: {cpu_count} cores")
        print(f"   Recommended --n-jobs: {recommended_jobs}")
    elif cpu_count >= 4:
        recommended_jobs = cpu_count - 1
        print(f"OK: Moderate CPU count: {cpu_count} cores")
        print(f"   Recommended --n-jobs: {recommended_jobs}")
    else:
        recommended_jobs = 1
        print(f"WARNING: Limited CPU cores: {cpu_count}")
        print(f"   Recommended --n-jobs: {recommended_jobs} (single-threaded)")
    
    # Memory recommendations
    try:
        if 'memory' in locals():
            available_gb = memory.available / (1024**3)
            if available_gb >= 16:
                print("SUCCESS: Sufficient RAM for large experiments")
            elif available_gb >= 8:
                print("WARNING: Moderate RAM - consider smaller batch sizes")
            else:
                print("WARNING: Low RAM - use single experiments or very small batches")
    except:
        pass
    
    # Windows-specific tips
    print("\n=== Windows-Specific Tips ===")
    print("1. Use PowerShell or Command Prompt as Administrator for best performance")
    print("2. Close unnecessary applications to free up memory")
    print("3. Consider using Windows Subsystem for Linux (WSL) for better compatibility")
    print("4. Joblib is more stable than Ray on Windows")
    
    # Example commands
    print("\n=== Recommended Commands for Windows ===")
    print(f"# Single experiment:")
    print(f"python main.py single experiment_plan.csv 1")
    print(f"")
    print(f"# Parallel experiment (Joblib - recommended):")
    print(f"python main.py parallel experiment_plan.csv 1 --parallel-backend joblib --n-jobs {recommended_jobs}")
    print(f"")
    print(f"# Parallel experiment (Multiprocessing):")
    print(f"python main.py parallel experiment_plan.csv 1 --parallel-backend multiprocessing --n-jobs {recommended_jobs}")
    
    # Check current directory structure
    print("\n=== ANNrun_code Structure Check ===")
    
    import os
    from pathlib import Path
    
    expected_dirs = [
        'core',
        'experiments', 
        'configs',
        'scripts'
    ]
    
    expected_files = [
        'main.py',
        'setup.py',
        'requirements.txt'
    ]
    
    for directory in expected_dirs:
        if Path(directory).exists():
            print(f"SUCCESS: Directory {directory}/ exists")
        else:
            print(f"MISSING: Directory {directory}/ not found")
    
    for file in expected_files:
        if Path(file).exists():
            print(f"SUCCESS: File {file} exists")
        else:
            print(f"MISSING: File {file} not found")
    
    # Final status
    print("\n=== Setup Status ===")
    
    # Check if core modules exist
    core_modules = [
        'core/data/data_manager.py',
        'core/models/neural_networks/builders.py',
        'experiments/parallel_runner.py',
        'main.py'
    ]
    
    missing_modules = []
    for module in core_modules:
        if not Path(module).exists():
            missing_modules.append(module)
    
    if missing_modules:
        print("WARNING: Missing core modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nNext steps:")
        print("1. Add the missing core module files")
        print("2. Run: pip install -e .")
        print("3. Test with: python main.py --help")
    else:
        print("SUCCESS: All core modules present!")
        print("Ready to run experiments!")


if __name__ == "__main__":
    try:
        check_system()
    except KeyboardInterrupt:
        print("\nSystem check interrupted by user")
    except Exception as e:
        print(f"\nError during system check: {e}")
        print("This is likely due to missing dependencies")
        print("Try: pip install numpy pandas scikit-learn psutil")
