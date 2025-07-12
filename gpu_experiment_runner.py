#!/usr/bin/env python3
"""
GPU Experiment Runner for RTX 2060 SUPER
Runs 216 experiment combinations with GPU optimization
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class GPUExperimentRunner:
    """Main class for running GPU-optimized experiments"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.start_time = None
        self.experiment_count = 0
        self.results_dir = Path("results_gpu")
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup environment
        self.setup_gpu_environment()
    
    def setup_gpu_environment(self):
        """Setup GPU environment variables"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        
        # Memory optimization
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        print(f"✅ GPU environment configured for GPU {self.gpu_id}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("=== Checking Prerequisites ===")
        
        issues = []
        
        # Check GPU availability
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=10)
            if result.returncode != 0:
                issues.append("nvidia-smi not working")
        except:
            issues.append("nvidia-smi not found")
        
        # Check experiment plan file
        if not Path("gpu_experiment_plan.csv").exists():
            issues.append("gpu_experiment_plan.csv not found")
        
        # Check main.py
        if not Path("main.py").exists():
            issues.append("main.py not found")
        
        # Check TensorFlow/PyTorch
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                issues.append("TensorFlow GPU not available")
        except ImportError:
            issues.append("TensorFlow not installed")
        
        if issues:
            print("❌ Prerequisites not met:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ All prerequisites met")
        return True
    
    def run_experiment(self, parallel_backend: str = "joblib", n_jobs: int = 1) -> bool:
        """Run the main experiment"""
        
        if not self.check_prerequisites():
            return False
        
        print("=== Starting GPU Experiment ===")
        print(f"📊 Expected combinations: 216 (12 arch × 9 reg × 2 adam)")
        print(f"🎯 GPU: RTX 2060 SUPER (Device {self.gpu_id})")
        print(f"⚡ Backend: {parallel_backend}")
        print(f"🔧 Jobs: {n_jobs}")
        
        self.start_time = time.time()
        
        # Construct command
        cmd = [
            sys.executable, "main.py", "parallel",
            "gpu_experiment_plan.csv", "1",
            "--parallel-backend", parallel_backend,
            "--n-jobs", str(n_jobs),
            "--gpu-ids", str(self.gpu_id),
            "--output-dir", str(self.results_dir),
            "--log-level", "INFO"
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        print("=" * 60)
        
        try:
            # Run experiment with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Save log file
            log_file = self.results_dir / f"gpu_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                for line in process.stdout:
                    print(line.rstrip())  # Print to console
                    f.write(line)  # Write to file
                    f.flush()
            
            # Wait for completion
            return_code = process.wait()
            
            elapsed_time = time.time() - self.start_time
            
            if return_code == 0:
                print("=" * 60)
                print("✅ Experiment completed successfully!")
                print(f"⏱️  Total time: {elapsed_time/3600:.1f} hours")
                print(f"📁 Results saved to: {self.results_dir}")
                print(f"📄 Log saved to: {log_file}")
                return True
            else:
                print("=" * 60)
                print(f"❌ Experiment failed with return code: {return_code}")
                print(f"📄 Check log file: {log_file}")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️  Experiment interrupted by user")
            if hasattr(process, 'terminate'):
                process.terminate()
            return False
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            return False
    
    def monitor_gpu_usage(self, interval: int = 30):
        """Monitor GPU usage during experiment"""
        print(f"\n=== GPU Monitoring (every {interval}s) ===")
        
        try:
            while True:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    data = result.stdout.strip().split(', ')
                    memory_used = int(data[0])
                    memory_total = int(data[1])
                    utilization = int(data[2])
                    temperature = int(data[3])
                    
                    memory_percent = (memory_used / memory_total) * 100
                    
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    
                    print(f"[{elapsed/3600:.1f}h] 💻 Mem: {memory_percent:.1f}% ({memory_used}MB) | "
                          f"GPU: {utilization}% | Temp: {temperature}°C")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  GPU monitoring stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GPU Experiment Runner for RTX 2060 SUPER")
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--backend', choices=['joblib', 'multiprocessing'], 
                       default='joblib', help='Parallel backend')
    parser.add_argument('--n-jobs', type=int, default=1, 
                       help='Number of parallel jobs (1 recommended for GPU)')
    parser.add_argument('--monitor-only', action='store_true', 
                       help='Only run GPU monitoring')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check prerequisites')
    
    args = parser.parse_args()
    
    # Create runner
    runner = GPUExperimentRunner(gpu_id=args.gpu_id)
    
    if args.check_only:
        success = runner.check_prerequisites()
        sys.exit(0 if success else 1)
    
    if args.monitor_only:
        print("Starting GPU monitoring mode...")
        print("Press Ctrl+C to stop")
        runner.start_time = time.time()
        runner.monitor_gpu_usage()
        return
    
    # Run full experiment
    print("=== GPU Experiment Runner ===")
    print("Target: 216 Adam-only combinations on RTX 2060 SUPER")
    
    # Option to run monitoring in background
    monitor_response = input("Run GPU monitoring in background? (y/n): ").lower()
    
    if monitor_response == 'y':
        import threading
        runner.start_time = time.time()
        monitor_thread = threading.Thread(target=runner.monitor_gpu_usage, args=(30,))
        monitor_thread.daemon = True
        monitor_thread.start()
        print("✅ GPU monitoring started in background")
    
    # Run experiment
    success = runner.run_experiment(
        parallel_backend=args.backend,
        n_jobs=args.n_jobs
    )
    
    if success:
        print("\n🎉 All experiments completed!")
        print("📊 Check results in: results_gpu/")
        print("📈 Run analysis: python analyze_results.py results_gpu/")
    else:
        print("\n💥 Experiment failed!")
        print("🔍 Check the log files for details")
        sys.exit(1)

if __name__ == "__main__":
    main()