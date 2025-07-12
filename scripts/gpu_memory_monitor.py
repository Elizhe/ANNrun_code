#!/usr/bin/env python3
"""
GPU Memory Monitor and Optimization for RTX 2060 SUPER
Checks GPU status and optimizes memory usage for experiments
"""

import os
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class GPUMemoryMonitor:
    """Monitor and optimize GPU memory usage"""
    
    def __init__(self):
        self.gpu_info = {}
        self.memory_history = []
        
    def check_gpu_availability(self) -> Dict:
        """Check GPU availability and get detailed information"""
        gpu_status = {
            'available': False,
            'frameworks': {},
            'hardware': {}
        }
        
        # Check NVIDIA GPU with nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_data = lines[0].split(', ')
                    gpu_status['hardware'] = {
                        'name': gpu_data[0],
                        'total_memory_mb': int(gpu_data[1]),
                        'used_memory_mb': int(gpu_data[2]),
                        'free_memory_mb': int(gpu_data[3]),
                        'temperature_c': int(gpu_data[4]),
                        'utilization_percent': int(gpu_data[5])
                    }
                    gpu_status['available'] = True
                    print(f"✅ GPU detected: {gpu_data[0]}")
                    print(f"   Memory: {gpu_data[3]}MB free / {gpu_data[1]}MB total")
                    print(f"   Temperature: {gpu_data[4]}°C, Utilization: {gpu_data[5]}%")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"❌ nvidia-smi check failed: {e}")
        
        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            gpu_status['frameworks']['tensorflow'] = {
                'available': len(gpus) > 0,
                'gpu_count': len(gpus),
                'version': tf.__version__
            }
            
            if len(gpus) > 0:
                print(f"✅ TensorFlow GPU support: {len(gpus)} GPU(s)")
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("❌ TensorFlow: No GPU devices found")
                
        except ImportError:
            print("❌ TensorFlow not installed")
            gpu_status['frameworks']['tensorflow'] = {'error': 'not installed'}
        except Exception as e:
            print(f"❌ TensorFlow GPU check failed: {e}")
            gpu_status['frameworks']['tensorflow'] = {'error': str(e)}
        
        # Check PyTorch GPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_status['frameworks']['pytorch'] = {
                'available': gpu_available,
                'gpu_count': torch.cuda.device_count() if gpu_available else 0,
                'version': torch.__version__
            }
            
            if gpu_available:
                print(f"✅ PyTorch GPU support: {torch.cuda.device_count()} GPU(s)")
                print(f"   Current device: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
            else:
                print("❌ PyTorch: CUDA not available")
                
        except ImportError:
            print("❌ PyTorch not installed")
            gpu_status['frameworks']['pytorch'] = {'error': 'not installed'}
        except Exception as e:
            print(f"❌ PyTorch GPU check failed: {e}")
            gpu_status['frameworks']['pytorch'] = {'error': str(e)}
        
        return gpu_status
    
    def optimize_tensorflow_gpu(self):
        """Optimize TensorFlow for RTX 2060 SUPER"""
        try:
            import tensorflow as tf
            
            # Configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit to 7.5GB (leave 0.5GB for system)
                tf.config.experimental.set_memory_limit(gpus[0], 7680)
                
                print("✅ TensorFlow GPU optimized:")
                print("   - Memory growth enabled")
                print("   - Memory limit set to 7.5GB")
                
                # Enable mixed precision for memory efficiency
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("   - Mixed precision (FP16) enabled")
                
                return True
        except Exception as e:
            print(f"❌ TensorFlow optimization failed: {e}")
            return False
    
    def optimize_pytorch_gpu(self):
        """Optimize PyTorch for RTX 2060 SUPER"""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                
                # Empty cache
                torch.cuda.empty_cache()
                
                print("✅ PyTorch GPU optimized:")
                print("   - Memory fraction set to 90%")
                print("   - Cache cleared")
                
                # Enable mixed precision training
                print("   - Mixed precision available via torch.cuda.amp")
                
                return True
        except Exception as e:
            print(f"❌ PyTorch optimization failed: {e}")
            return False
    
    def get_optimal_batch_sizes(self) -> Dict[str, int]:
        """Get optimal batch sizes for different architectures on RTX 2060 SUPER"""
        return {
            # Single layer networks - can handle larger batches
            'single_32': 1024,
            'single_64': 512,
            'single_128': 256,
            
            # Double layer networks
            'double_32_16': 512,
            'double_64_32': 256,
            'double_128_64': 128,
            'double_64_64': 256,
            
            # Triple layer networks
            'triple_128_64_32': 128,
            'triple_64_32_16': 256,
            'triple_32_32_16': 256,
            
            # Deep networks - need smaller batches
            'deep_256_128_64_32': 64,
            'deep_128_128_64_32': 128
        }
    
    def monitor_during_training(self, duration_seconds: int = 60):
        """Monitor GPU usage during training"""
        print(f"\n=== Monitoring GPU for {duration_seconds} seconds ===")
        
        start_time = time.time()
        samples = []
        
        try:
            while time.time() - start_time < duration_seconds:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    data = result.stdout.strip().split(', ')
                    sample = {
                        'timestamp': time.time(),
                        'memory_used_mb': int(data[0]),
                        'memory_total_mb': int(data[1]),
                        'utilization_percent': int(data[2]),
                        'temperature_c': int(data[3])
                    }
                    samples.append(sample)
                    
                    memory_percent = (sample['memory_used_mb'] / sample['memory_total_mb']) * 100
                    print(f"\r💻 Memory: {memory_percent:.1f}% | GPU: {sample['utilization_percent']}% | Temp: {sample['temperature_c']}°C", end='')
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        
        if samples:
            avg_memory = sum(s['memory_used_mb'] for s in samples) / len(samples)
            avg_util = sum(s['utilization_percent'] for s in samples) / len(samples)
            max_temp = max(s['temperature_c'] for s in samples)
            
            print(f"\n📊 Monitoring Summary:")
            print(f"   Average memory usage: {avg_memory:.0f}MB ({avg_memory/8192*100:.1f}%)")
            print(f"   Average GPU utilization: {avg_util:.1f}%")
            print(f"   Peak temperature: {max_temp}°C")
        
        return samples
    
    def run_memory_stress_test(self):
        """Run a quick memory stress test to find optimal settings"""
        print("\n=== GPU Memory Stress Test ===")
        
        try:
            import tensorflow as tf
            
            # Test different batch sizes
            test_sizes = [64, 128, 256, 512, 1024]
            results = {}
            
            for batch_size in test_sizes:
                try:
                    print(f"Testing batch size: {batch_size}")
                    
                    # Create a simple model
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(1)
                    ])
                    
                    # Generate test data
                    x_test = tf.random.normal((batch_size, 8))
                    y_test = tf.random.normal((batch_size, 1))
                    
                    # Test forward pass
                    with tf.device('/GPU:0'):
                        prediction = model(x_test)
                    
                    # Get memory usage
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                    memory_used = int(result.stdout.strip())
                    
                    results[batch_size] = {
                        'success': True,
                        'memory_used_mb': memory_used
                    }
                    print(f"   ✅ Success - Memory used: {memory_used}MB")
                    
                    # Clean up
                    del model, x_test, y_test, prediction
                    tf.keras.backend.clear_session()
                    
                except tf.errors.ResourceExhaustedError:
                    results[batch_size] = {'success': False, 'error': 'Out of memory'}
                    print(f"   ❌ Out of memory")
                    break
                except Exception as e:
                    results[batch_size] = {'success': False, 'error': str(e)}
                    print(f"   ❌ Error: {e}")
            
            # Find optimal batch size
            successful_sizes = [size for size, result in results.items() if result['success']]
            if successful_sizes:
                optimal_size = max(successful_sizes)
                print(f"\n🎯 Recommended batch size: {optimal_size}")
            else:
                print("\n❌ No successful batch sizes found")
            
            return results
            
        except ImportError:
            print("❌ TensorFlow not available for stress test")
            return {}

def main():
    """Main function"""
    print("=== GPU Memory Monitor & Optimizer ===")
    print("Target: RTX 2060 SUPER (8GB VRAM)")
    
    monitor = GPUMemoryMonitor()
    
    # Check GPU availability
    print("\n1. Checking GPU availability...")
    gpu_status = monitor.check_gpu_availability()
    
    if not gpu_status['available']:
        print("❌ No GPU detected. Please check:")
        print("   - NVIDIA drivers installed")
        print("   - CUDA toolkit installed")
        print("   - GPU properly connected")
        return
    
    # Optimize frameworks
    print("\n2. Optimizing GPU frameworks...")
    monitor.optimize_tensorflow_gpu()
    monitor.optimize_pytorch_gpu()
    
    # Show optimal batch sizes
    print("\n3. Optimal batch sizes for RTX 2060 SUPER:")
    batch_sizes = monitor.get_optimal_batch_sizes()
    for arch, batch_size in batch_sizes.items():
        print(f"   {arch}: {batch_size}")
    
    # Optional stress test
    response = input("\n4. Run memory stress test? (y/n): ").lower()
    if response == 'y':
        monitor.run_memory_stress_test()
    
    # Optional monitoring
    response = input("\n5. Start GPU monitoring? (y/n): ").lower()
    if response == 'y':
        duration = int(input("Duration in seconds (default 60): ") or "60")
        monitor.monitor_during_training(duration)
    
    print("\n✅ GPU check completed!")
    print("\nNext steps:")
    print("1. Run experiments: python main.py parallel gpu_experiment_plan.csv 1 --gpu-ids 0")
    print("2. Monitor with: nvidia-smi -l 1")
    print("3. Check results in: results/")

if __name__ == "__main__":
    main()