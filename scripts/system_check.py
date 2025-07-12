#!/usr/bin/env python3
"""
System Check Script for ANNrun_code
Validates system requirements, dependencies, and performance capabilities
"""

import os
import sys
import platform
import subprocess
import importlib
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SystemChecker:
    """Comprehensive system checker for ANNrun_code"""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'python_info': {},
            'dependencies': {},
            'hardware': {},
            'performance': {},
            'recommendations': [],
            'overall_status': 'unknown'
        }
    
    def check_system_info(self) -> Dict[str, Any]:
        """Check basic system information"""
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
        
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        self.results['system_info'] = info
        return info
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment details"""
        print("\n" + "=" * 60)
        print("PYTHON ENVIRONMENT")
        print("=" * 60)
        
        info = {
            'version': sys.version,
            'executable': sys.executable,
            'path': sys.path[:3],  # First 3 paths
            'encoding': sys.getdefaultencoding(),
            'platform': sys.platform
        }
        
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Executable: {sys.executable}")
        print(f"Platform: {sys.platform}")
        print(f"Default Encoding: {sys.getdefaultencoding()}")
        
        self.results['python_info'] = info
        return info
    
    def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check required and optional dependencies"""
        print("\n" + "=" * 60)
        print("DEPENDENCY CHECK")
        print("=" * 60)
        
        # Required packages
        required_packages = {
            'numpy': 'Numerical computations',
            'pandas': 'Data manipulation',
            'scikit-learn': 'Machine learning',
            'matplotlib': 'Plotting',
            'seaborn': 'Statistical plotting',
            'joblib': 'Parallel processing',
            'psutil': 'System monitoring'
        }
        
        # Optional packages
        optional_packages = {
            'tensorflow': 'Deep learning framework',
            'torch': 'PyTorch deep learning',
            'ray': 'Distributed computing',
            'xgboost': 'Gradient boosting',
            'lightgbm': 'Light gradient boosting',
            'plotly': 'Interactive plotting'
        }
        
        dependency_status = {}
        
        print("Required packages:")
        for package, description in required_packages.items():
            status = self._check_package(package, description)
            dependency_status[package] = status
        
        print("\nOptional packages:")
        for package, description in optional_packages.items():
            status = self._check_package(package, description)
            dependency_status[f"{package}_optional"] = status
        
        self.results['dependencies'] = dependency_status
        return dependency_status
    
    def _check_package(self, package_name: str, description: str) -> Dict[str, Any]:
        """Check individual package"""
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            
            status = {
                'installed': True,
                'version': version,
                'description': description,
                'module': module
            }
            
            print(f"  ✅ {package_name} ({version}): {description}")
            return status
            
        except ImportError:
            status = {
                'installed': False,
                'version': None,
                'description': description,
                'module': None
            }
            
            print(f"  ❌ {package_name}: {description}")
            return status
    
    def check_hardware(self) -> Dict[str, Any]:
        """Check hardware specifications"""
        print("\n" + "=" * 60)
        print("HARDWARE CHECK")
        print("=" * 60)
        
        hardware_info = {}
        
        # CPU information
        try:
            import psutil
            
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'unknown',
                'cpu_usage': psutil.cpu_percent(interval=1)
            }
            
            print(f"CPU Physical Cores: {cpu_info['physical_cores']}")
            print(f"CPU Logical Cores: {cpu_info['logical_cores']}")
            print(f"CPU Max Frequency: {cpu_info['cpu_freq_max']} MHz")
            print(f"CPU Usage: {cpu_info['cpu_usage']:.1f}%")
            
            hardware_info['cpu'] = cpu_info
            
        except ImportError:
            print("❌ psutil not available for CPU info")
            hardware_info['cpu'] = {'error': 'psutil not available'}
        
        # Memory information
        try:
            memory_info = {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'usage_percent': psutil.virtual_memory().percent
            }
            
            print(f"Total Memory: {memory_info['total_gb']:.2f} GB")
            print(f"Available Memory: {memory_info['available_gb']:.2f} GB")
            print(f"Memory Usage: {memory_info['usage_percent']:.1f}%")
            
            hardware_info['memory'] = memory_info
            
        except:
            print("❌ Memory info not available")
            hardware_info['memory'] = {'error': 'not available'}
        
        # GPU information
        gpu_info = self._check_gpu()
        hardware_info['gpu'] = gpu_info
        
        self.results['hardware'] = hardware_info
        return hardware_info
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability"""
        gpu_info = {'available': False, 'frameworks': {}}
        
        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            gpu_info['frameworks']['tensorflow'] = {
                'gpu_available': gpu_available,
                'gpu_count': len(tf.config.list_physical_devices('GPU'))
            }
            
            if gpu_available:
                print("✅ TensorFlow GPU available")
                gpu_info['available'] = True
            else:
                print("❌ TensorFlow GPU not available")
                
        except ImportError:
            print("❌ TensorFlow not installed")
            gpu_info['frameworks']['tensorflow'] = {'error': 'not installed'}
        except Exception as e:
            print(f"❌ TensorFlow GPU check failed: {e}")
            gpu_info['frameworks']['tensorflow'] = {'error': str(e)}
        
        # Check PyTorch GPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_info['frameworks']['pytorch'] = {
                'gpu_available': gpu_available,
                'gpu_count': torch.cuda.device_count() if gpu_available else 0
            }
            
            if gpu_available:
                print("✅ PyTorch GPU available")
                gpu_info['available'] = True
            else:
                print("❌ PyTorch GPU not available")
                
        except ImportError:
            print("❌ PyTorch not installed")
            gpu_info['frameworks']['pytorch'] = {'error': 'not installed'}
        except Exception as e:
            print(f"❌ PyTorch GPU check failed: {e}")
            gpu_info['frameworks']['pytorch'] = {'error': str(e)}
        
        return gpu_info
    
    def check_performance(self) -> Dict[str, Any]:
        """Run basic performance tests"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST")
        print("=" * 60)
        
        performance_info = {}
        
        # NumPy performance test
        try:
            import numpy as np
            
            print("Running NumPy performance test...")
            start_time = time.time()
            
            # Matrix multiplication test
            a = np.random.random((1000, 1000))
            b = np.random.random((1000, 1000))
            c = np.dot(a, b)
            
            numpy_time = time.time() - start_time
            performance_info['numpy_matmul_1000x1000'] = numpy_time
            
            print(f"✅ NumPy 1000x1000 matrix multiplication: {numpy_time:.3f}s")
            
        except Exception as e:
            print(f"❌ NumPy test failed: {e}")
            performance_info['numpy_matmul_1000x1000'] = {'error': str(e)}
        
        # Pandas performance test
        try:
            import pandas as pd
            
            print("Running Pandas performance test...")
            start_time = time.time()
            
            # DataFrame operations test
            df = pd.DataFrame(np.random.random((100000, 10)))
            df_grouped = df.groupby(df.iloc[:, 0] > 0.5).mean()
            
            pandas_time = time.time() - start_time
            performance_info['pandas_groupby_100k'] = pandas_time
            
            print(f"✅ Pandas 100k row groupby: {pandas_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Pandas test failed: {e}")
            performance_info['pandas_groupby_100k'] = {'error': str(e)}
        
        self.results['performance'] = performance_info
        return performance_info
    
    def generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Check memory
        if 'memory' in self.results['hardware'] and 'total_gb' in self.results['hardware']['memory']:
            memory_gb = self.results['hardware']['memory']['total_gb']
            if memory_gb < 8:
                recommendations.append("⚠️  Low memory (< 8GB). Consider upgrading for better performance.")
            elif memory_gb < 16:
                recommendations.append("💡 Memory is adequate (8-16GB). 32GB+ recommended for large experiments.")
            else:
                recommendations.append("✅ Good memory capacity (16GB+).")
        
        # Check CPU cores
        if 'cpu' in self.results['hardware'] and 'logical_cores' in self.results['hardware']['cpu']:
            cores = self.results['hardware']['cpu']['logical_cores']
            if cores < 4:
                recommendations.append("⚠️  Few CPU cores (< 4). Parallel processing will be limited.")
            elif cores < 8:
                recommendations.append("💡 Decent CPU cores (4-8). Consider using parallel processing.")
            else:
                recommendations.append("✅ Good CPU core count (8+). Parallel processing highly recommended.")
        
        # Check dependencies
        required_missing = []
        for package in ['numpy', 'pandas', 'scikit-learn', 'joblib']:
            if package in self.results['dependencies']:
                if not self.results['dependencies'][package]['installed']:
                    required_missing.append(package)
        
        if required_missing:
            recommendations.append(f"❌ Install missing required packages: {', '.join(required_missing)}")
        else:
            recommendations.append("✅ All required packages are installed.")
        
        # Check GPU
        if 'gpu' in self.results['hardware']:
            if self.results['hardware']['gpu']['available']:
                recommendations.append("✅ GPU available. Consider using GPU-accelerated models.")
            else:
                recommendations.append("💡 No GPU detected. CPU-only processing will be used.")
        
        # Performance recommendations
        if 'numpy_matmul_1000x1000' in self.results['performance']:
            numpy_time = self.results['performance']['numpy_matmul_1000x1000']
            if isinstance(numpy_time, float):
                if numpy_time > 2.0:
                    recommendations.append("⚠️  Slow NumPy performance. Check BLAS/LAPACK installation.")
                elif numpy_time < 0.5:
                    recommendations.append("✅ Excellent NumPy performance.")
                else:
                    recommendations.append("💡 Good NumPy performance.")
        
        # Experiment size recommendations
        if 'cpu' in self.results['hardware'] and 'logical_cores' in self.results['hardware']['cpu']:
            cores = self.results['hardware']['cpu']['logical_cores']
            est_time_hours = 324 * 5 / 60 / cores  # Rough estimate
            recommendations.append(f"⏱️  Estimated time for 324 experiments: ~{est_time_hours:.1f} hours with {cores} cores")
        
        self.results['recommendations'] = recommendations
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def determine_overall_status(self) -> str:
        """Determine overall system status"""
        
        # Check critical components
        has_required_deps = True
        for package in ['numpy', 'pandas', 'scikit-learn']:
            if package in self.results['dependencies']:
                if not self.results['dependencies'][package]['installed']:
                    has_required_deps = False
                    break
        
        has_adequate_memory = True
        if 'memory' in self.results['hardware'] and 'total_gb' in self.results['hardware']['memory']:
            if self.results['hardware']['memory']['total_gb'] < 4:
                has_adequate_memory = False
        
        has_adequate_cpu = True
        if 'cpu' in self.results['hardware'] and 'logical_cores' in self.results['hardware']['cpu']:
            if self.results['hardware']['cpu']['logical_cores'] < 2:
                has_adequate_cpu = False
        
        # Determine status
        if not has_required_deps:
            status = "🔴 NOT READY - Missing required dependencies"
        elif not has_adequate_memory or not has_adequate_cpu:
            status = "🟠 LIMITED - Hardware constraints"
        elif self.results['hardware'].get('gpu', {}).get('available', False):
            status = "🟢 EXCELLENT - GPU available"
        else:
            status = "🟡 GOOD - CPU only"
        
        self.results['overall_status'] = status
        return status
    
    def run_full_check(self) -> Dict[str, Any]:
        """Run complete system check"""
        print("ANNrun_code System Check")
        print("Checking system compatibility and performance...")
        print()
        
        # Run all checks
        self.check_system_info()
        self.check_python_environment()
        self.check_dependencies()
        self.check_hardware()
        self.check_performance()
        self.generate_recommendations()
        
        # Determine overall status
        status = self.determine_overall_status()
        
        print("\n" + "=" * 60)
        print("SYSTEM CHECK SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {status}")
        print(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def save_report(self, filename: str = "system_check_report.json"):
        """Save system check report to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 System check report saved to: {filename}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ANNrun_code System Check")
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed report to JSON file')
    parser.add_argument('--output-file', default='system_check_report.json',
                       help='Output file for report (default: system_check_report.json)')
    
    args = parser.parse_args()
    
    # Run system check
    checker = SystemChecker()
    results = checker.run_full_check()
    
    # Save report if requested
    if args.save_report:
        checker.save_report(args.output_file)
    
    # Exit with appropriate code based on status
    status = results['overall_status']
    if '🔴' in status:
        sys.exit(1)  # Critical issues
    elif '🟠' in status:
        sys.exit(2)  # Warnings
    else:
        sys.exit(0)  # Good to go


if __name__ == "__main__":
    main()
