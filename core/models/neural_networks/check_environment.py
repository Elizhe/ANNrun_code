#!/usr/bin/env python3
"""
Environment Check Script
Check Python environment and installed packages
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}")

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        # Try to import
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def check_pip_packages():
    """Check pip packages"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n=== Installed Packages ===")
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # Show first 10 packages
                if line.strip():
                    print(line)
            print("... (use 'pip list' to see all packages)")
        else:
            print("Error running pip list")
    except Exception as e:
        print(f"Error checking pip packages: {e}")

def main():
    """Main check function"""
    check_python_version()
    
    print("\n=== Required Packages ===")
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'), 
        ('scikit-learn', 'sklearn'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
        ('matplotlib', 'matplotlib'),
        ('joblib', 'joblib')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n=== Missing Packages ===")
        print("Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # Specific installation commands
        print("\n=== Specific Installation Commands ===")
        if 'numpy' in missing_packages:
            print("pip install numpy")
        if 'pandas' in missing_packages:
            print("pip install pandas")
        if 'scikit-learn' in missing_packages:
            print("pip install scikit-learn")
        if 'tensorflow' in missing_packages:
            print("pip install tensorflow")
        if 'torch' in missing_packages:
            print("pip install torch")
        if 'matplotlib' in missing_packages:
            print("pip install matplotlib")
        if 'joblib' in missing_packages:
            print("pip install joblib")
    else:
        print("\n✓ All required packages are installed!")
    
    check_pip_packages()
    
    print("\n=== Recommendations ===")
    if missing_packages:
        print("1. Install missing packages")
        print("2. Make sure your virtual environment is activated")
        print("3. Restart your IDE/editor after installation")
    else:
        print("Your environment looks good!")

if __name__ == "__main__":
    main()