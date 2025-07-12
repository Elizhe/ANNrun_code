from setuptools import setup, find_packages

setup(
    name="annrun_code",
    version="1.0.0",
    description="Enhanced Neural Network Experiment Framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "pyyaml>=5.4.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.8.0"],
        "pytorch": ["torch>=1.12.0"],
        "ray": ["ray>=2.0.0"],
        "all": ["tensorflow>=2.8.0", "torch>=1.12.0", "ray>=2.0.0"]
    }
)