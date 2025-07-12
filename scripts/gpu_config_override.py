#!/usr/bin/env python3
"""
GPU Configuration Override for Adam-only experiments
Modifies the default configuration to use only Adam optimizers
"""

import json
from pathlib import Path

class GPUConfigManager:
    """Manage GPU-specific configurations for experiments"""
    
    def __init__(self):
        # GPU-optimized architectures (12 architectures)
        self.architectures = [
            'single_32', 'single_64', 'single_128',
            'double_32_16', 'double_64_32', 'double_128_64', 'double_64_64',
            'triple_128_64_32', 'triple_64_32_16', 'triple_32_32_16',
            'deep_256_128_64_32', 'deep_128_128_64_32'
        ]
        
        # Regularization configurations (9 regularizations)
        self.regularizations = [
            'none', 'dropout_0.1', 'dropout_0.2', 'dropout_0.3',
            'batch_norm', 'l2_light', 'l1l2_medium',
            'dropout_batch_norm', 'full_regularization'
        ]
        
        # Adam-only optimizers (2 optimizers) - REMOVED rmsprop
        self.optimizers = [
            'adam_default',  # lr=0.001, beta1=0.9, beta2=0.999
            'adam_low'       # lr=0.0001, beta1=0.9, beta2=0.999
        ]
        
        # GPU-specific batch sizes for RTX 2060 SUPER (8GB VRAM)
        self.gpu_batch_sizes = {
            'single_32': 512,
            'single_64': 512,
            'single_128': 256,
            'double_32_16': 512,
            'double_64_32': 256,
            'double_128_64': 128,
            'double_64_64': 256,
            'triple_128_64_32': 128,
            'triple_64_32_16': 256,
            'triple_32_32_16': 256,
            'deep_256_128_64_32': 64,
            'deep_128_128_64_32': 128
        }
    
    def get_total_combinations(self):
        """Calculate total number of combinations"""
        return len(self.architectures) * len(self.regularizations) * len(self.optimizers)
    
    def get_gpu_memory_config(self):
        """Get GPU memory configuration for RTX 2060 SUPER"""
        return {
            'gpu_memory_limit': 7.5,  # Leave 0.5GB for system
            'memory_growth': True,
            'allow_memory_growth': True,
            'memory_limit_mb': 7680,  # 7.5GB in MB
            'tensorflow_memory_limit': 0.9,  # Use 90% of available GPU memory
            'pytorch_memory_fraction': 0.9
        }
    
    def get_gpu_training_config(self):
        """Get GPU-specific training configuration"""
        return {
            'use_mixed_precision': True,  # Use FP16 for memory efficiency
            'gradient_accumulation_steps': 1,
            'max_epochs': 200,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10,
            'min_lr': 1e-7,
            'validation_split': 0.2
        }
    
    def generate_experiment_combinations(self, base_config):
        """Generate all 216 experiment combinations"""
        combinations = []
        combo_id = 1
        
        print(f"Generating {self.get_total_combinations()} combinations...")
        print(f"Architectures: {len(self.architectures)}")
        print(f"Regularizations: {len(self.regularizations)}")
        print(f"Optimizers: {len(self.optimizers)} (Adam only)")
        
        for arch in self.architectures:
            for reg in self.regularizations:
                for opt in self.optimizers:
                    combo_config = base_config.copy()
                    combo_config.update({
                        'combination_id': combo_id,
                        'architecture': arch,
                        'regularization': reg,
                        'optimizer': opt,
                        'batch_size': self.gpu_batch_sizes.get(arch, 128),
                        'description': f"{base_config.get('description', '')} - {arch}_{reg}_{opt}",
                        'gpu_config': self.get_gpu_memory_config(),
                        'training_config': self.get_gpu_training_config()
                    })
                    combinations.append(combo_config)
                    combo_id += 1
        
        print(f"✅ Generated {len(combinations)} experiment combinations")
        return combinations
    
    def save_gpu_config(self, output_dir="configs"):
        """Save GPU configuration to file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        config = {
            'gpu_info': {
                'model': 'RTX 2060 SUPER',
                'memory_gb': 8,
                'compute_capability': '7.5'
            },
            'architectures': self.architectures,
            'regularizations': self.regularizations,
            'optimizers': self.optimizers,
            'batch_sizes': self.gpu_batch_sizes,
            'memory_config': self.get_gpu_memory_config(),
            'training_config': self.get_gpu_training_config(),
            'total_combinations': self.get_total_combinations()
        }
        
        config_path = Path(output_dir) / "gpu_config_adam_only.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ GPU configuration saved to: {config_path}")
        return config_path

def main():
    """Main function to generate GPU configuration"""
    gpu_manager = GPUConfigManager()
    
    print("=== GPU Configuration Manager ===")
    print(f"Target GPU: RTX 2060 SUPER (8GB VRAM)")
    print(f"Total combinations: {gpu_manager.get_total_combinations()}")
    print(f"Optimizers: Adam only (removed rmsprop)")
    
    # Save configuration
    config_path = gpu_manager.save_gpu_config()
    
    print("\n=== Configuration Summary ===")
    print(f"📁 Configuration saved to: {config_path}")
    print(f"🔧 {len(gpu_manager.architectures)} architectures")
    print(f"🛡️  {len(gpu_manager.regularizations)} regularization methods")
    print(f"⚡ {len(gpu_manager.optimizers)} optimizers (Adam variants)")
    print(f"🎯 {gpu_manager.get_total_combinations()} total experiments")
    
    print("\n=== Next Steps ===")
    print("1. Run: python gpu_memory_check.py")
    print("2. Run: python main.py parallel gpu_experiment_plan.csv 1 --gpu-ids 0")
    print("3. Monitor: nvidia-smi -l 1")

if __name__ == "__main__":
    main()