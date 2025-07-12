#!/usr/bin/env python3
"""
Configuration Manager for ANNrun_code
Manages experiment configurations, model settings, and data configurations
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Experiment configuration data class"""
    id: int
    features: str
    normalization: str
    bias_correction: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'features': self.features,
            'normalization': self.normalization,
            'bias_correction': self.bias_correction,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        return cls(
            id=int(data['id']),
            features=str(data['features']),
            normalization=str(data['normalization']),
            bias_correction=str(data['bias_correction']),
            description=str(data.get('description', ''))
        )


class ConfigManager:
    """Configuration manager for ANNrun_code"""
    
    def __init__(self, config_dir: str = "./configs"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Configuration paths
        self.experiment_plans_dir = self.config_dir / "experiment_plans"
        self.model_configs_dir = self.config_dir / "model_configs"
        self.data_configs_dir = self.config_dir / "data_configs"
        
        # Ensure directories exist
        for dir_path in [self.experiment_plans_dir, self.model_configs_dir, self.data_configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Configuration manager initialized with config_dir: {self.config_dir}")
    
    def load_experiment_config(self, plan_file: str, experiment_id: int) -> Dict[str, Any]:
        """Load experiment configuration from plan file"""
        
        # Try absolute path first, then relative to experiment_plans_dir
        plan_path = Path(plan_file)
        if not plan_path.exists():
            plan_path = self.experiment_plans_dir / plan_file
            if not plan_path.exists():
                raise FileNotFoundError(f"Experiment plan file not found: {plan_file}")
        
        self.logger.info(f"Loading experiment config from: {plan_path}")
        
        try:
            df = pd.read_csv(plan_path, encoding='utf-8')
            exp_row = df[df['id'] == experiment_id]
            
            if exp_row.empty:
                raise ValueError(f"Experiment ID {experiment_id} not found in {plan_path}")
            
            row = exp_row.iloc[0]
            config = ExperimentConfig(
                id=int(row['id']),
                features=str(row['features']),
                normalization=str(row['normalization']),
                bias_correction=str(row['bias_correction']),
                description=str(row.get('description', ''))
            )
            
            self.logger.info(f"Loaded experiment {experiment_id}: {config.description}")
            return config.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error loading experiment config: {e}")
            raise
    
    def load_all_experiment_configs(self, plan_file: str) -> List[Dict[str, Any]]:
        """Load all experiment configurations from plan file"""
        
        plan_path = Path(plan_file)
        if not plan_path.exists():
            plan_path = self.experiment_plans_dir / plan_file
            if not plan_path.exists():
                raise FileNotFoundError(f"Experiment plan file not found: {plan_file}")
        
        try:
            df = pd.read_csv(plan_path, encoding='utf-8')
            configs = []
            
            for _, row in df.iterrows():
                config = ExperimentConfig(
                    id=int(row['id']),
                    features=str(row['features']),
                    normalization=str(row['normalization']),
                    bias_correction=str(row['bias_correction']),
                    description=str(row.get('description', ''))
                )
                configs.append(config.to_dict())
            
            self.logger.info(f"Loaded {len(configs)} experiment configurations")
            return configs
            
        except Exception as e:
            self.logger.error(f"Error loading all experiment configs: {e}")
            raise
    
    def generate_experiment_combinations(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experiment combinations based on base configuration"""
        
        # Architecture combinations (12 architectures × 9 regularizations × 3 optimizers = 324)
        architectures = [
            'single_32', 'single_64', 'single_128',
            'double_32_16', 'double_64_32', 'double_128_64', 'double_64_64',
            'triple_128_64_32', 'triple_64_32_16', 'triple_32_32_16',
            'deep_256_128_64_32', 'deep_128_128_64_32'
        ]
        
        regularizations = [
            'none', 'dropout_0.1', 'dropout_0.2', 'dropout_0.3',
            'batch_norm', 'l2_light', 'l1l2_medium',
            'dropout_batch_norm', 'full_regularization'
        ]
        
        optimizers = [
            'adam_default', 'adam_low', 'rmsprop'
        ]
        
        combinations = []
        combo_id = 1
        
        for arch in architectures:
            for reg in regularizations:
                for opt in optimizers:
                    combo_config = base_config.copy()
                    combo_config.update({
                        'combination_id': combo_id,
                        'architecture': arch,
                        'regularization': reg,
                        'optimizer': opt,
                        'description': f"{combo_config.get('description', '')} - {arch}_{reg}_{opt}"
                    })
                    combinations.append(combo_config)
                    combo_id += 1
        
        self.logger.info(f"Generated {len(combinations)} experiment combinations")
        return combinations
    
    def save_experiment_config(self, config: Dict[str, Any], filename: str):
        """Save experiment configuration to file"""
        
        filepath = self.experiment_plans_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Experiment configuration saved to {filepath}")
    
    def load_model_config(self, config_name: str) -> Dict[str, Any]:
        """Load model configuration"""
        
        config_path = self.model_configs_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"Loaded model config: {config_name}")
        return config
    
    def save_model_config(self, config: Dict[str, Any], config_name: str):
        """Save model configuration"""
        
        config_path = self.model_configs_dir / f"{config_name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Model config saved: {config_path}")
    
    def load_data_config(self, config_name: str) -> Dict[str, Any]:
        """Load data configuration"""
        
        config_path = self.data_configs_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Data config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"Loaded data config: {config_name}")
        return config
    
    def save_data_config(self, config: Dict[str, Any], config_name: str):
        """Save data configuration"""
        
        config_path = self.data_configs_dir / f"{config_name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Data config saved: {config_path}")
    
    def get_available_experiments(self, plan_file: str) -> List[int]:
        """Get list of available experiment IDs"""
        
        plan_path = Path(plan_file)
        if not plan_path.exists():
            plan_path = self.experiment_plans_dir / plan_file
            if not plan_path.exists():
                raise FileNotFoundError(f"Experiment plan file not found: {plan_file}")
        
        df = pd.read_csv(plan_path, encoding='utf-8')
        return df['id'].tolist()
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Validate experiment configuration"""
        
        validation = {
            'has_id': 'id' in config,
            'has_features': 'features' in config and config['features'],
            'has_normalization': 'normalization' in config and config['normalization'],
            'has_bias_correction': 'bias_correction' in config and config['bias_correction']
        }
        
        validation['is_valid'] = all(validation.values())
        
        return validation
    
    def export_config_summary(self, output_file: str):
        """Export configuration summary"""
        
        summary = {
            'config_manager_info': {
                'config_directory': str(self.config_dir),
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'available_experiment_plans': [],
            'available_model_configs': [],
            'available_data_configs': []
        }
        
        # List experiment plans
        if self.experiment_plans_dir.exists():
            for file in self.experiment_plans_dir.glob("*.csv"):
                summary['available_experiment_plans'].append(file.name)
        
        # List model configs
        if self.model_configs_dir.exists():
            for file in self.model_configs_dir.glob("*.json"):
                summary['available_model_configs'].append(file.stem)
        
        # List data configs
        if self.data_configs_dir.exists():
            for file in self.data_configs_dir.glob("*.json"):
                summary['available_data_configs'].append(file.stem)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Configuration summary exported to {output_file}")


# Convenience functions
def create_config_manager(config_dir: str = "./configs") -> ConfigManager:
    """Create configuration manager instance"""
    return ConfigManager(config_dir)


def load_experiment(plan_file: str, experiment_id: int, config_dir: str = "./configs") -> Dict[str, Any]:
    """Load single experiment configuration"""
    manager = ConfigManager(config_dir)
    return manager.load_experiment_config(plan_file, experiment_id)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config manager
    cm = create_config_manager()
    
    # Export summary
    cm.export_config_summary("config_summary.json")
    
    # Test loading experiment (if files exist)
    try:
        config = cm.load_experiment_config("experiment_plan_full.csv", 1)
        print(f"\nTest experiment config: {config}")
        
        # Test validation
        validation = cm.validate_experiment_config(config)
        print(f"Config validation: {validation}")
        
    except Exception as e:
        print(f"Test loading failed (expected if files don't exist): {e}")
