#!/usr/bin/env python3
"""
Missing Files Creator for ANNrun_code
Automatically creates all missing required files
"""

import os
from pathlib import Path
import json

def create_missing_files(project_root="ANNrun_code"):
    """Create all missing files for ANNrun_code project"""
    
    project_path = Path(project_root)
    
    print(f"Creating missing files in: {project_path.absolute()}")
    print("=" * 60)
    
    # 1. Create configs/__init__.py
    configs_init = project_path / "configs" / "__init__.py"
    configs_init.parent.mkdir(parents=True, exist_ok=True)
    
    with open(configs_init, 'w', encoding='utf-8') as f:
        f.write('"""Configuration management package"""\n')
    print("✅ Created configs/__init__.py")
    
    # 2. Create core/data/data_manager.py
    data_manager = project_path / "core" / "data" / "data_manager.py"
    
    with open(data_manager, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Enhanced Data Manager for ANNrun_code
Manages AGERA5, Himawari, and other data sources with advanced features
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
from datetime import datetime, timedelta

# Import loaders
from .loaders.base_loader import BaseDataLoader
from .loaders.pangaea_loader import PangaeaDataLoader


class EnhancedDataManager:
    """Enhanced data manager supporting multiple data sources"""
    
    # Supported data sources and their features
    SUPPORTED_FEATURES = {
        'AGERA5': ['AGERA5_SRAD', 'AGERA5_TMAX', 'AGERA5_DTR', 'AGERA5_PREC', 'AGERA5_VPRE', 'AGERA5_WIND'],
        'HIMAWARI': ['HIMA_SRAD', 'HIMA_TMAX', 'HIMA_DTR', 'HIMA_RAIN'],
        'THEORETICAL': ['THEO'],
        'PANGAEA': ['station_data']  # Legacy support
    }
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize enhanced data manager"""
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize loaders
        self.loaders = {
            'pangaea': PangaeaDataLoader(str(self.data_dir)),
            'base': BaseDataLoader(str(self.data_dir))
        }
        
        # Data cache
        self._data_cache = {}
        self._metadata_cache = {}
        
        self.logger.info(f"Enhanced Data Manager initialized with data_dir: {self.data_dir}")
    
    def validate_data_paths(self) -> Dict[str, bool]:
        """Validate that required data paths exist"""
        paths_to_check = {
            'data_dir': self.data_dir.exists(),
            'agera5_dir': (self.data_dir / 'agera5').exists(),
            'himawari_dir': (self.data_dir / 'himawari').exists(),
            'pangaea_dir': (self.data_dir / 'pangaea').exists(),
            'theoretical_dir': (self.data_dir / 'theoretical').exists()
        }
        
        self.logger.info("Data path validation:")
        for path, exists in paths_to_check.items():
            status = "✅" if exists else "❌"
            self.logger.info(f"  {status} {path}")
        
        return paths_to_check
    
    def validate_feature_combination(self, features: Union[str, List[str]]) -> Dict[str, bool]:
        """Validate that requested features are supported"""
        if isinstance(features, str):
            features = [f.strip() for f in features.split(',')]
        
        all_supported = []
        for source_features in self.SUPPORTED_FEATURES.values():
            all_supported.extend(source_features)
        
        validation = {}
        for feature in features:
            validation[feature] = feature in all_supported
        
        return validation
    
    def get_feature_source(self, feature: str) -> Optional[str]:
        """Get the data source for a given feature"""
        for source, features in self.SUPPORTED_FEATURES.items():
            if feature in features:
                return source
        return None
    
    def load_data_for_features(self, features: Union[str, List[str]], 
                             station_ids: Optional[List[int]] = None,
                             date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Load data for specified features"""
        
        if isinstance(features, str):
            features = [f.strip() for f in features.split(',')]
        
        self.logger.info(f"Loading data for features: {features}")
        
        # Group features by data source
        source_features = {}
        for feature in features:
            source = self.get_feature_source(feature)
            if source:
                if source not in source_features:
                    source_features[source] = []
                source_features[source].append(feature)
            else:
                self.logger.warning(f"Unknown feature: {feature}")
        
        # Load data from each source
        loaded_data = {}
        for source, source_feature_list in source_features.items():
            if source == 'PANGAEA':
                # Use pangaea loader for legacy support
                data = self.loaders['pangaea'].load_all_stations()
                loaded_data['PANGAEA'] = data
            else:
                # Load other data sources
                data = self._load_source_data(source, source_feature_list, station_ids, date_range)
                loaded_data[source] = data
        
        return loaded_data
    
    def _load_source_data(self, source: str, features: List[str], 
                         station_ids: Optional[List[int]] = None,
                         date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Load data from a specific source"""
        
        cache_key = f"{source}_{','.join(features)}_{station_ids}_{date_range}"
        
        if cache_key in self._data_cache:
            self.logger.info(f"Loading {source} data from cache")
            return self._data_cache[cache_key]
        
        self.logger.info(f"Loading {source} data from files")
        
        if source == 'AGERA5':
            data = self._load_agera5_data(features, station_ids, date_range)
        elif source == 'HIMAWARI':
            data = self._load_himawari_data(features, station_ids, date_range)
        elif source == 'THEORETICAL':
            data = self._load_theoretical_data(features, station_ids, date_range)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Cache the data
        self._data_cache[cache_key] = data
        
        return data
    
    def _load_agera5_data(self, features: List[str], station_ids: Optional[List[int]], 
                         date_range: Optional[Tuple[str, str]]) -> Dict[str, Any]:
        """Load AGERA5 data"""
        agera5_dir = self.data_dir / 'agera5'
        
        if not agera5_dir.exists():
            raise FileNotFoundError(f"AGERA5 directory not found: {agera5_dir}")
        
        # Mock implementation - replace with actual AGERA5 loading logic
        data = {
            'source': 'AGERA5',
            'features': features,
            'data': pd.DataFrame(),  # Placeholder
            'metadata': {
                'loaded_at': datetime.now().isoformat(),
                'feature_count': len(features),
                'station_ids': station_ids
            }
        }
        
        self.logger.info(f"Loaded AGERA5 data for {len(features)} features")
        return data
    
    def _load_himawari_data(self, features: List[str], station_ids: Optional[List[int]], 
                          date_range: Optional[Tuple[str, str]]) -> Dict[str, Any]:
        """Load Himawari data"""
        himawari_dir = self.data_dir / 'himawari'
        
        if not himawari_dir.exists():
            raise FileNotFoundError(f"Himawari directory not found: {himawari_dir}")
        
        # Mock implementation - replace with actual Himawari loading logic
        data = {
            'source': 'HIMAWARI',
            'features': features,
            'data': pd.DataFrame(),  # Placeholder
            'metadata': {
                'loaded_at': datetime.now().isoformat(),
                'feature_count': len(features),
                'station_ids': station_ids,
                'note': 'HIMA_DTR calculated as TMAX - TMIN, HIMA_RAIN is atmospheric moisture'
            }
        }
        
        self.logger.info(f"Loaded Himawari data for {len(features)} features")
        return data
    
    def _load_theoretical_data(self, features: List[str], station_ids: Optional[List[int]], 
                             date_range: Optional[Tuple[str, str]]) -> Dict[str, Any]:
        """Load theoretical data"""
        theoretical_dir = self.data_dir / 'theoretical'
        
        if not theoretical_dir.exists():
            raise FileNotFoundError(f"Theoretical directory not found: {theoretical_dir}")
        
        # Mock implementation - replace with actual theoretical data loading logic
        data = {
            'source': 'THEORETICAL',
            'features': features,
            'data': pd.DataFrame(),  # Placeholder
            'metadata': {
                'loaded_at': datetime.now().isoformat(),
                'feature_count': len(features),
                'station_ids': station_ids
            }
        }
        
        self.logger.info(f"Loaded theoretical data for {len(features)} features")
        return data
    
    def get_supported_features(self) -> Dict[str, List[str]]:
        """Get all supported features grouped by source"""
        return self.SUPPORTED_FEATURES.copy()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        summary = {
            'data_directory': str(self.data_dir),
            'supported_sources': list(self.SUPPORTED_FEATURES.keys()),
            'total_features': sum(len(features) for features in self.SUPPORTED_FEATURES.values()),
            'cache_status': {
                'cached_datasets': len(self._data_cache),
                'cached_metadata': len(self._metadata_cache)
            },
            'path_validation': self.validate_data_paths()
        }
        
        return summary
    
    def clear_cache(self):
        """Clear data cache"""
        self._data_cache.clear()
        self._metadata_cache.clear()
        self.logger.info("Data cache cleared")
    
    def export_data_config(self, filepath: str):
        """Export current data configuration"""
        config = {
            'data_manager_version': '2.0',
            'data_directory': str(self.data_dir),
            'supported_features': self.SUPPORTED_FEATURES,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Data configuration exported to {filepath}")


# Convenience functions
def create_data_manager(data_dir: str = "./data") -> EnhancedDataManager:
    """Create enhanced data manager instance"""
    return EnhancedDataManager(data_dir)


def validate_features(features: Union[str, List[str]], data_dir: str = "./data") -> Dict[str, bool]:
    """Validate features without creating full data manager"""
    manager = EnhancedDataManager(data_dir)
    return manager.validate_feature_combination(features)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data manager
    dm = create_data_manager()
    
    # Show summary
    summary = dm.get_data_summary()
    print("Data Manager Summary:")
    print(json.dumps(summary, indent=2))
    
    # Test feature validation
    test_features = ['AGERA5_SRAD', 'HIMA_TMAX', 'THEO', 'INVALID_FEATURE']
    validation = dm.validate_feature_combination(test_features)
    print(f"\\nFeature validation: {validation}")
''')
    print("✅ Created core/data/data_manager.py")
    
    # 3. Create configs/config_manager.py
    config_manager = project_path / "configs" / "config_manager.py"
    
    with open(config_manager, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
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
        print(f"\\nTest experiment config: {config}")
        
        # Test validation
        validation = cm.validate_experiment_config(config)
        print(f"Config validation: {validation}")
        
    except Exception as e:
        print(f"Test loading failed (expected if files don't exist): {e}")
''')
    print("✅ Created configs/config_manager.py")
    
    # 4. Create core/utils/logging_config.py
    logging_config = project_path / "core" / "utils" / "logging_config.py"
    
    with open(logging_config, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Logging Configuration for ANNrun_code
Centralized logging setup with multiple handlers and formats
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\\033[36m',      # Cyan
        'INFO': '\\033[32m',       # Green
        'WARNING': '\\033[33m',    # Yellow
        'ERROR': '\\033[31m',      # Red
        'CRITICAL': '\\033[35m',   # Magenta
        'RESET': '\\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File formatter (detailed)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console formatter (clean)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handlers = []
    
    # File handler with rotation
    if file_output:
        log_file = log_path / f"annrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        logger.addHandler(console_handler)
    
    # Error file handler (separate file for errors)
    if file_output:
        error_log_file = log_path / f"annrun_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)
        logger.addHandler(error_handler)
    
    # Log the setup
    logger.info("=" * 60)
    logger.info("ANNrun_code Logging System Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    logger.info(f"Console Output: {console_output}")
    logger.info(f"File Output: {file_output}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_system_info():
    """Log system information"""
    logger = get_logger(__name__)
    
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"  Current Working Directory: {os.getcwd()}")


def log_experiment_start(experiment_id: int, config: Dict[str, Any]):
    """Log experiment start information"""
    logger = get_logger(__name__)
    
    logger.info("🚀 EXPERIMENT START")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Started at: {datetime.now().isoformat()}")


def log_experiment_end(experiment_id: int, duration: float, success: bool):
    """Log experiment end information"""
    logger = get_logger(__name__)
    
    status = "✅ COMPLETED" if success else "❌ FAILED"
    logger.info(f"🏁 EXPERIMENT END - {status}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Ended at: {datetime.now().isoformat()}")


def setup_module_logger(module_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger for a specific module"""
    logger = logging.getLogger(module_name)
    
    # If root logger is already configured, use it
    if logging.getLogger().handlers:
        return logger
    
    # Otherwise setup basic logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logger.setLevel(numeric_level)
    
    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Performance logging helpers
class PerformanceLogger:
    """Helper class for logging performance metrics"""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = get_logger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"⏱️  Started: {operation}")
    
    def end_timer(self, operation: str):
        """End timing an operation and log duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"⏱️  Completed: {operation} in {duration:.3f}s")
            del self.start_times[operation]
            return duration
        else:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return None
    
    def log_memory_usage(self, operation: str = ""):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"💾 Memory usage{' for ' + operation if operation else ''}: {memory_mb:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory logging")


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(
        log_dir="./test_logs",
        log_level="DEBUG",
        console_output=True,
        file_output=True
    )
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.start_timer("test_operation")
    
    import time
    time.sleep(1)  # Simulate work
    
    perf_logger.end_timer("test_operation")
    perf_logger.log_memory_usage("test")
    
    # Log system info
    log_system_info()
    
    print("Logging test completed!")
''')
    print("✅ Created core/utils/logging_config.py")
    
    # 5. Create scripts/system_check.py
    scripts_dir = project_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    system_check = scripts_dir / "system_check.py"
    
    with open(system_check, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
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
        print("\\n" + "=" * 60)
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
        print("\\n" + "=" * 60)
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
        
        print("\\nOptional packages:")
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
        print("\\n" + "=" * 60)
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
        print("\\n" + "=" * 60)
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
        print("\\n" + "=" * 60)
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
        
        print("\\n" + "=" * 60)
        print("SYSTEM CHECK SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {status}")
        print(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def save_report(self, filename: str = "system_check_report.json"):
        """Save system check report to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\\n📄 System check report saved to: {filename}")


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
''')
    print("✅ Created scripts/system_check.py")
    
    print("\n" + "=" * 60)
    print("🎉 ALL MISSING FILES CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run progress checker again: python progress_checker.py")
    print("2. If all tests pass, try: python scripts/system_check.py")
    print("3. Test main runner: python main.py --dry-run")
    print("\nAll files are now ready for use!")


if __name__ == "__main__":
    create_missing_files()