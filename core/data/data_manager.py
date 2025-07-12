#!/usr/bin/env python3
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
    print(f"\nFeature validation: {validation}")
