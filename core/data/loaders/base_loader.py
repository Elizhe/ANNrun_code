#!/usr/bin/env python3
"""
Base Data Loader for ANNrun_code
Provides base functionality for all data loaders
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders"""
    
    def __init__(self, data_dir: str):
        """Initialize base data loader"""
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with data_dir: {self.data_dir}")
    
    @abstractmethod
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate loaded data - must be implemented by subclasses"""
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about available data"""
        return {
            'data_dir': str(self.data_dir),
            'exists': self.data_dir.exists(),
            'loader_type': self.__class__.__name__
        }
    
    def list_available_files(self, extension: str = None) -> List[str]:
        """List available files in data directory"""
        if not self.data_dir.exists():
            return []
        
        if extension:
            pattern = f"*.{extension.lstrip('.')}"
            files = list(self.data_dir.glob(pattern))
        else:
            files = [f for f in self.data_dir.iterdir() if f.is_file()]
        
        return [str(f.name) for f in files]
    
    def check_file_exists(self, filename: str) -> bool:
        """Check if a specific file exists"""
        filepath = self.data_dir / filename
        return filepath.exists()
    
    def load_csv_file(self, filename: str, **kwargs) -> pd.DataFrame:
        """Generic CSV file loader with error handling"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            # Default CSV reading parameters
            default_params = {
                'encoding': 'utf-8',
                'skipinitialspace': True
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(filepath, **default_params)
            self.logger.info(f"Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {filename}: {e}")
            raise
    
    def save_data(self, data: Any, filename: str, format: str = 'csv') -> bool:
        """Save data to file"""
        try:
            filepath = self.data_dir / filename
            
            if format.lower() == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False, encoding='utf-8')
            elif format.lower() == 'json':
                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Data saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Get information about a specific file"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return {'exists': False, 'filepath': str(filepath)}
        
        stat = filepath.stat()
        
        return {
            'exists': True,
            'filepath': str(filepath),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime
        }


class DataLoader(BaseDataLoader):
    """Concrete implementation of BaseDataLoader for general use"""
    
    def __init__(self, data_dir: str):
        """Initialize general data loader"""
        super().__init__(data_dir)
        self.loaded_data = {}
    
    def load_data(self, filenames: Union[str, List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Load data from specified files or all available files"""
        
        if filenames is None:
            # Load all CSV files by default
            filenames = self.list_available_files('csv')
        elif isinstance(filenames, str):
            filenames = [filenames]
        
        loaded_data = {}
        
        for filename in filenames:
            try:
                if filename.endswith('.csv'):
                    data = self.load_csv_file(filename, **kwargs)
                    loaded_data[filename] = data
                else:
                    self.logger.warning(f"Unsupported file type: {filename}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {filename}: {e}")
                continue
        
        self.loaded_data.update(loaded_data)
        
        return loaded_data
    
    def validate_data(self, data: Any) -> bool:
        """Validate loaded data"""
        if isinstance(data, pd.DataFrame):
            # Basic DataFrame validation
            if data.empty:
                self.logger.warning("DataFrame is empty")
                return False
            
            if data.isnull().all().all():
                self.logger.warning("DataFrame contains only null values")
                return False
            
            return True
        
        elif isinstance(data, dict):
            # Validate dictionary of DataFrames
            return all(self.validate_data(df) for df in data.values() if isinstance(df, pd.DataFrame))
        
        else:
            self.logger.warning(f"Unknown data type for validation: {type(data)}")
            return False
    
    def get_loaded_data_summary(self) -> Dict[str, Any]:
        """Get summary of currently loaded data"""
        summary = {
            'total_datasets': len(self.loaded_data),
            'datasets': {}
        }
        
        for name, data in self.loaded_data.items():
            if isinstance(data, pd.DataFrame):
                summary['datasets'][name] = {
                    'type': 'DataFrame',
                    'rows': len(data),
                    'columns': len(data.columns),
                    'memory_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
                }
            else:
                summary['datasets'][name] = {
                    'type': str(type(data).__name__),
                    'size': len(data) if hasattr(data, '__len__') else 'unknown'
                }
        
        return summary
    
    def clear_cache(self):
        """Clear loaded data cache"""
        self.loaded_data.clear()
        self.logger.info("Data cache cleared")


# Convenience functions
def create_data_loader(data_dir: str) -> DataLoader:
    """Create a DataLoader instance"""
    return DataLoader(data_dir)


def load_csv_data(data_dir: str, filename: str, **kwargs) -> pd.DataFrame:
    """Quick function to load CSV data"""
    loader = DataLoader(data_dir)
    return loader.load_csv_file(filename, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the loader
    loader = create_data_loader("./test_data")
    
    # Get info
    info = loader.get_data_info()
    print(f"Data loader info: {info}")
    
    # List files
    files = loader.list_available_files()
    print(f"Available files: {files}")
    
    print("Base data loader ready for use!")
