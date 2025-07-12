#!/usr/bin/env python3
"""
Minimal Data Loader - Safe fallback version
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

class DataLoader:
    """Minimal safe data loader"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv_file(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load CSV file safely"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            default_params = {
                'encoding': 'utf-8',
                'skipinitialspace': True,
                'na_values': ['', 'NULL', 'null', 'None', 'none', 'NaN', 'nan']
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(filepath, **default_params)
            self.logger.info(f"Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {filename}: {e}")
            raise
    
    def safe_divide(self, numerator: Any, denominator: Any) -> np.ndarray:
        """Safe division with proper error handling"""
        num_array = np.asarray(numerator, dtype=np.float64)
        den_array = np.asarray(denominator, dtype=np.float64)
        
        # Avoid division by zero
        result = np.divide(
            num_array, 
            den_array, 
            out=np.full_like(num_array, np.nan, dtype=np.float64), 
            where=(den_array != 0)
        )
        
        return result
    
    def calculate_clearness_index(self, obs_radiation: Any, theo_radiation: Any = 25.0) -> np.ndarray:
        """Calculate clearness index safely"""
        return self.safe_divide(obs_radiation, theo_radiation)
    
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load data - basic implementation"""
        return {}
    
    def validate_data(self, data: Any) -> bool:
        """Validate data - basic implementation"""
        return data is not None

# Backward compatibility
class Agera5Loader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)

class HimawariLoader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)

class PangaeaDataLoader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)
    
    def load_awos_data(self):
        return pd.DataFrame()
    
    def load_station_info(self):
        return pd.DataFrame()

# Legacy support
def create_data_loader(data_dir: str):
    return DataLoader(data_dir)

if __name__ == "__main__":
    print("✅ Minimal data loader ready")
