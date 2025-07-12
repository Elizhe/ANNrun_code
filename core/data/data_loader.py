#!/usr/bin/env python3
"""
Enhanced Data Loader for ANNrun_code
Supports AGERA5 and Himawari features with new naming convention:
- AGERA5_SRAD, AGERA5_TMAX, AGERA5_DTR, AGERA5_PREC, AGERA5_VPRE, AGERA5_WIND
- HIMA_SRAD, HIMA_TMAX, HIMA_DTR, HIMA_RAIN
- THEO (theoretical radiation)
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def validate_paths(self) -> Dict[str, bool]:
        """Validate data paths"""
        pass
        
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data"""
        pass


class Agera5Loader(BaseDataLoader):
    """AGERA5 data loader with enhanced feature mapping"""
    
    def __init__(self, base_dir: str = None):
        super().__init__(base_dir)
        self.agera5_dir = self.base_dir / 'data' / 'agera5'
        
        # Enhanced AGERA5 feature mapping
        self.feature_map = {
            'AGERA5_SRAD': 'Solar_Radiation_Flux',
            'AGERA5_TMIN': 'Temperature_Air_2m_Min_24h',
            'AGERA5_TMAX': 'Temperature_Air_2m_Max_24h',
            'AGERA5_DTR': 'temperature_range',  # Will be calculated
            'AGERA5_VPRE': 'Vapour_Pressure_Mean',
            'AGERA5_WIND': 'Wind_Speed_10m_Mean',
            'AGERA5_PREC': 'Precipitation_Flux'
        }
        
    def validate_paths(self) -> Dict[str, bool]:
        """Validate AGERA5 data paths"""
        return {'agera5_dir': self.agera5_dir.exists()}
    
    def load_data(self, station_id: str, start_year: int, end_year: int, 
                  features: List[str]) -> Optional[pd.DataFrame]:
        """Load AGERA5 data for specified features"""
        
        # Filter only AGERA5 features
        agera5_features = [f for f in features if f.startswith('AGERA5_')]
        
        if not agera5_features:
            return None
            
        ag_list = []
        
        for year in range(start_year, end_year + 1):
            file_path = self.agera5_dir / f"{station_id}_{year}.csv"
            
            if not file_path.exists():
                continue
                
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                ag_list.append(df)
                
            except Exception as e:
                self.logger.warning(f"Error loading AGERA5 file {file_path}: {e}")
                continue
        
        if not ag_list:
            return None
            
        # Combine all years
        agera5_df = pd.concat(ag_list, ignore_index=True)
        
        # Calculate DTR if needed
        if 'AGERA5_DTR' in agera5_features:
            if ('Temperature_Air_2m_Max_24h' in agera5_df.columns and 
                'Temperature_Air_2m_Min_24h' in agera5_df.columns):
                agera5_df['temperature_range'] = (
                    agera5_df['Temperature_Air_2m_Max_24h'] - 
                    agera5_df['Temperature_Air_2m_Min_24h']
                )
        
        # Rename columns to standard format
        rename_dict = {}
        for feature in agera5_features:
            if feature in self.feature_map:
                original_col = self.feature_map[feature]
                if original_col in agera5_df.columns:
                    rename_dict[original_col] = feature
        
        agera5_df = agera5_df.rename(columns=rename_dict)
        
        # Select only requested features plus date
        columns_to_keep = ['date'] + [f for f in agera5_features if f in agera5_df.columns]
        agera5_df = agera5_df[columns_to_keep]
        
        return agera5_df


class HimawariLoader(BaseDataLoader):
    """Himawari data loader with enhanced feature support"""
    
    def __init__(self, base_dir: str = None):
        super().__init__(base_dir)
        self.hima_dir = self.base_dir / 'data' / 'hima'
        
        # Himawari feature mapping
        self.feature_map = {
            'HIMA_SRAD': 'SRAD',
            'HIMA_TMAX': 'TMAX', 
            'HIMA_TMIN': 'TMIN',
            'HIMA_DTR': 'DTR',  # Will be calculated
            'HIMA_RAIN': 'RAIN'  # Atmospheric moisture content
        }
        
    def validate_paths(self) -> Dict[str, bool]:
        """Validate Himawari data paths"""
        return {'hima_dir': self.hima_dir.exists()}
    
    def load_data(self, station_id: str, start_year: int, end_year: int,
                  features: List[str]) -> Optional[pd.DataFrame]:
        """Load Himawari data for specified features"""
        
        # Filter only Himawari features
        hima_features = [f for f in features if f.startswith('HIMA_')]
        
        if not hima_features:
            return None
            
        hima_list = []
        
        for year in range(start_year, end_year + 1):
            file_path = self.hima_dir / f"{station_id}_{year}.wth"
            
            if not file_path.exists():
                continue
                
            try:
                df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
                
                # Handle date format (Julian date YYDDD)
                if 'DATE' in df.columns:
                    date_str = '20' + df['DATE'].astype(str).str[:2] + df['DATE'].astype(str).str[2:].str.zfill(3)
                    df['date'] = pd.to_datetime(date_str, format='%Y%j', errors='coerce')
                
                hima_list.append(df)
                
            except Exception as e:
                self.logger.warning(f"Error loading Himawari file {file_path}: {e}")
                continue
        
        if not hima_list:
            return None
            
        # Combine all years
        hima_df = pd.concat(hima_list, ignore_index=True)
        
        # Calculate DTR if needed
        if 'HIMA_DTR' in hima_features:
            if 'TMAX' in hima_df.columns and 'TMIN' in hima_df.columns:
                hima_df['DTR'] = hima_df['TMAX'] - hima_df['TMIN']
        
        # Rename columns to standard format
        rename_dict = {}
        for feature in hima_features:
            if feature in self.feature_map:
                original_col = self.feature_map[feature]
                if original_col in hima_df.columns:
                    rename_dict[original_col] = feature
        
        hima_df = hima_df.rename(columns=rename_dict)
        
        # Select only requested features plus date
        columns_to_keep = ['date'] + [f for f in hima_features if f in hima_df.columns]
        hima_df = hima_df[columns_to_keep]
        
        return hima_df


class TheoreticalRadiationLoader(BaseDataLoader):
    """Theoretical radiation data loader"""
    
    def __init__(self, base_dir: str = None):
        super().__init__(base_dir)
        self.rad_file = self.base_dir / 'data' / 'rad_est' / 'daily_radiation_estimates.csv'
        self._cache = None
        
    def validate_paths(self) -> Dict[str, bool]:
        """Validate theoretical radiation file"""
        return {'rad_file': self.rad_file.exists()}
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """Load theoretical radiation data"""
        
        if self._cache is not None and not force_reload:
            return self._cache
            
        try:
            rad_df = pd.read_csv(self.rad_file)
            rad_df['date'] = pd.to_datetime(rad_df['date'], errors='coerce')
            
            # Standardize column name
            if 'daily_radiation' in rad_df.columns:
                rad_df = rad_df.rename(columns={'daily_radiation': 'THEO'})
            elif 'theoretical' in rad_df.columns:
                rad_df = rad_df.rename(columns={'theoretical': 'THEO'})
            
            # Cache and return
            self._cache = rad_df
            return rad_df
            
        except Exception as e:
            self.logger.error(f"Error loading theoretical radiation: {e}")
            raise


class EnhancedDataManager:
    """Enhanced data manager that combines all data sources"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize loaders
        self.agera5_loader = Agera5Loader(base_dir)
        self.hima_loader = HimawariLoader(base_dir)
        self.theo_loader = TheoreticalRadiationLoader(base_dir)
        
        # Load station info and AWOS data
        self.stations_df = self._load_station_info()
        self.awos_df = self._load_awos_data()
        
    def _load_station_info(self) -> pd.DataFrame:
        """Load station information"""
        station_file = self.base_dir / 'data' / 'latlon.csv'
        try:
            return pd.read_csv(station_file)
        except Exception as e:
            self.logger.error(f"Error loading station info: {e}")
            return pd.DataFrame()
    
    def _load_awos_data(self) -> pd.DataFrame:
        """Load AWOS observation data"""
        awos_file = self.base_dir / 'data' / 'awos' / 'bsrn_solar_radiation_daily.csv'
        try:
            awos_df = pd.read_csv(awos_file)
            awos_df['date'] = pd.to_datetime(awos_df['date'], errors='coerce')
            return awos_df
        except Exception as e:
            self.logger.error(f"Error loading AWOS data: {e}")
            return pd.DataFrame()
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """Get all available features by data source"""
        return {
            'AGERA5': ['AGERA5_SRAD', 'AGERA5_TMAX', 'AGERA5_DTR', 'AGERA5_PREC', 'AGERA5_VPRE', 'AGERA5_WIND'],
            'HIMAWARI': ['HIMA_SRAD', 'HIMA_TMAX', 'HIMA_DTR', 'HIMA_RAIN'],
            'THEORETICAL': ['THEO']
        }
    
    def validate_feature_combination(self, features: List[str]) -> Dict[str, bool]:
        """Validate if feature combination is available"""
        available = self.get_available_features()
        all_features = []
        for source_features in available.values():
            all_features.extend(source_features)
        
        validation = {}
        for feature in features:
            validation[feature] = feature in all_features
            
        return validation
    
    def load_combined_data(self, station_id: str, start_year: int, end_year: int,
                          features: List[str], target_column: str = 'obs') -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and combine data from all sources
        
        Args:
            station_id: Station ID
            start_year: Start year
            end_year: End year  
            features: List of requested features
            target_column: Target column name
            
        Returns:
            Tuple of (combined_dataframe, feature_columns)
        """
        
        self.logger.info(f"Loading data for station {station_id}, years {start_year}-{end_year}")
        self.logger.info(f"Requested features: {features}")
        
        # Validate features
        validation = self.validate_feature_combination(features)
        invalid_features = [f for f, valid in validation.items() if not valid]
        if invalid_features:
            self.logger.warning(f"Invalid features: {invalid_features}")
        
        # Load AWOS observations (target)
        station_awos = self.awos_df[self.awos_df['stn'] == int(station_id)].copy()
        if station_awos.empty:
            raise ValueError(f"No AWOS data found for station {station_id}")
        
        # Get station info
        station_info = self.stations_df[self.stations_df['id'] == station_id]
        if station_info.empty:
            raise ValueError(f"Station {station_id} not found in station info")
        
        station_row = station_info.iloc[0]
        
        # Start with AWOS data
        combined_data = station_awos[['date', 'obs']].copy()
        
        # Load AGERA5 data
        agera5_data = self.agera5_loader.load_data(station_id, start_year, end_year, features)
        if agera5_data is not None:
            combined_data = pd.merge(combined_data, agera5_data, on='date', how='left')
            self.logger.info(f"AGERA5 data merged: {len(agera5_data)} rows")
        
        # Load Himawari data
        hima_data = self.hima_loader.load_data(station_id, start_year, end_year, features)
        if hima_data is not None:
            combined_data = pd.merge(combined_data, hima_data, on='date', how='left')
            self.logger.info(f"Himawari data merged: {len(hima_data)} rows")
        
        # Load theoretical radiation if requested
        if 'THEO' in features:
            theo_data = self.theo_loader.load_data()
            if not theo_data.empty:
                theo_station = theo_data[theo_data['station_id'] == station_id]
                if not theo_station.empty:
                    combined_data = pd.merge(combined_data, theo_station[['date', 'THEO']], 
                                           on='date', how='left')
                    self.logger.info(f"Theoretical radiation merged: {len(theo_station)} rows")
        
        # Add station metadata
        combined_data['station_id'] = station_id
        combined_data['station_name'] = station_row.get('name', f'Station_{station_id}')
        
        # Filter valid data
        combined_data = combined_data.dropna(subset=[target_column])
        
        # Get actual feature columns present in data
        feature_columns = [col for col in features if col in combined_data.columns]
        
        self.logger.info(f"Combined data shape: {combined_data.shape}")
        self.logger.info(f"Available feature columns: {feature_columns}")
        
        return combined_data, feature_columns
    
    def prepare_model_data(self, combined_data: pd.DataFrame, feature_columns: List[str], 
                          target_column: str = 'obs') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training
        
        Args:
            combined_data: Combined dataframe
            feature_columns: Feature column names
            target_column: Target column name
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        
        # Check for required columns
        missing_features = [col for col in feature_columns if col not in combined_data.columns]
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")
            feature_columns = [col for col in feature_columns if col in combined_data.columns]
        
        if not feature_columns:
            raise ValueError("No valid feature columns found")
        
        if target_column not in combined_data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Extract features and target
        X = combined_data[feature_columns].values
        y = combined_data[target_column].values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Model data prepared - X: {X.shape}, y: {y.shape}")
        self.logger.info(f"Feature columns: {feature_columns}")
        
        return X, y, feature_columns


# Convenience function for backward compatibility
def create_enhanced_data_manager(base_dir: str = None) -> EnhancedDataManager:
    """Create enhanced data manager instance"""
    return EnhancedDataManager(base_dir)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test enhanced data manager
    manager = EnhancedDataManager()
    
    # Show available features
    available = manager.get_available_features()
    print("Available features:")
    for source, features in available.items():
        print(f"  {source}: {features}")
    
    # Test feature validation
    test_features = ['AGERA5_SRAD', 'HIMA_SRAD', 'AGERA5_TMAX', 'HIMA_TMAX', 'THEO']
    validation = manager.validate_feature_combination(test_features)
    print(f"\nFeature validation: {validation}")
    
    print("\nEnhanced data loader ready for use!")
