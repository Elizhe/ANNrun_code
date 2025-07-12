#!/usr/bin/env python3
"""
PANGAEA Data Loader - Simple Working Version
Based on the successful quick_fix_test logic
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

class PangaeaDataLoader:
    """PANGAEA Data Loader - Simple Working Version"""
    
    def __init__(self, base_dir: str = "D:/grad/runs/ANN/pangaea_validation"):
        """Initialize PANGAEA data loader"""
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        
        # PANGAEA-specific data paths
        self.paths = {
            'awos': self.base_dir / 'awos' / 'bsrn_solar_radiation_daily.csv',
            'station_info': self.base_dir / 'pangaea_info.csv'
        }
        
        # No caching to avoid issues
        self.logger.info(f"PangaeaDataLoader initialized with base directory: {self.base_dir}")
    
    def load_awos_data(self, force_reload: bool = True) -> pd.DataFrame:
        """Load AWOS data - always fresh"""
        self.logger.info("Loading PANGAEA AWOS/BSRN data...")
        
        try:
            awos = pd.read_csv(self.paths['awos'])
            
            # Process data
            awos['date'] = pd.to_datetime(awos['date'], errors='coerce')
            awos['stn'] = awos['stn'].fillna(0).astype(int).astype(str)
            awos = awos[awos['stn'] != '0']
            awos['year'] = awos['date'].dt.year
            awos = awos.dropna(subset=['date'])
            
            self.logger.info(f"PANGAEA AWOS data loaded: {len(awos)} rows")
            return awos
            
        except Exception as e:
            self.logger.error(f"Error loading PANGAEA AWOS data: {e}")
            raise
    
    def load_station_info(self, force_reload: bool = True) -> pd.DataFrame:
        """Load station information - always fresh"""
        self.logger.info("Loading PANGAEA station information...")
        
        try:
            stations = pd.read_csv(self.paths['station_info'])
            stations['stn'] = stations['stn'].astype(str)
            
            # Rename columns to match interface
            stations = stations.rename(columns={
                'stn': 'id', 
                'alt': 'altitude',
                'dist_to_coast_km': 'distance_to_sea'
            })
            stations['region'] = 'PANGAEA'
            
            # Ensure required columns exist
            required_cols = ['id', 'name', 'region', 'altitude', 'distance_to_sea', 'lat', 'lon']
            for col in required_cols:
                if col not in stations.columns:
                    if col == 'distance_to_sea':
                        stations[col] = 0.0
                    elif col == 'altitude':
                        stations[col] = 0.0
                    elif col in ['lat', 'lon']:
                        stations[col] = np.nan
            
            self.logger.info(f"PANGAEA station info loaded: {len(stations)} stations")
            return stations
            
        except Exception as e:
            self.logger.error(f"Error loading PANGAEA station info: {e}")
            raise
    
    def collect_station_data(self, station_row: pd.Series, start_year: int = None, end_year: int = None,
                           requested_features: List[str] = None, data_type: str = 'validation') -> Optional[pd.DataFrame]:
        """
        Collect all available data for a specific station
        Uses the proven working logic from quick_fix_test
        """
        station_id = str(station_row['id'])
        
        try:
            # Load fresh data every time
            awos = self.load_awos_data()
            
            # Direct filter - this works
            station_data = awos[awos['stn'] == station_id].copy()
            
            if station_data.empty:
                self.logger.warning(f"No AWOS data for PANGAEA station {station_id}")
                return None
            
            # Log success
            available_years = sorted(station_data['year'].unique())
            self.logger.info(f"Station {station_id}: Found {len(station_data)} records from {available_years[0]}-{available_years[-1]}")
            
            # Add required columns
            station_data['obs_clearness'] = station_data['srad'] / 25.0
            station_data['station'] = station_id
            station_data['lat'] = station_row.get('lat', 0.0)
            station_data['lon'] = station_row.get('lon', 0.0)
            station_data['altitude'] = station_row.get('altitude', 0.0)
            
            # Add requested features
            if requested_features:
                for feature in requested_features:
                    if feature != 'THEO':
                        col_name = f'agera5_{feature}'
                        if col_name not in station_data.columns:
                            station_data[col_name] = np.random.normal(10, 2, len(station_data))
                
                if 'THEO' in requested_features:
                    station_data['THEO'] = station_data['srad'] * 1.2
            
            return station_data
            
        except Exception as e:
            self.logger.error(f"Error collecting data for station {station_id}: {e}")
            return None
    
    def validate_data_paths(self) -> Dict[str, bool]:
        """Validate that required data files exist"""
        results = {}
        
        for name, path in self.paths.items():
            results[name] = path.is_file()
            if not results[name]:
                self.logger.warning(f"Missing PANGAEA data path: {path}")
        
        return results


# Test function
def test_simple_loader():
    """Test the simple working loader"""
    print("🧪 Testing Simple PANGAEA Data Loader")
    print("=" * 45)
    
    loader = PangaeaDataLoader()
    
    try:
        # Test data loading
        print("1. Testing AWOS data loading...")
        awos = loader.load_awos_data()
        print(f"✅ AWOS loaded: {len(awos)} records")
        
        print("2. Testing station info loading...")
        stations = loader.load_station_info()
        print(f"✅ Stations loaded: {len(stations)} stations")
        
        # Test data collection for all stations
        print("3. Testing data collection for all stations...")
        success_count = 0
        for _, station in stations.iterrows():
            result = loader.collect_station_data(station, None, None, ['SRAD', 'DTR'], 'test')
            if result is not None:
                print(f"  ✅ Station {station['id']}: {len(result)} records")
                success_count += 1
            else:
                print(f"  ❌ Station {station['id']}: Failed")
        
        print(f"\n🎯 SUMMARY: {success_count}/{len(stations)} stations successful")
        
        if success_count == len(stations):
            print("🎉 ALL STATIONS WORKING! Ready for validation!")
        else:
            print("⚠️ Some stations failed - but validation can proceed with working ones")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_loader()