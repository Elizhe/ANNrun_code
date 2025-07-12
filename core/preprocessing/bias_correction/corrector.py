#!/usr/bin/env python3
# bias.py - Bias Correction Methods for Solar Radiation Models
# Provides selective bias correction features for LUMEN model

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class BiasCorrection:
    """Bias correction methods for solar radiation prediction"""
    
    def __init__(self):
        """Initialize bias correction system"""
        self.logger = logging.getLogger(__name__)
        self.enabled_methods = []
        self.feature_cache = {}
        
        # Available bias correction methods
        self.available_methods = {
            'clearness': self.add_clearness_bias,
            'geographic': self.add_geographic_bias,
            'coastal_distance': self.add_coastal_distance_bias,
            'radiation_flags': self.add_radiation_flags,
            'solar_angles': self.add_solar_angles,
            'weather_flags': self.add_weather_flags
        }
    
    def configure(self, bias_methods: List[str]):
        """
        Configure which bias correction methods to use
        
        Args:
            bias_methods: List of method names to enable
        """
        self.enabled_methods = []
        
        for method in bias_methods:
            method = method.strip().lower()
            if method in self.available_methods:
                self.enabled_methods.append(method)
                self.logger.info(f"Enabled bias correction: {method}")
            else:
                self.logger.warning(f"Unknown bias correction method: {method}")
                self.logger.info(f"Available methods: {list(self.available_methods.keys())}")
        
        if not self.enabled_methods:
            self.logger.info("No bias correction methods enabled")
    
    def apply_corrections(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Apply all configured bias corrections to data
        
        Args:
            df: Station data DataFrame
            station_row: Station information
            
        Returns:
            DataFrame with bias correction features added
        """
        if not self.enabled_methods:
            return df
        
        # Reset index to ensure proper alignment
        df = df.reset_index(drop=True)
        
        # Apply each enabled method
        for method in self.enabled_methods:
            try:
                correction_func = self.available_methods[method]
                df = correction_func(df, station_row)
                
            except Exception as e:
                self.logger.error(f"Error applying {method} bias correction: {e}")
                continue
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get names of bias correction features that exist in the DataFrame
        
        Args:
            df: DataFrame to check for bias features
            
        Returns:
            List of bias feature names
        """
        bias_features = []
        
        # Define all possible bias features by method
        all_bias_features = {
            'clearness': [
                'hima_vs_theoretical_ratio', 'hima_bias',
                'agera5_vs_theoretical_ratio', 'agera5_bias',
                'clearness_difference', 'clearness_ratio'
            ],
            'geographic': [
                'latitude_bias_factor', 'longitude_bias_factor',
                'altitude_bias_factor', 'region_bias_factor'
            ],
            'coastal_distance': [
                'coastal_0_1km', 'coastal_1_5km', 'coastal_5_50km', 'coastal_50km_plus',
                'coastal_bias_factor', 'coastal_clearness_interaction'
            ],
            'radiation_flags': [
                'high_radiation_flag', 'low_radiation_flag',
                'very_clear', 'cloudy', 'moderate_radiation'
            ],
            'solar_angles': [
                'solar_zenith_angle', 'solar_elevation_angle',
                'declination_angle', 'solar_factor'
            ],
            'weather_flags': [
                'rainy_day', 'no_rain', 'hot_day', 'cold_day',
                'high_dtr', 'weather_bias_factor'
            ]
        }
        
        # Check which bias features exist in the dataframe
        for method in self.enabled_methods:
            if method in all_bias_features:
                for feature in all_bias_features[method]:
                    if feature in df.columns:
                        bias_features.append(feature)
        
        return bias_features
    
    def add_clearness_bias(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        try:
            # 1. Source comparison features (NO target information)
            if 'hima_clearness' in df.columns and 'agera5_clearness' in df.columns:
                # Ratio between s ources
                agera5_safe = df['agera5_clearness'].fillna(1.0).replace(0, 1e-6)
                df.loc[:, 'agera5_hima_ratio'] = df['hima_clearness'].fillna(1.0) / agera5_safe
                
                # Difference between sources (AGERA5 - Himawari)
                df.loc[:, 'clearness_diff'] = df['agera5_clearness'] - df['hima_clearness']
                df.loc[:, 'clearness_abs_diff'] = np.abs(df['clearness_diff'])
                
                # Average of two sources
                df.loc[:, 'clearness_mean'] = (df['hima_clearness'] + df['agera5_clearness']) / 2
                
                # AGERA5 overestimation indicators
                df.loc[:, 'agera5_overestimate'] = (df['agera5_clearness'] > df['hima_clearness']).astype(int)
                df.loc[:, 'agera5_overestimate_magnitude'] = np.maximum(0, df['clearness_diff'])
                
                # Disagreement levels
                df.loc[:, 'high_disagreement'] = (df['clearness_abs_diff'] > 0.2).astype(int)
                df.loc[:, 'moderate_disagreement'] = ((df['clearness_abs_diff'] > 0.1) & 
                                                       (df['clearness_abs_diff'] <= 0.2)).astype(int)
                
                # Source confidence (agreement = higher confidence)
                df.loc[:, 'source_agreement'] = 1 - np.minimum(1, df['clearness_abs_diff'])
            
            # 2. Conditional bias patterns (based on clearness levels)
            if 'hima_clearness' in df.columns and 'clearness_diff' in df.columns:
                # Clear sky conditions (high clearness)
                clear_mask = df['hima_clearness'] > 0.7
                df.loc[:, 'clear_sky_diff'] = df['clearness_diff'] * clear_mask.astype(int)
                df.loc[:, 'clear_sky_agera5_bias'] = df['agera5_overestimate_magnitude'] * clear_mask.astype(int)
                
                # Cloudy conditions (low clearness)
                cloudy_mask = df['hima_clearness'] < 0.3
                df.loc[:, 'cloudy_sky_diff'] = df['clearness_diff'] * cloudy_mask.astype(int)
                df.loc[:, 'cloudy_sky_agera5_bias'] = df['agera5_overestimate_magnitude'] * cloudy_mask.astype(int)
                
                # Intermediate conditions
                intermediate_mask = (df['hima_clearness'] >= 0.3) & (df['hima_clearness'] <= 0.7)
                df.loc[:, 'intermediate_sky_diff'] = df['clearness_diff'] * intermediate_mask.astype(int)
            
            # 3. Weighted ensemble features
            if 'source_agreement' in df.columns:
                # Weight Himawari more when sources agree (more reliable)
                # Weight AGERA5 less when it overestimates
                hima_weight = 0.6 + 0.2 * df['source_agreement'] - 0.1 * df['agera5_overestimate']
                agera5_weight = 1 - hima_weight
                
                df.loc[:, 'weighted_clearness'] = (
                    df['hima_clearness'] * hima_weight + 
                    df['agera5_clearness'] * agera5_weight
                )
                
                # Confidence-weighted features
                df.loc[:, 'confident_hima'] = df['hima_clearness'] * df['source_agreement']
                df.loc[:, 'confident_agera5'] = df['agera5_clearness'] * df['source_agreement']
            
            # 4. Temporal consistency features (if we have enough data)
            if len(df) > 10:
                # Rolling statistics to capture temporal patterns
                window = 5
                df.loc[:, 'hima_rolling_std'] = df['hima_clearness'].rolling(window, center=True).std().fillna(0)
                df.loc[:, 'agera5_rolling_std'] = df['agera5_clearness'].rolling(window, center=True).std().fillna(0)
                df.loc[:, 'diff_rolling_mean'] = df['clearness_diff'].rolling(window, center=True).mean().fillna(0)
            
            # 5. Extreme value handling
            if 'hima_clearness' in df.columns:
                # Flag extreme values that might indicate errors
                df.loc[:, 'hima_extreme_low'] = (df['hima_clearness'] < 0.1).astype(int)
                df.loc[:, 'hima_extreme_high'] = (df['hima_clearness'] > 0.9).astype(int)
                df.loc[:, 'agera5_extreme_low'] = (df['agera5_clearness'] < 0.1).astype(int)
                df.loc[:, 'agera5_extreme_high'] = (df['agera5_clearness'] > 0.9).astype(int)
            
            self.logger.debug("Added fixed clearness bias features (no data leakage)")
            
        except Exception as e:
            self.logger.error(f"Error in fixed clearness bias correction: {e}")
        
        return df

    
    def add_geographic_bias(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Add geographic bias correction features
        
        Args:
            df: Station data
            station_row: Station information with location data
            
        Returns:
            DataFrame with geographic bias features
        """
        try:
            # Get station location
            if 'latitude' in station_row and pd.notna(station_row['latitude']):
                station_lat = float(station_row['latitude'])
            else:
                station_lat = 37.0 if station_row['region'] == 'Korea' else 35.0
            
            if 'longitude' in station_row and pd.notna(station_row['longitude']):
                station_lon = float(station_row['longitude'])
            else:
                station_lon = 127.0 if station_row['region'] == 'Korea' else 135.0
            
            altitude = float(station_row.get('altitude', 0))
            
            # Geographic bias factors (interaction with clearness)
            if 'hima_clearness' in df.columns:
                df.loc[:, 'latitude_bias_factor'] = station_lat * df['hima_clearness']
                df.loc[:, 'longitude_bias_factor'] = station_lon * df['hima_clearness']
                df.loc[:, 'altitude_bias_factor'] = altitude * df['hima_clearness']
            
            # Region-specific bias
            region_factor = 1.0 if station_row['region'] == 'Korea' else 0.0
            if 'hima_clearness' in df.columns:
                df.loc[:, 'region_bias_factor'] = region_factor * df['hima_clearness']
            
            self.logger.debug("Added geographic bias features")
            
        except Exception as e:
            self.logger.error(f"Error in geographic bias correction: {e}")
        
        return df
    
    def add_coastal_distance_bias(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Add coastal distance bias correction features
        
        Args:
            df: Station data
            station_row: Station information with distance_to_sea
            
        Returns:
            DataFrame with coastal distance bias features
        """
        try:
            distance_to_sea = float(station_row.get('distance_to_sea', 50.0))
            
            # Coastal distance categories (as binary features)
            df.loc[:, 'coastal_0_1km'] = 1.0 if distance_to_sea <= 1.0 else 0.0
            df.loc[:, 'coastal_1_5km'] = 1.0 if 1.0 < distance_to_sea <= 5.0 else 0.0
            df.loc[:, 'coastal_5_50km'] = 1.0 if 5.0 < distance_to_sea <= 50.0 else 0.0
            df.loc[:, 'coastal_50km_plus'] = 1.0 if distance_to_sea > 50.0 else 0.0
            
            # Coastal bias factor (interaction with clearness)
            if 'hima_clearness' in df.columns:
                df.loc[:, 'coastal_bias_factor'] = distance_to_sea * df['hima_clearness']
                df.loc[:, 'coastal_clearness_interaction'] = df['hima_clearness'] * (1.0 / (1.0 + distance_to_sea))
            
            self.logger.debug("Added coastal distance bias features")
            
        except Exception as e:
            self.logger.error(f"Error in coastal distance bias correction: {e}")
        
        return df
    
    def add_radiation_flags(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Add radiation condition flags
        
        Args:
            df: Station data with theoretical radiation
            station_row: Station information
            
        Returns:
            DataFrame with radiation flag features
        """
        try:
            if 'theoretical' not in df.columns:
                return df
            
            # Radiation level flags
            theoretical_q30 = df['theoretical'].quantile(0.3)
            theoretical_q70 = df['theoretical'].quantile(0.7)
            
            df.loc[:, 'low_radiation_flag'] = (df['theoretical'] < theoretical_q30).astype(int)
            df.loc[:, 'high_radiation_flag'] = (df['theoretical'] > theoretical_q70).astype(int)
            df.loc[:, 'moderate_radiation'] = ((df['theoretical'] >= theoretical_q30) & 
                                              (df['theoretical'] <= theoretical_q70)).astype(int)
            
            # Clear sky condition flags
            if 'hima_clearness' in df.columns:
                df.loc[:, 'very_clear'] = (df['hima_clearness'] > 0.8).astype(int)
                df.loc[:, 'cloudy'] = (df['hima_clearness'] < 0.5).astype(int)
            
            self.logger.debug("Added radiation flag features")
            
        except Exception as e:
            self.logger.error(f"Error in radiation flags: {e}")
        
        return df
    
    def add_solar_angles(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Add solar angle features
        
        Args:
            df: Station data with date information
            station_row: Station information with location
            
        Returns:
            DataFrame with solar angle features
        """
        try:
            if 'date' not in df.columns:
                return df
            
            # Get station location
            if 'latitude' in station_row and pd.notna(station_row['latitude']):
                station_lat = float(station_row['latitude'])
            else:
                station_lat = 37.0 if station_row['region'] == 'Korea' else 35.0
            
            # Calculate day of year
            day_of_year = df['date'].dt.dayofyear.values
            
            # Solar angle calculations
            solar_angles = self._calculate_solar_angles(station_lat, day_of_year)
            
            df.loc[:, 'solar_zenith_angle'] = solar_angles['zenith']
            df.loc[:, 'solar_elevation_angle'] = solar_angles['elevation']
            df.loc[:, 'declination_angle'] = solar_angles['declination']
            
            # Solar factor (combination of angles)
            df.loc[:, 'solar_factor'] = np.sin(np.radians(solar_angles['elevation']))
            
            self.logger.debug("Added solar angle features")
            
        except Exception as e:
            self.logger.error(f"Error in solar angles: {e}")
        
        return df
    
    def add_weather_flags(self, df: pd.DataFrame, station_row: pd.Series) -> pd.DataFrame:
        """
        Add weather condition flags
        
        Args:
            df: Station data with AGERA5 weather variables
            station_row: Station information
            
        Returns:
            DataFrame with weather flag features
        """
        try:
            # Precipitation flags
            if 'agera5_PREC' in df.columns:
                prec_data = pd.to_numeric(df['agera5_PREC'], errors='coerce')
                df.loc[:, 'rainy_day'] = (prec_data > 10.0).astype(int)
                df.loc[:, 'no_rain'] = (prec_data == 0).astype(int)
            
            # Temperature flags
            if 'agera5_TMAX' in df.columns:
                tmax_data = pd.to_numeric(df['agera5_TMAX'], errors='coerce')
                df.loc[:, 'hot_day'] = (tmax_data > 30.0).astype(int)
                df.loc[:, 'cold_day'] = (tmax_data < 10.0).astype(int)
            
            # DTR flag
            if 'agera5_DTR' in df.columns:
                dtr_data = pd.to_numeric(df['agera5_DTR'], errors='coerce')
                if len(dtr_data.dropna()) > 0:
                    dtr_q70 = dtr_data.quantile(0.7)
                    df.loc[:, 'high_dtr'] = (dtr_data > dtr_q70).astype(int)
            
            # Weather bias factor (interaction with clearness)
            if 'hima_clearness' in df.columns:
                weather_conditions = 0
                if 'rainy_day' in df.columns:
                    weather_conditions += df['rainy_day']
                if 'hot_day' in df.columns:
                    weather_conditions += df['hot_day']
                if 'cold_day' in df.columns:
                    weather_conditions += df['cold_day']
                
                df.loc[:, 'weather_bias_factor'] = weather_conditions * df['hima_clearness']
            
            self.logger.debug("Added weather flag features")
            
        except Exception as e:
            self.logger.error(f"Error in weather flags: {e}")
        
        return df
    
    def _calculate_solar_angles(self, latitude: float, day_of_year: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate solar angles for given latitude and day of year
        
        Args:
            latitude: Station latitude in degrees
            day_of_year: Array of day of year values
            
        Returns:
            Dictionary with solar angle arrays
        """
        # Convert latitude to radians
        lat_rad = np.radians(latitude)
        
        # Solar declination angle
        declination = np.radians(23.45) * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Solar elevation angle at solar noon
        solar_elevation = np.arcsin(
            np.sin(lat_rad) * np.sin(declination) + 
            np.cos(lat_rad) * np.cos(declination)
        )
        
        # Solar zenith angle
        solar_zenith = np.pi/2 - solar_elevation
        
        return {
            'zenith': np.degrees(solar_zenith),
            'elevation': np.degrees(solar_elevation),
            'declination': np.degrees(declination)
        }
    
    def get_method_summary(self) -> Dict[str, str]:
        """
        Get summary of available bias correction methods
        
        Returns:
            Dictionary with method descriptions
        """
        return {
            'clearness': 'Clearness index ratio and bias features',
            'geographic': 'Latitude, longitude, altitude bias factors',
            'coastal_distance': 'Coastal distance categories and interactions',
            'radiation_flags': 'High/low radiation and clear sky flags',
            'solar_angles': 'Solar zenith, elevation, and declination angles',
            'weather_flags': 'Precipitation, temperature, and DTR flags'
        }
    
    def validate_configuration(self, df: pd.DataFrame) -> bool:
        """
        Validate that the current configuration can be applied to the given data
        
        Args:
            df: DataFrame to validate against
            
        Returns:
            True if configuration is valid
        """
        required_columns = {
            'clearness': ['hima_clearness', 'obs_clearness'],
            'geographic': ['hima_clearness'],
            'coastal_distance': ['distance_to_sea'],
            'radiation_flags': ['theoretical'],
            'solar_angles': ['date'],
            'weather_flags': ['agera5_PREC', 'agera5_TMAX']
        }
        
        valid = True
        
        for method in self.enabled_methods:
            if method in required_columns:
                missing = [col for col in required_columns[method] if col not in df.columns]
                if missing:
                    self.logger.warning(f"Method '{method}' missing required columns: {missing}")
                    valid = False
        
        return valid

class BiasCorrector:
    """
    Bias Correction class for ANNrun_code
    Handles various bias correction methods
    """
    
    def __init__(self, method='linear', **kwargs):
        """Initialize bias corrector"""
        self.method = method
        self.config = kwargs
        self.is_fitted = False
        self.correction_params = {}
    
    def fit(self, observed, predicted, **kwargs):
        """Fit bias correction parameters"""
        import numpy as np
        
        if self.method == 'linear':
            # Simple linear bias correction
            diff = np.array(observed) - np.array(predicted)
            self.correction_params = {
                'mean_bias': np.mean(diff),
                'std_ratio': np.std(observed) / np.std(predicted) if np.std(predicted) > 0 else 1.0
            }
        elif self.method == 'quantile':
            # Quantile mapping
            self.correction_params = {
                'observed_quantiles': np.percentile(observed, np.arange(0, 101, 5)),
                'predicted_quantiles': np.percentile(predicted, np.arange(0, 101, 5))
            }
        else:
            # Default: simple mean bias correction
            self.correction_params = {
                'mean_bias': np.mean(np.array(observed) - np.array(predicted))
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, predicted, **kwargs):
        """Apply bias correction"""
        if not self.is_fitted:
            raise ValueError("BiasCorrector must be fitted before transform")
        
        import numpy as np
        predicted = np.array(predicted)
        
        if self.method == 'linear':
            corrected = predicted + self.correction_params['mean_bias']
            corrected = corrected * self.correction_params['std_ratio']
        elif self.method == 'quantile':
            # Simple quantile mapping
            corrected = np.interp(
                predicted,
                self.correction_params['predicted_quantiles'],
                self.correction_params['observed_quantiles']
            )
        else:
            # Default: simple bias correction
            corrected = predicted + self.correction_params['mean_bias']
        
        return corrected
    
    def fit_transform(self, observed, predicted, **kwargs):
        """Fit and transform in one step"""
        return self.fit(observed, predicted, **kwargs).transform(predicted, **kwargs)
    
    def get_correction_info(self):
        """Get bias correction information"""
        return {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'parameters': self.correction_params
        }


# For backward compatibility
def create_bias_corrector(method='linear', **kwargs):
    """Create bias corrector instance"""
    return BiasCorrector(method=method, **kwargs)
