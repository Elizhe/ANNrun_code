#!/usr/bin/env python3
# LUMEN.py - ANN Model Implementation with Selective Bias Correction and THEO support
# Handles model training, validation, and prediction with bias features

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

class LUMENModel:
    """LUMEN ANN Model with selective bias correction and THEO support"""
    
    def __init__(self, normalization='decimal', bias_corrector=None, output_dir=None):
        """
        Initialize LUMEN model
        
        Args:
            normalization: Normalization method ('decimal', 'minmax', 'standard', 'none')
            bias_corrector: BiasCorrection instance from bias.py
            output_dir: Directory to save model and results
        """
        self.normalization = normalization
        self.bias_corrector = bias_corrector
        self.output_dir = Path(output_dir) if output_dir else Path("./model_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = None
        self.decimal_params = None
        self.feature_names = None
        
        # AGERA5 feature mapping - Added THEO
        self.alias_map = {
            'SRAD': 'Solar_Radiation_Flux',
            'TMIN': 'Temperature_Air_2m_Min_24h',
            'TMAX': 'Temperature_Air_2m_Max_24h',
            'DTR': 'temperature_range',
            'VPRE': 'Vapour_Pressure_Mean',
            'WIND': 'Wind_Speed_10m_Mean',
            'PREC': 'Precipitation_Flux',
            'THEO': 'theoretical'  # Direct mapping for theoretical radiation
        }
        
        self.logger = logging.getLogger(__name__)
    
    def build_ann_model(self, input_dim):
        """Build ANN architecture"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def apply_decimal_normalization(self, X, is_training=True):
        """Apply decimal normalization - Paper best performance method"""
        if is_training:
            # Calculate decimal factors for each feature
            X_values = X.values if hasattr(X, 'values') else X
            X_abs_max = np.abs(X_values).max(axis=0)
            
            # Handle small values (like clearness indices)
            X_abs_max_safe = np.maximum(X_abs_max, 1e-3)
            decimal_factors = np.ceil(np.log10(X_abs_max_safe + 1e-8)).astype(int)
            decimal_factors = np.maximum(decimal_factors, 0)  # Ensure non-negative
            
            # Save parameters
            self.decimal_params = {
                'factors': decimal_factors.tolist(),
                'X_abs_max': X_abs_max.tolist(),
                'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
            }
            
            # Apply normalization
            X_normalized = X_values / (10.0 ** decimal_factors)
            
            # Save parameters
            joblib.dump(self.decimal_params, self.output_dir / 'decimal_params.pkl')
            
            self.logger.debug(f"Decimal normalization - factors: {decimal_factors}")
            return X_normalized
            
        else:
            # For prediction: use saved factors
            if self.decimal_params is not None:
                decimal_factors = np.array(self.decimal_params['factors'])
                decimal_factors = np.maximum(decimal_factors, 0)
                X_values = X.values if hasattr(X, 'values') else X
                return X_values / (10.0 ** decimal_factors)
            else:
                self.logger.warning("Decimal parameters not found for prediction")
                return X.values if hasattr(X, 'values') else X
    
    def apply_normalization(self, X, is_training=True):
        """Apply selected normalization method"""
        if self.normalization == 'decimal':
            return self.apply_decimal_normalization(X, is_training)
            
        elif self.normalization == 'minmax':
            if is_training:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
                joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
                return X_scaled
            else:
                if self.scaler is not None:
                    return self.scaler.transform(X)
                else:
                    self.logger.warning("MinMax scaler not found for prediction")
                    return X.values if hasattr(X, 'values') else X
                    
        elif self.normalization == 'standard':
            if is_training:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
                return X_scaled
            else:
                if self.scaler is not None:
                    return self.scaler.transform(X)
                else:
                    self.logger.warning("Standard scaler not found for prediction")
                    return X.values if hasattr(X, 'values') else X
        
        else:  # 'none'
            return X.values if hasattr(X, 'values') else X
    
    def collect_station_data(self, station_row, data_loader, features, start_year, end_year):
        """Collect and prepare data for a single station"""
        try:
            # Use data_loader to collect basic station data
            df = data_loader.collect_station_data(
                station_row, start_year, end_year, features, 'train'
            )
            
            if df is None or len(df) == 0:
                return None
            
            # Apply bias corrections if configured
            if self.bias_corrector:
                df = self.bias_corrector.apply_corrections(df, station_row)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting data for station {station_row['id']}: {e}")
            return None
    
    def prepare_features(self, df, features):
        """Prepare feature matrix from collected data - Enhanced for THEO support"""
        # Base features (from clearness-based model)
        base_features = [
            'hima_clearness', 'distance_to_sea'
        ]
        
        # Add AGERA5 clearness if SRAD is requested
        if 'SRAD' in features:
            base_features.append('agera5_clearness')
        
        # Add theoretical radiation if THEO is requested
        if 'THEO' in features:
            base_features.append('theoretical')
            self.logger.debug("Added theoretical radiation (THEO) to features")
        
        # Add other AGERA5 features
        agera5_features = []
        for alias in features:
            if alias not in ['SRAD', 'THEO']:  # Skip SRAD and THEO as they're handled separately
                if alias == 'DTR':
                    agera5_features.append('agera5_DTR')
                else:
                    agera5_features.append(f'agera5_{alias}')
        
        # Get bias features if bias correction is enabled
        bias_features = []
        if self.bias_corrector:
            bias_features = self.bias_corrector.get_feature_names(df)
        
        # Combine all features
        all_features = base_features + agera5_features + bias_features
        
        # Filter to available features
        available_features = [f for f in all_features if f in df.columns]
        
        if not available_features:
            self.logger.error("No valid features found in data")
            return None, None, None
        
        # Log feature composition
        self.logger.debug(f"Feature composition:")
        self.logger.debug(f"  Base features: {[f for f in base_features if f in available_features]}")
        self.logger.debug(f"  AGERA5 features: {[f for f in agera5_features if f in available_features]}")
        self.logger.debug(f"  Bias features: {[f for f in bias_features if f in available_features]}")
        if 'theoretical' in available_features:
            self.logger.info("  ✓ THEO (theoretical radiation) included as feature")
        
        # Prepare feature matrix and target
        X = df[available_features].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(df['obs_clearness'], errors='coerce')  # Target: clearness index
        
        # Remove NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        self.feature_names = available_features
        
        return X_clean, y_clean, valid_mask
    
    def train(self, train_stations, features, data_loader):
        """Train the model on given stations"""
        self.logger.info(f"Training model with {len(train_stations)} stations")
        self.logger.info(f"Features: {features}")
        if 'THEO' in features:
            self.logger.info("  → Including THEO (theoretical radiation) as input feature")
        self.logger.info(f"Normalization: {self.normalization}")
        
        # Collect training data
        train_data_list = []
        
        for idx, station_row in train_stations.iterrows():
            df = self.collect_station_data(
                station_row, data_loader, features, 2011, 2015
            )
            
            if df is not None and len(df) > 0:
                df['station_id'] = station_row['id']
                train_data_list.append(df)
        
        if not train_data_list:
            self.logger.error("No training data collected")
            return False
        
        # Combine all training data
        train_df = pd.concat(train_data_list, ignore_index=True)
        self.logger.info(f"Combined training data: {len(train_df)} rows from {len(train_data_list)} stations")
        
        # Prepare features
        X_train, y_train, valid_mask = self.prepare_features(train_df, features)
        
        if X_train is None or len(X_train) == 0:
            self.logger.error("No valid training features")
            return False
        
        self.logger.info(f"Valid training data: {len(X_train)} rows, {X_train.shape[1]} features")
        self.logger.info(f"Target range: {y_train.min():.3f} - {y_train.max():.3f}")
        
        # Check if THEO is in features and log its statistics
        if 'theoretical' in self.feature_names:
            theo_idx = self.feature_names.index('theoretical')
            theo_values = X_train.iloc[:, theo_idx]
            self.logger.info(f"THEO statistics: min={theo_values.min():.1f}, max={theo_values.max():.1f}, mean={theo_values.mean():.1f}")
        
        # Apply normalization
        X_train_scaled = self.apply_normalization(X_train, is_training=True)
        
        # Build and train model
        self.model = self.build_ann_model(X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stopping]
        )
        
        # Evaluate training performance
        train_preds = self.model.predict(X_train_scaled).flatten()
        train_rmse = self.compute_rmse(y_train.values, train_preds)
        
        self.logger.info(f"Training RMSE (clearness): {train_rmse:.4f}")
        
        # Save model and metadata
        self.save_model()
        
        return True
    
    def validate(self, val_stations, features, data_loader, val_name="validation"):
        """Validate model on given stations"""
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info(f"Validating on {len(val_stations)} stations ({val_name})")
        if 'THEO' in features:
            self.logger.info("  → Using THEO (theoretical radiation) in validation")
        
        # Collect validation data
        val_data_list = []
        
        for idx, station_row in val_stations.iterrows():
            df = self.collect_station_data(
                station_row, data_loader, features, 2016, 2020
            )
            
            if df is not None and len(df) > 0:
                df['station_id'] = station_row['id']
                df['station_name'] = station_row['name']
                df['region'] = station_row['region']
                val_data_list.append(df)
        
        if not val_data_list:
            self.logger.warning(f"No validation data collected for {val_name}")
            return None
        
        # Combine validation data
        val_df = pd.concat(val_data_list, ignore_index=True)
        
        # Prepare features (same as training)
        X_val, y_val, valid_mask = self.prepare_features(val_df, features)
        
        if X_val is None or len(X_val) == 0:
            self.logger.warning(f"No valid validation features for {val_name}")
            return None
        
        # Apply normalization (prediction mode)
        X_val_scaled = self.apply_normalization(X_val, is_training=False)
        
        # Predict
        pred_clearness = self.model.predict(X_val_scaled).flatten()
        
        # Convert back to absolute values
        val_df_clean = val_df[valid_mask].copy()
        pred_absolute = pred_clearness * val_df_clean['theoretical']
        
        # Calculate metrics
        y_val_absolute = val_df_clean['obs'].values
        metrics = self.compute_metrics(y_val_absolute, pred_absolute)
        
        # Save predictions
        val_df_clean['ensemble_clearness'] = pred_clearness
        val_df_clean['ensemble'] = pred_absolute
        
        pred_file = self.output_dir / f"predictions_{val_name}.csv"
        val_df_clean.to_csv(pred_file, index=False)
        
        self.logger.info(f"Validation {val_name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def compute_rmse(self, y_true, y_pred):
        """Compute RMSE"""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.any():
            return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
        else:
            return np.nan
    
    def compute_metrics(self, y_true, y_pred):
        """Compute comprehensive evaluation metrics"""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        
        if not mask.any():
            return {
                'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 
                'r2': np.nan, 'mse': np.nan, 'n_samples': 0
            }
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Basic metrics
        mse = np.mean((y_true_clean - y_pred_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE with safe division
        denominator = np.abs(y_true_clean) + 1e-8
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / denominator)) * 100
        
        # R²
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
        except:
            r2 = np.nan
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'n_samples': len(y_true_clean)
        }
    
    def save_model(self):
        """Save model and associated components"""
        try:
            # Save Keras model
            model_file = self.output_dir / "model.h5"
            self.model.save(model_file)
            
            # Save feature names
            if self.feature_names:
                feature_file = self.output_dir / "feature_names.txt"
                with open(feature_file, 'w') as f:
                    for feature in self.feature_names:
                        f.write(f"{feature}\n")
            
            # Save model metadata
            metadata = {
                'normalization': self.normalization,
                'n_features': len(self.feature_names) if self.feature_names else 0,
                'feature_names': self.feature_names,
                'has_theo': 'theoretical' in self.feature_names if self.feature_names else False
            }
            
            import json
            metadata_file = self.output_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, model_dir):
        """Load saved model and components"""
        model_dir = Path(model_dir)
        
        try:
            # Load Keras model
            model_file = model_dir / "model.h5"
            if model_file.exists():
                self.model = load_model(model_file)
            else:
                self.logger.error(f"Model file not found: {model_file}")
                return False
            
            # Load normalization components
            if self.normalization == 'decimal':
                decimal_file = model_dir / "decimal_params.pkl"
                if decimal_file.exists():
                    self.decimal_params = joblib.load(decimal_file)
            else:
                scaler_file = model_dir / "scaler.pkl"
                if scaler_file.exists():
                    self.scaler = joblib.load(scaler_file)
            
            # Load feature names
            feature_file = model_dir / "feature_names.txt"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            
            # Check if THEO was used
            import json
            metadata_file = model_dir / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('has_theo', False):
                        self.logger.info("Loaded model was trained with THEO feature")
            
            self.logger.info(f"Model loaded from {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict_single_station(self, station_row, features, data_loader, start_year=2016, end_year=2020):
        """Predict for a single station (for testing)"""
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        # Collect data
        df = self.collect_station_data(station_row, data_loader, features, start_year, end_year)
        
        if df is None or len(df) == 0:
            return None
        
        # Prepare features
        X, y, valid_mask = self.prepare_features(df, features)
        
        if X is None or len(X) == 0:
            return None
        
        # Apply normalization and predict
        X_scaled = self.apply_normalization(X, is_training=False)
        pred_clearness = self.model.predict(X_scaled).flatten()
        
        # Convert to absolute values
        df_clean = df[valid_mask].copy()
        pred_absolute = pred_clearness * df_clean['theoretical']
        
        # Add predictions to dataframe
        df_clean['ensemble_clearness'] = pred_clearness
        df_clean['ensemble'] = pred_absolute
        
        return df_clean

# Wrapper class for compatibility with new structure
class LumenModel(LUMENModel):
    """
    Compatibility wrapper for LUMEN model
    Extends the original LUMENModel class
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize LumenModel with original class"""
        super().__init__(*args, **kwargs)
        self.model_type = "LUMEN"
    
    @classmethod
    def create_model(cls, *args, **kwargs):
        """Factory method to create LUMEN model"""
        return cls(*args, **kwargs)


# For backward compatibility
def create_lumen_model(*args, **kwargs):
    """Create LUMEN model instance"""
    return LumenModel(*args, **kwargs)
