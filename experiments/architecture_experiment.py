#!/usr/bin/env python3
"""
Enhanced Neural Network Architecture Experiment with Detailed Prediction Storage
Full architecture comparison experiment with detailed fold-level predictions and station analysis
Usage: python architecture_experiment_main.py <experiment_id>
Example: python architecture_experiment_main.py 198
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from core.data.loaders.base_loader import DataLoader
from core.models.lumen.model import LumenModel
from core.preprocessing.bias_correction.corrector import BiasCorrection

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    
    # Configure GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPU detected: {physical_devices}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU detected, using CPU")
        
except ImportError:
    print("TensorFlow not available - will use sklearn fallback")
    TF_AVAILABLE = False

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / "experiment.log"
    
    # Create logger
    logger = logging.getLogger("ArchitectureExperiment")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_experiment_config(experiment_id: int) -> Dict:
    """Load experiment configuration from CSV files"""
    
    plan_files = [
        'experiment_plan_full.csv',
        'experiment_plan_priority.csv',
        'experiment_plan.csv'
    ]
    
    for plan_file in plan_files:
        if Path(plan_file).exists():
            try:
                df = pd.read_csv(plan_file, encoding='utf-8')
                exp_row = df[df['id'] == experiment_id]
                
                if not exp_row.empty:
                    row = exp_row.iloc[0]
                    return {
                        'id': int(row['id']),
                        'features': str(row['features']),
                        'normalization': str(row['normalization']),
                        'bias_correction': str(row['bias_correction']),
                        'description': str(row.get('description', ''))
                    }
                    
            except Exception as e:
                print(f"Error reading {plan_file}: {e}")
                continue
    
    raise ValueError(f"Experiment ID {experiment_id} not found in any plan file")



def prepare_experiment_data(config: Dict, data_loader: DataLoader, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List, pd.DataFrame]:
    """Prepare data for architecture experiment with station metadata"""
    
    logger.info("=== DATA PREPARATION PHASE ===")
    logger.info(f"Features: {config['features']}")
    logger.info(f"Normalization: {config['normalization']}")
    logger.info(f"Bias Correction: {config['bias_correction']}")
    
    # Parse features
    features = config['features'].split()
    logger.info(f"Parsed features: {features}")
    
    # Setup bias corrector
    bias_corrector = BiasCorrection()
    if config['bias_correction'] and config['bias_correction'] != 'none':
        bias_methods = [m.strip() for m in config['bias_correction'].split(',')]
        bias_corrector.configure(bias_methods)
        logger.info(f"Configured bias correction: {bias_methods}")
    else:
        logger.info("No bias correction configured")
    
    # Load stations
    stations = data_loader.load_station_info()
    stations = data_loader.filter_stations_by_data_availability(stations, min_years=3)
    logger.info(f"Loaded {len(stations)} stations with sufficient data")
    
    # Debug: Show station info columns
    logger.info(f"Station info columns: {list(stations.columns)}")
    if len(stations) > 0:
        logger.info(f"Sample station info: {dict(stations.iloc[0])}")
    
    # Collect data from all stations for years 2011-2015 (training period)
    logger.info("Collecting training data from all stations (2011-2015)...")
    all_data = []
    
    for idx, station_row in stations.iterrows():
        try:
            # Collect station data using DataLoader
            df = data_loader.collect_station_data(
                station_row, 2011, 2015, features, 'training'
            )
            
            if df is None or len(df) == 0:
                continue
            
            # Apply bias corrections if configured
            if bias_corrector.enabled_methods:
                df = bias_corrector.apply_corrections(df, station_row)
            
            # Add station information for tracking
            df['station_id'] = station_row['id']
            df['station_name'] = station_row['name']
            df['region'] = station_row['region']
            
            # Safely get latitude/longitude with fallbacks
            if 'latitude' in station_row and pd.notna(station_row['latitude']):
                df['latitude'] = station_row['latitude']
            elif 'lat' in station_row and pd.notna(station_row['lat']):
                df['latitude'] = station_row['lat']
            else:
                df['latitude'] = 37.0 if station_row['region'] == 'Korea' else 35.0
                
            if 'longitude' in station_row and pd.notna(station_row['longitude']):
                df['longitude'] = station_row['longitude']
            elif 'lon' in station_row and pd.notna(station_row['lon']):
                df['longitude'] = station_row['lon']
            else:
                df['longitude'] = 127.0 if station_row['region'] == 'Korea' else 135.0
            
            # Add to collection
            all_data.append(df)
            
            if len(all_data) % 10 == 0:
                logger.info(f"Collected data from {len(all_data)} stations...")
                
        except Exception as e:
            logger.warning(f"Error collecting data from station {station_row['id']}: {e}")
            continue
    
    if not all_data:
        logger.error("No training data collected from any station")
        logger.error("Cannot proceed without valid training data")
        logger.error("Experiment terminated due to insufficient data")
        sys.exit(1)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_data.shape}")
    
    # Remove any rows with missing values
    combined_data = combined_data.dropna()
    logger.info(f"After removing NaN: {combined_data.shape}")
    
    # Check for target column
    target_column = None
    possible_target_cols = ['obs_clearness', 'obs', 'srad', 'SRAD', 'solar_radiation']
    
    for col in possible_target_cols:
        if col in combined_data.columns:
            target_column = col
            logger.info(f"Found target column: {target_column}")
            break
    
    if target_column is None:
        logger.error("No target column found!")
        raise ValueError("Target column not found in combined data")
    
    # Prepare feature matrix X
    feature_columns = []
    for feature in features:
        if feature == 'THEO':
            if 'theoretical' in combined_data.columns:
                feature_columns.append('theoretical')
            else:
                logger.warning(f"THEO feature requested but 'theoretical' column not found")
        else:
            agera5_col = f"agera5_{feature}"
            if agera5_col in combined_data.columns:
                feature_columns.append(agera5_col)
            else:
                logger.warning(f"Feature {feature} not found (expected column: {agera5_col})")
    
    if not feature_columns:
        raise ValueError("No valid feature columns found in data")
    
    logger.info(f"Using feature columns: {feature_columns}")
    
    # Extract features and target
    X = combined_data[feature_columns].values
    y = combined_data[target_column].values
    
    logger.info(f"Final data shapes - X: {X.shape}, y: {y.shape}")
    logger.info(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    logger.info(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    
    return X, y, feature_columns, combined_data


class ArchitectureExperiment:
    """Enhanced neural network architecture comparison with detailed predictions"""
    
    def __init__(self, config: Dict, output_dir: str, logger: logging.Logger):
        """Initialize experiment"""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        
        # Create predictions directory
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Define experiment configurations
        self.architectures = self._define_architectures()
        self.regularization_configs = self._define_regularization()
        self.optimizers = self._define_optimizers()
        
        self.results = []
        self.experiment_count = 0
        self.total_experiments = len(self.architectures) * len(self.regularization_configs) * len(self.optimizers)
        
        # Storage for all predictions
        self.all_predictions = []
        
        self.logger.info("=== ENHANCED ARCHITECTURE EXPERIMENT INITIALIZED ===")
        self.logger.info(f"Total combinations: {self.total_experiments}")
        self.logger.info(f"Using TensorFlow: {TF_AVAILABLE}")
        self.logger.info(f"Predictions directory: {self.predictions_dir}")
    
    def _define_architectures(self) -> List[Dict]:
        """Define neural network architectures"""
        return [
            # Single hidden layer
            {'name': 'Single_32', 'layers': [32], 'description': 'Single layer: 32 neurons'},
            {'name': 'Single_64', 'layers': [64], 'description': 'Single layer: 64 neurons'},
            {'name': 'Single_128', 'layers': [128], 'description': 'Single layer: 128 neurons'},
            
            # Double hidden layers
            {'name': 'Double_32_16', 'layers': [32, 16], 'description': 'Two layers: 32->16'},
            {'name': 'Double_64_32', 'layers': [64, 32], 'description': 'Two layers: 64->32'},
            {'name': 'Double_128_64', 'layers': [128, 64], 'description': 'Two layers: 128->64'},
            {'name': 'Double_64_64', 'layers': [64, 64], 'description': 'Two equal layers: 64->64'},
            
            # Triple hidden layers
            {'name': 'Triple_128_64_32', 'layers': [128, 64, 32], 'description': 'Three layers: 128->64->32'},
            {'name': 'Triple_64_32_16', 'layers': [64, 32, 16], 'description': 'Three layers: 64->32->16'},
            
            # Deep networks
            {'name': 'Deep_256_128_64_32', 'layers': [256, 128, 64, 32], 'description': 'Deep: 256->128->64->32'},
        ]
    
    def _define_regularization(self) -> List[Dict]:
        """Define regularization configurations"""
        return [
            {'name': 'None', 'dropout': 0.0, 'batch_norm': False, 'l1': 0.0, 'l2': 0.0},
            {'name': 'L2_Weak', 'dropout': 0.0, 'batch_norm': False, 'l1': 0.0, 'l2': 0.001},
            {'name': 'L2_Strong', 'dropout': 0.0, 'batch_norm': False, 'l1': 0.0, 'l2': 0.01},
            {'name': 'Dropout_Light', 'dropout': 0.1, 'batch_norm': False, 'l1': 0.0, 'l2': 0.0},
            {'name': 'Dropout_Medium', 'dropout': 0.2, 'batch_norm': False, 'l1': 0.0, 'l2': 0.0},
            {'name': 'BatchNorm', 'dropout': 0.0, 'batch_norm': True, 'l1': 0.0, 'l2': 0.0},
            {'name': 'L2_Dropout', 'dropout': 0.2, 'batch_norm': False, 'l1': 0.0, 'l2': 0.001},
        ]
    
    def _define_optimizers(self) -> List[Dict]:
        """Define optimizer configurations"""
        return [
            {'name': 'Adam_Default', 'type': 'adam', 'lr': 0.001},
            {'name': 'Adam_Fast', 'type': 'adam', 'lr': 0.01},
            {'name': 'Adam_Slow', 'type': 'adam', 'lr': 0.0001},
            {'name': 'RMSprop_Default', 'type': 'rmsprop', 'lr': 0.001},
        ]
    
    def create_tensorflow_model(self, arch: Dict, reg: Dict, opt: Dict, input_dim: int):
        """Create TensorFlow model"""
        try:
            model = Sequential()
            
            # Setup regularization
            if reg['l1'] > 0 and reg['l2'] > 0:
                regularizer = l1_l2(l1=reg['l1'], l2=reg['l2'])
            elif reg['l1'] > 0:
                regularizer = l1(reg['l1'])
            elif reg['l2'] > 0:
                regularizer = l2(reg['l2'])
            else:
                regularizer = None
            
            # Input layer
            model.add(Dense(
                arch['layers'][0], 
                activation='relu', 
                input_shape=(input_dim,),
                kernel_regularizer=regularizer
            ))
            
            if reg['batch_norm']:
                model.add(BatchNormalization())
            
            if reg['dropout'] > 0:
                model.add(Dropout(reg['dropout']))
            
            # Hidden layers
            for units in arch['layers'][1:]:
                model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
                
                if reg['batch_norm']:
                    model.add(BatchNormalization())
                
                if reg['dropout'] > 0:
                    model.add(Dropout(reg['dropout']))
            
            # Output layer
            model.add(Dense(1, activation='linear'))
            
            # Compile model
            if opt['type'] == 'rmsprop':
                optimizer = RMSprop(learning_rate=opt['lr'])
            else:
                optimizer = Adam(learning_rate=opt['lr'])
            
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating TensorFlow model: {e}")
            return None
    
    def create_sklearn_model(self, arch: Dict, reg: Dict, opt: Dict) -> MLPRegressor:
        """Create sklearn model as fallback"""
        
        alpha = max(reg['l1'], reg['l2']) if reg['l1'] > 0 or reg['l2'] > 0 else 0.0001
        learning_rate_init = opt['lr']
        
        model = MLPRegressor(
            hidden_layer_sizes=tuple(arch['layers']),
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        
        return model
    
    def calculate_station_metrics(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed station-wise metrics including NRMSE, RMSE, and bias"""
        
        station_metrics = []
        
        for station_id, group in predictions_df.groupby('station_id'):
            if len(group) == 0:
                continue
                
            actual = group['actual'].values
            predicted = group['predicted'].values
            
            # Basic metrics
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            bias = np.mean(predicted - actual)  # Positive = overprediction, Negative = underprediction
            
            # NRMSE (normalized by mean of observed values)
            mean_actual = np.mean(actual)
            nrmse = rmse / mean_actual if mean_actual > 0 else np.nan
            
            # R-squared
            r2 = r2_score(actual, predicted)
            
            # Additional statistics
            std_actual = np.std(actual)
            std_predicted = np.std(predicted)
            
            station_info = group.iloc[0]
            
            # Safely get station location
            latitude = station_info.get('latitude', 37.0)
            longitude = station_info.get('longitude', 127.0)
            
            station_metrics.append({
                'station_id': station_id,
                'station_name': station_info['station_name'],
                'region': station_info['region'],
                'latitude': latitude,
                'longitude': longitude,
                'n_samples': len(group),
                'rmse': rmse,
                'nrmse': nrmse,
                'mae': mae,
                'bias': bias,
                'r2': r2,
                'mean_actual': mean_actual,
                'mean_predicted': np.mean(predicted),
                'std_actual': std_actual,
                'std_predicted': std_predicted,
                'min_actual': np.min(actual),
                'max_actual': np.max(actual),
                'min_predicted': np.min(predicted),
                'max_predicted': np.max(predicted)
            })
        
        return pd.DataFrame(station_metrics)
    
    def evaluate_single_architecture(self, arch: Dict, reg: Dict, opt: Dict, X: np.ndarray, y: np.ndarray, data_df: pd.DataFrame) -> Optional[Dict]:
        """Evaluate a single architecture configuration with detailed prediction storage"""
        
        config_name = f"{arch['name']}_{reg['name']}_{opt['name']}"
        self.experiment_count += 1
        
        # Create experiment directory
        experiment_dir = self.predictions_dir / f"exp_{self.experiment_count:03d}_{config_name}"
        experiment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"\n[{self.experiment_count}/{self.total_experiments}] Testing: {config_name}")
        
        start_time = time.time()
        
        try:
            # Setup cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            fold_times = []
            fold_predictions = []
            epochs_trained = 0
            
            if TF_AVAILABLE:
                # TensorFlow implementation with prediction storage
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                    fold_start = time.time()
                    
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Get corresponding data for this fold
                    val_data = data_df.iloc[val_idx].copy()
                    
                    # Create model
                    model = self.create_tensorflow_model(arch, reg, opt, X.shape[1])
                    if model is None:
                        raise Exception("Failed to create TensorFlow model")
                    
                    # Train model
                    early_stopping = EarlyStopping(
                        monitor='val_loss', 
                        patience=20, 
                        restore_best_weights=True, 
                        verbose=0
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=200,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    epochs_trained = len(history.history['loss'])
                    fold_time = time.time() - fold_start
                    fold_times.append(fold_time)
                    
                    # Generate predictions
                    y_pred = model.predict(X_val, verbose=0).flatten()
                    mse = np.mean((y_val - y_pred) ** 2)
                    cv_scores.append(mse)
                    
                    # Store fold predictions with metadata
                    fold_pred_data = val_data.copy()
                    fold_pred_data['fold'] = fold
                    fold_pred_data['actual'] = y_val
                    fold_pred_data['predicted'] = y_pred
                    fold_pred_data['residual'] = y_val - y_pred
                    fold_pred_data['abs_error'] = np.abs(y_val - y_pred)
                    fold_pred_data['squared_error'] = (y_val - y_pred) ** 2
                    fold_pred_data['experiment_name'] = config_name
                    fold_pred_data['experiment_num'] = self.experiment_count
                    
                    fold_predictions.append(fold_pred_data)
                    
                    # Save individual fold predictions
                    fold_file = experiment_dir / f"fold_{fold}_predictions.csv"
                    fold_pred_data.to_csv(fold_file, index=False)
                
                cv_rmse = np.sqrt(cv_scores)
                avg_time = np.mean(fold_times)
                
            else:
                # Sklearn implementation with prediction storage
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                    fold_start = time.time()
                    
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Get corresponding data for this fold
                    val_data = data_df.iloc[val_idx].copy()
                    
                    # Create and train model
                    model = self.create_sklearn_model(arch, reg, opt)
                    model.fit(X_train, y_train)
                    
                    fold_time = time.time() - fold_start
                    fold_times.append(fold_time)
                    
                    # Generate predictions
                    y_pred = model.predict(X_val)
                    mse = np.mean((y_val - y_pred) ** 2)
                    cv_scores.append(mse)
                    
                    # Store fold predictions with metadata
                    fold_pred_data = val_data.copy()
                    fold_pred_data['fold'] = fold
                    fold_pred_data['actual'] = y_val
                    fold_pred_data['predicted'] = y_pred
                    fold_pred_data['residual'] = y_val - y_pred
                    fold_pred_data['abs_error'] = np.abs(y_val - y_pred)
                    fold_pred_data['squared_error'] = (y_val - y_pred) ** 2
                    fold_pred_data['experiment_name'] = config_name
                    fold_pred_data['experiment_num'] = self.experiment_count
                    
                    fold_predictions.append(fold_pred_data)
                    
                    # Save individual fold predictions
                    fold_file = experiment_dir / f"fold_{fold}_predictions.csv"
                    fold_pred_data.to_csv(fold_file, index=False)
                
                cv_rmse = np.sqrt(cv_scores)
                avg_time = np.mean(fold_times)
                epochs_trained = getattr(model, 'n_iter_', 100)
            
            # Combine all fold predictions
            all_fold_predictions = pd.concat(fold_predictions, ignore_index=True)
            all_fold_predictions.to_csv(experiment_dir / "all_fold_predictions.csv", index=False)
            
            # Calculate station-wise metrics
            station_metrics = self.calculate_station_metrics(all_fold_predictions)
            station_metrics.to_csv(experiment_dir / "station_performance.csv", index=False)
            
            # Calculate fold summary
            fold_summary = all_fold_predictions.groupby('fold').agg({
                'actual': 'count',
                'abs_error': 'mean',
                'squared_error': lambda x: np.sqrt(x.mean()),
                'residual': 'mean'
            }).round(4)
            fold_summary.columns = ['n_samples', 'mae', 'rmse', 'bias']
            fold_summary.to_csv(experiment_dir / "fold_summary.csv")
            
            # Final model training on all data
            if TF_AVAILABLE:
                final_model = self.create_tensorflow_model(arch, reg, opt, X.shape[1])
                final_model.fit(X, y, epochs=epochs_trained, batch_size=32, verbose=0)
                y_pred_train = final_model.predict(X, verbose=0).flatten()
                total_params = final_model.count_params()
            else:
                final_model = self.create_sklearn_model(arch, reg, opt)
                final_model.fit(X, y)
                y_pred_train = final_model.predict(X)
                # Estimate parameters
                layer_sizes = [X.shape[1]] + arch['layers'] + [1]
                total_params = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] 
                                 for i in range(len(layer_sizes)-1))
            
            train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
            train_r2 = r2_score(y, y_pred_train)
            train_mae = mean_absolute_error(y, y_pred_train)
            
            total_time = time.time() - start_time
            
            # Create comprehensive result
            result = {
                'experiment_num': self.experiment_count,
                'config_name': config_name,
                'architecture_name': arch['name'],
                'architecture_layers': str(arch['layers']),
                'regularization_name': reg['name'],
                'optimizer_name': opt['name'],
                'framework': 'TensorFlow' if TF_AVAILABLE else 'Scikit-learn',
                
                # Performance metrics
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'train_mae': train_mae,
                'overfitting_ratio': np.mean(cv_rmse) / train_rmse,
                
                # Architecture info
                'total_params': total_params,
                'hidden_layers': len(arch['layers']),
                'max_neurons': max(arch['layers']),
                
                # Training info
                'training_time': total_time,
                'avg_fold_time': avg_time,
                'epochs_trained': epochs_trained,
                
                # Configuration details
                'dropout': reg['dropout'],
                'batch_norm': reg['batch_norm'],
                'l1_reg': reg['l1'],
                'l2_reg': reg['l2'],
                'learning_rate': opt['lr'],
                'optimizer_type': opt['type'],
                
                # Prediction statistics
                'n_samples': len(y),
                'n_folds': len(cv_scores),
                'n_stations': len(station_metrics),
                'mean_station_rmse': station_metrics['rmse'].mean(),
                'mean_station_nrmse': station_metrics['nrmse'].mean(),
                'mean_station_bias': station_metrics['bias'].mean(),
                'predictions_saved': len(all_fold_predictions)
            }
            
            # Add to master predictions list
            self.all_predictions.append(all_fold_predictions)
            
            self.logger.info(f"  CV RMSE: {result['cv_rmse_mean']:.4f} ± {result['cv_rmse_std']:.4f}")
            self.logger.info(f"  Train R²: {result['train_r2']:.4f}")
            self.logger.info(f"  Stations: {result['n_stations']}")
            self.logger.info(f"  Predictions: {result['predictions_saved']} saved")
            
            return result
            
        except Exception as e:
            self.logger.error(f"  FAILED: {str(e)}")
            
            # Try once more
            self.logger.info("  Retrying once...")
            try:
                time.sleep(1)
                return self.evaluate_single_architecture(arch, reg, opt, X, y, data_df)
            except Exception as e2:
                self.logger.error(f"  RETRY FAILED: {str(e2)}")
                
                # Create failed marker
                failed_file = experiment_dir / "failed.txt"
                with open(failed_file, 'w') as f:
                    f.write(f"Experiment failed: {config_name}\n")
                    f.write(f"Primary error: {str(e)}\n")
                    f.write(f"Retry error: {str(e2)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                return None
    
    def run_full_experiment(self, X: np.ndarray, y: np.ndarray, data_df: pd.DataFrame) -> List[Dict]:
        """Run complete architecture comparison experiment with detailed predictions"""
        
        self.logger.info("=== STARTING ENHANCED ARCHITECTURE EXPERIMENT ===")
        self.logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Data includes station metadata: {len(data_df)} rows")
        
        successful_results = []
        failed_count = 0
        
        # Save intermediate results every 25 experiments
        save_interval = 25
        
        for arch in self.architectures:
            for reg in self.regularization_configs:
                for opt in self.optimizers:
                    
                    result = self.evaluate_single_architecture(arch, reg, opt, X, y, data_df)
                    
                    if result is not None:
                        successful_results.append(result)
                        self.results.append(result)
                    else:
                        failed_count += 1
                    
                    # Save intermediate results periodically
                    if self.experiment_count % save_interval == 0:
                        self.save_intermediate_results()
                        self._save_master_predictions()
                        self.logger.info(f"=== CHECKPOINT: {self.experiment_count}/{self.total_experiments} completed ===")
                        self.logger.info(f"Successful: {len(successful_results)}, Failed: {failed_count}")
        
        self.logger.info("=== EXPERIMENT COMPLETED ===")
        self.logger.info(f"Total experiments: {self.total_experiments}")
        self.logger.info(f"Successful: {len(successful_results)}")
        self.logger.info(f"Failed: {failed_count}")
        
        # Save final results
        self.save_final_results()
        self._save_master_predictions()
        self._create_summary_analysis()
        
        return successful_results
    
    def _save_master_predictions(self):
        """Save master predictions file containing all experiments"""
        if self.all_predictions:
            master_predictions = pd.concat(self.all_predictions, ignore_index=True)
            master_file = self.output_dir / "master_predictions.csv"
            master_predictions.to_csv(master_file, index=False)
            self.logger.info(f"Master predictions saved: {len(master_predictions)} total predictions")
    
    def _create_summary_analysis(self):
        """Create comprehensive summary analysis"""
        if not self.all_predictions:
            return
            
        master_predictions = pd.concat(self.all_predictions, ignore_index=True)
        
        # Overall station performance across all experiments
        overall_station_performance = master_predictions.groupby(['station_id', 'station_name', 'region']).agg({
            'actual': 'count',
            'abs_error': 'mean',
            'squared_error': lambda x: np.sqrt(x.mean()),
            'residual': 'mean'
        }).round(4)
        overall_station_performance.columns = ['total_samples', 'mean_mae', 'mean_rmse', 'mean_bias']
        
        # Calculate NRMSE for each station
        station_nrmse = []
        for station_id, group in master_predictions.groupby('station_id'):
            actual = group['actual'].values
            predicted = group['predicted'].values
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mean_actual = np.mean(actual)
            nrmse = rmse / mean_actual if mean_actual > 0 else np.nan
            station_nrmse.append({'station_id': station_id, 'overall_nrmse': nrmse})
        
        station_nrmse_df = pd.DataFrame(station_nrmse)
        overall_station_performance = overall_station_performance.reset_index().merge(
            station_nrmse_df, on='station_id', how='left'
        )
        
        overall_station_performance.to_csv(self.output_dir / "overall_station_performance.csv", index=False)
        
        # Best experiment by station
        best_by_station = master_predictions.groupby(['station_id', 'experiment_name']).agg({
            'squared_error': lambda x: np.sqrt(x.mean())
        }).reset_index()
        
        best_by_station = best_by_station.loc[
            best_by_station.groupby('station_id')['squared_error'].idxmin()
        ]
        best_by_station.columns = ['station_id', 'best_experiment', 'best_rmse']
        best_by_station.to_csv(self.output_dir / "best_experiment_by_station.csv", index=False)
        
        self.logger.info("Summary analysis saved:")
        self.logger.info(f"  - overall_station_performance.csv")
        self.logger.info(f"  - best_experiment_by_station.csv")
    
    def save_intermediate_results(self):
        """Save intermediate results"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_dir / "intermediate_results.csv", index=False)
            self.logger.info(f"Saved intermediate results: {len(self.results)} experiments")
    
    def save_final_results(self):
        """Save comprehensive final results"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        # Save main results
        df.to_csv(self.output_dir / "architecture_comparison_results.csv", index=False)
        
        # Generate summary report
        self.generate_summary_report(df)
        
        self.logger.info(f"Final results saved to {self.output_dir}")
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate comprehensive summary report"""
        
        report_path = self.output_dir / "experiment_summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED NEURAL NETWORK ARCHITECTURE COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Experiment configuration
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Experiment ID: {self.config['id']}\n")
            f.write(f"Features: {self.config['features']}\n")
            f.write(f"Normalization: {self.config['normalization']}\n")
            f.write(f"Bias Correction: {self.config['bias_correction']}\n")
            f.write(f"Framework: {'TensorFlow' if TF_AVAILABLE else 'Scikit-learn'}\n")
            f.write(f"Total Experiments: {len(df)}\n\n")
            
            # Overall statistics
            f.write("OVERALL RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total configurations tested: {len(df)}\n")
            f.write(f"Best CV RMSE: {df['cv_rmse_mean'].min():.4f}\n")
            f.write(f"Worst CV RMSE: {df['cv_rmse_mean'].max():.4f}\n")
            f.write(f"Mean CV RMSE: {df['cv_rmse_mean'].mean():.4f} ± {df['cv_rmse_mean'].std():.4f}\n")
            f.write(f"Best Train R²: {df['train_r2'].max():.4f}\n")
            f.write(f"Total training time: {df['training_time'].sum():.1f} seconds\n")
            f.write(f"Total predictions generated: {df['predictions_saved'].sum()}\n\n")
            
            # Top 10 models
            f.write("TOP 10 MODELS (by CV RMSE):\n")
            f.write("-" * 30 + "\n")
            top_10 = df.nsmallest(10, 'cv_rmse_mean')
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. {row['config_name']}\n")
                f.write(f"    CV RMSE: {row['cv_rmse_mean']:.4f} ± {row['cv_rmse_std']:.4f}\n")
                f.write(f"    Train R²: {row['train_r2']:.4f}\n")
                f.write(f"    Station RMSE: {row['mean_station_rmse']:.4f}\n")
                f.write(f"    Station NRMSE: {row['mean_station_nrmse']:.4f}\n")
                f.write(f"    Station Bias: {row['mean_station_bias']:.4f}\n")
                f.write(f"    Architecture: {row['architecture_layers']}\n")
                f.write(f"    Predictions: exp_{row['experiment_num']:03d}_*/\n\n")
            
            f.write("PREDICTION FILES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Individual experiments: {self.predictions_dir}/exp_XXX_*/\n")
            f.write(f"  - all_fold_predictions.csv: All CV predictions\n")
            f.write(f"  - station_performance.csv: Station-wise metrics\n")
            f.write(f"  - fold_summary.csv: Fold-wise summary\n")
            f.write(f"Master predictions: {self.output_dir}/master_predictions.csv\n")
            f.write(f"Overall station performance: {self.output_dir}/overall_station_performance.csv\n")
            f.write(f"Best experiment by station: {self.output_dir}/best_experiment_by_station.csv\n\n")
            
            f.write("STATION METRICS INCLUDED:\n")
            f.write("-" * 30 + "\n")
            f.write("- RMSE: Root Mean Square Error\n")
            f.write("- NRMSE: Normalized RMSE (RMSE / mean(observed))\n")
            f.write("- Bias: Mean(predicted - actual)\n")
            f.write("  * Positive bias = overprediction\n")
            f.write("  * Negative bias = underprediction\n")
            f.write("- MAE: Mean Absolute Error\n")
            f.write("- R²: Coefficient of determination\n")


def main():
    """Main execution function"""
    
    if len(sys.argv) != 2:
        print("Usage: python architecture_experiment_main.py <experiment_id>")
        print("Example: python architecture_experiment_main.py 198")
        sys.exit(1)
    
    try:
        experiment_id = int(sys.argv[1])
    except ValueError:
        print("Error: experiment_id must be an integer")
        sys.exit(1)
    
    print("=" * 80)
    print("ENHANCED NEURAL NETWORK ARCHITECTURE COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print(f"TensorFlow Available: {TF_AVAILABLE}")
    print("Enhanced features: Detailed predictions, station analysis, bias tracking")
    print("")
    
    try:
        # Load configuration
        config = load_experiment_config(experiment_id)
        output_dir = f"architecture_results_exp_{experiment_id}_enhanced"
        
        # Setup logging
        logger = setup_logging(output_dir)
        
        logger.info("=== ENHANCED ARCHITECTURE EXPERIMENT STARTED ===")
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Configuration: {config}")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize data loader
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader()
        
        # Validate data paths
        validation = data_loader.validate_data_paths()
        missing_data = [name for name, exists in validation.items() if not exists]
        if missing_data:
            logger.warning(f"Missing data files: {missing_data}")
            logger.info("Will attempt to use available data or generate mock data if needed")
        
        # Prepare experiment data (will fallback to mock data if needed)
        X, y, feature_columns, data_df = prepare_experiment_data(config, data_loader, logger)
        
        # Initialize and run experiment
        experiment = ArchitectureExperiment(config, output_dir, logger)
        
        # Run comprehensive experiment
        results = experiment.run_full_experiment(X, y, data_df)
        
        if results:
            logger.info("=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
            
            # Show top 3 results
            df = pd.DataFrame(results)
            top_3 = df.nsmallest(3, 'cv_rmse_mean')
            
            print("\n" + "=" * 80)
            print("ENHANCED EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nTOP 3 CONFIGURATIONS:")
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"{i}. {row['config_name']}")
                print(f"   CV RMSE: {row['cv_rmse_mean']:.4f} ± {row['cv_rmse_std']:.4f}")
                print(f"   Train R²: {row['train_r2']:.4f}")
                print(f"   Station NRMSE: {row['mean_station_nrmse']:.4f}")
                print(f"   Station Bias: {row['mean_station_bias']:.4f}")
                print(f"   Predictions: {row['predictions_saved']}")
                print()
            
            print(f"Results directory: {output_dir}")
            print("Key output files:")
            print("- architecture_comparison_results.csv (main results)")
            print("- master_predictions.csv (all predictions)")
            print("- predictions/ (individual experiment predictions)")
            print("- overall_station_performance.csv (station analysis)")
            print("- experiment_summary_report.txt (detailed report)")
            print("- experiment.log (detailed log)")
            
        else:
            logger.error("ERROR: No successful experiments completed")
            sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()