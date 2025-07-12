#!/usr/bin/env python3
"""
Modular Neural Network Architecture Builder for ANNrun_code
Supports TensorFlow and PyTorch backends with comprehensive architecture options
Fixed version with proper imports and error handling
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available")

# PyTorch imports with fallback - Fixed import order and error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Fallback for type hints
    logging.warning("PyTorch not available")

# Scikit-learn fallback
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")


@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture"""
    name: str
    layers: List[int]
    activation: str = 'relu'
    output_activation: Optional[str] = None
    use_bias: bool = True
    kernel_initializer: str = 'glorot_uniform'
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'layers': self.layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer
        }


@dataclass 
class RegularizationConfig:
    """Configuration for regularization"""
    name: str
    dropout_rate: Optional[float] = None
    l1_lambda: Optional[float] = None
    l2_lambda: Optional[float] = None
    batch_norm: bool = False
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'dropout_rate': self.dropout_rate,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'batch_norm': self.batch_norm,
            'early_stopping_patience': self.early_stopping_patience
        }


@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""
    name: str
    optimizer_type: str
    learning_rate: float
    beta_1: Optional[float] = None
    beta_2: Optional[float] = None
    epsilon: Optional[float] = None
    momentum: Optional[float] = None
    decay: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'decay': self.decay
        }


class BaseModelBuilder(ABC):
    """Abstract base class for model builders"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def build_model(self, architecture: ArchitectureConfig, 
                   regularization: RegularizationConfig,
                   optimizer: OptimizerConfig,
                   input_dim: int) -> Tuple[Any, Dict]:
        """Build and compile model"""
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> Dict:
        """Get model information"""
        pass


class TensorFlowModelBuilder(BaseModelBuilder):
    """TensorFlow/Keras model builder"""
    
    def __init__(self):
        super().__init__()
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        # Configure GPU if available
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"Configured {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    self.logger.warning(f"GPU configuration error: {e}")
            else:
                self.logger.info("No GPU detected, using CPU")
        except Exception as e:
            self.logger.warning(f"Error checking GPU configuration: {e}")
    
    def build_model(self, architecture: ArchitectureConfig,
                   regularization: RegularizationConfig, 
                   optimizer: OptimizerConfig,
                   input_dim: int) -> Tuple[tf.keras.Model, Dict]:
        """Build TensorFlow model"""
        
        self.logger.debug(f"Building TensorFlow model: {architecture.name}")
        
        # Create model
        model = self._create_tf_model(architecture, regularization, input_dim)
        
        # Create optimizer
        optimizer_instance = self._create_tf_optimizer(optimizer)
        
        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss='mse',
            metrics=['mae']
        )
        
        # Model info
        model_info = {
            'framework': 'tensorflow',
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'architecture': architecture.to_dict(),
            'regularization': regularization.to_dict(),
            'optimizer': optimizer.to_dict()
        }
        
        return model, model_info
    
    def _create_tf_model(self, architecture: ArchitectureConfig,
                        regularization: RegularizationConfig,
                        input_dim: int) -> tf.keras.Model:
        """Create TensorFlow model"""
        
        # Input layer
        inputs = Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(architecture.layers):
            # Dense layer
            x = Dense(
                units,
                activation=None,
                use_bias=architecture.use_bias,
                kernel_initializer=architecture.kernel_initializer,
                kernel_regularizer=self._get_tf_regularizer(regularization),
                name=f'dense_{i+1}'
            )(x)
            
            # Batch normalization
            if regularization.batch_norm:
                x = BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            # Activation
            if architecture.activation == 'relu':
                x = tf.keras.layers.ReLU(name=f'relu_{i+1}')(x)
            elif architecture.activation == 'tanh':
                x = tf.keras.activations.tanh(x)
            elif architecture.activation == 'sigmoid':
                x = tf.keras.activations.sigmoid(x)
            
            # Dropout
            if regularization.dropout_rate and regularization.dropout_rate > 0:
                x = Dropout(regularization.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(
            1,
            activation=architecture.output_activation,
            use_bias=architecture.use_bias,
            name='output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=architecture.name)
        return model
    
    def _get_tf_regularizer(self, regularization: RegularizationConfig):
        """Get TensorFlow regularizer"""
        if regularization.l1_lambda and regularization.l2_lambda:
            return l1_l2(l1=regularization.l1_lambda, l2=regularization.l2_lambda)
        elif regularization.l1_lambda:
            return l1(regularization.l1_lambda)
        elif regularization.l2_lambda:
            return l2(regularization.l2_lambda)
        return None
    
    def _create_tf_optimizer(self, opt_config: OptimizerConfig):
        """Create TensorFlow optimizer instance"""
        if opt_config.optimizer_type.lower() == 'adam':
            return Adam(
                learning_rate=opt_config.learning_rate,
                beta_1=opt_config.beta_1 or 0.9,
                beta_2=opt_config.beta_2 or 0.999,
                epsilon=opt_config.epsilon or 1e-7
            )
        elif opt_config.optimizer_type.lower() == 'rmsprop':
            return RMSprop(
                learning_rate=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0,
                epsilon=opt_config.epsilon or 1e-7
            )
        elif opt_config.optimizer_type.lower() == 'sgd':
            return SGD(
                learning_rate=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer_type}")
    
    def get_model_info(self, model: tf.keras.Model) -> Dict:
        """Get TensorFlow model information"""
        return {
            'framework': 'tensorflow',
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }


class PyTorchModelBuilder(BaseModelBuilder):
    """PyTorch model builder - Fixed version with proper imports"""
    
    def __init__(self):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def build_model(self, architecture: ArchitectureConfig,
                   regularization: RegularizationConfig,
                   optimizer: OptimizerConfig,
                   input_dim: int) -> Tuple[nn.Module, Dict]:
        """Build PyTorch model"""
        
        self.logger.debug(f"Building PyTorch model: {architecture.name}")
        
        # Create model
        model = self._create_pytorch_model(architecture, regularization, input_dim)
        model = model.to(self.device)
        
        # Create optimizer
        optimizer_instance = self._create_pytorch_optimizer(model, optimizer)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'framework': 'pytorch',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device),
            'architecture': architecture.to_dict(),
            'regularization': regularization.to_dict(),
            'optimizer': optimizer.to_dict(),
            'optimizer_instance': optimizer_instance
        }
        
        return model, model_info
    
    def _create_pytorch_model(self, architecture: ArchitectureConfig,
                             regularization: RegularizationConfig,
                             input_dim: int) -> nn.Module:
        """Create PyTorch model"""
        
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                
                layers = []
                in_features = input_dim
                
                # Hidden layers
                for i, units in enumerate(architecture.layers):
                    layers.append(nn.Linear(in_features, units, bias=architecture.use_bias))
                    
                    if regularization.batch_norm:
                        layers.append(nn.BatchNorm1d(units))
                    
                    # Activation
                    if architecture.activation.lower() == 'relu':
                        layers.append(nn.ReLU())
                    elif architecture.activation.lower() == 'tanh':
                        layers.append(nn.Tanh())
                    elif architecture.activation.lower() == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    elif architecture.activation.lower() == 'leaky_relu':
                        layers.append(nn.LeakyReLU())
                    
                    if regularization.dropout_rate and regularization.dropout_rate > 0:
                        layers.append(nn.Dropout(regularization.dropout_rate))
                    
                    in_features = units
                
                # Output layer
                layers.append(nn.Linear(in_features, 1, bias=architecture.use_bias))
                
                # Apply output activation if specified
                if architecture.output_activation:
                    if architecture.output_activation.lower() == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    elif architecture.output_activation.lower() == 'tanh':
                        layers.append(nn.Tanh())
                
                self.network = nn.Sequential(*layers)
                
                # Initialize weights
                self._init_weights(architecture.kernel_initializer)
            
            def _init_weights(self, initializer: str):
                """Initialize weights"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        if initializer == 'glorot_uniform' or initializer == 'xavier_uniform':
                            nn.init.xavier_uniform_(module.weight)
                        elif initializer == 'glorot_normal' or initializer == 'xavier_normal':
                            nn.init.xavier_normal_(module.weight)
                        elif initializer == 'he_uniform':
                            nn.init.kaiming_uniform_(module.weight)
                        elif initializer == 'he_normal':
                            nn.init.kaiming_normal_(module.weight)
                        
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            
            def forward(self, x):
                return self.network(x)
        
        return NeuralNetwork()
    
    def _create_pytorch_optimizer(self, model: nn.Module, opt_config: OptimizerConfig):
        """Create PyTorch optimizer"""
        # Add L1/L2 regularization to parameters if specified
        params = model.parameters()
        
        if opt_config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                params,
                lr=opt_config.learning_rate,
                betas=(opt_config.beta_1 or 0.9, opt_config.beta_2 or 0.999),
                eps=opt_config.epsilon or 1e-8,
                weight_decay=opt_config.decay or 0.0
            )
        elif opt_config.optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0,
                eps=opt_config.epsilon or 1e-8,
                weight_decay=opt_config.decay or 0.0
            )
        elif opt_config.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0,
                weight_decay=opt_config.decay or 0.0
            )
        elif opt_config.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=(opt_config.beta_1 or 0.9, opt_config.beta_2 or 0.999),
                eps=opt_config.epsilon or 1e-8,
                weight_decay=opt_config.decay or 0.01
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer_type}")
    
    def get_model_info(self, model: nn.Module) -> Dict:
        """Get PyTorch model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'framework': 'pytorch',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': len(list(model.named_modules())) - 1,  # Exclude root module
            'device': next(model.parameters()).device if list(model.parameters()) else 'cpu'
        }


class SklearnModelBuilder(BaseModelBuilder):
    """Scikit-learn model builder (fallback)"""
    
    def __init__(self):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
    
    def build_model(self, architecture: ArchitectureConfig,
                   regularization: RegularizationConfig,
                   optimizer: OptimizerConfig, 
                   input_dim: int) -> Tuple[MLPRegressor, Dict]:
        """Build scikit-learn model"""
        
        self.logger.debug(f"Building sklearn model: {architecture.name}")
        
        # Convert architecture to sklearn format
        hidden_layer_sizes = tuple(architecture.layers) if len(architecture.layers) > 1 else (architecture.layers[0],)
        
        # Map activation functions
        activation_map = {
            'relu': 'relu',
            'tanh': 'tanh', 
            'sigmoid': 'logistic'
        }
        activation = activation_map.get(architecture.activation, 'relu')
        
        # Map optimizer
        solver_map = {
            'adam': 'adam',
            'sgd': 'sgd',
            'rmsprop': 'adam'  # Use adam as fallback for rmsprop
        }
        solver = solver_map.get(optimizer.optimizer_type.lower(), 'adam')
        
        # Create model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=regularization.l2_lambda or 0.0001,  # L2 regularization
            learning_rate_init=optimizer.learning_rate,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=regularization.early_stopping_patience,
            random_state=42
        )
        
        # Model info
        model_info = {
            'framework': 'sklearn',
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'architecture': architecture.to_dict(),
            'regularization': regularization.to_dict(),
            'optimizer': optimizer.to_dict()
        }
        
        return model, model_info
    
    def get_model_info(self, model: MLPRegressor) -> Dict:
        """Get scikit-learn model information"""
        # Try to get layer information if model is fitted
        try:
            n_layers = model.n_layers_
            coefs_shapes = [coef.shape for coef in model.coefs_]
            total_params = sum(coef.size for coef in model.coefs_) + sum(intercept.size for intercept in model.intercepts_)
        except AttributeError:
            # Model not fitted yet
            n_layers = len(model.hidden_layer_sizes) + 2  # +input +output
            coefs_shapes = []
            total_params = 0
        
        return {
            'framework': 'sklearn',
            'n_layers': n_layers,
            'coefs_shapes': coefs_shapes,
            'total_params': total_params,
            'hidden_layer_sizes': model.hidden_layer_sizes,
            'activation': model.activation,
            'solver': model.solver
        }


class ArchitectureFactory:
    """Factory for creating standard architecture configurations"""
    
    @staticmethod
    def get_standard_architectures() -> List[ArchitectureConfig]:
        """Get standard architecture configurations"""
        return [
            # Simple architectures
            ArchitectureConfig("simple_32", [32], "relu"),
            ArchitectureConfig("simple_64", [64], "relu"),
            ArchitectureConfig("simple_128", [128], "relu"),
            
            # Two-layer architectures
            ArchitectureConfig("deep_64_32", [64, 32], "relu"),
            ArchitectureConfig("deep_128_64", [128, 64], "relu"),
            ArchitectureConfig("deep_256_128", [256, 128], "relu"),
            
            # Three-layer architectures
            ArchitectureConfig("deep_128_64_32", [128, 64, 32], "relu"),
            ArchitectureConfig("deep_256_128_64", [256, 128, 64], "relu"),
            
            # Wide architectures
            ArchitectureConfig("wide_256", [256], "relu"),
            ArchitectureConfig("wide_512", [512], "relu"),
            
            # Alternative activations
            ArchitectureConfig("tanh_64_32", [64, 32], "tanh"),
            ArchitectureConfig("sigmoid_64_32", [64, 32], "sigmoid"),
        ]
    
    @staticmethod
    def get_standard_regularizations() -> List[RegularizationConfig]:
        """Get standard regularization configurations"""
        return [
            # No regularization
            RegularizationConfig("none"),
            
            # Dropout only
            RegularizationConfig("dropout_01", dropout_rate=0.1),
            RegularizationConfig("dropout_02", dropout_rate=0.2),
            RegularizationConfig("dropout_03", dropout_rate=0.3),
            
            # L2 regularization
            RegularizationConfig("l2_001", l2_lambda=0.001),
            RegularizationConfig("l2_01", l2_lambda=0.01),
            
            # L1 regularization
            RegularizationConfig("l1_001", l1_lambda=0.001),
            RegularizationConfig("l1_01", l1_lambda=0.01),
            
            # Combined regularization
            RegularizationConfig("dropout_l2", dropout_rate=0.2, l2_lambda=0.001),
            RegularizationConfig("batch_norm", batch_norm=True),
            RegularizationConfig("batch_norm_dropout", batch_norm=True, dropout_rate=0.2),
        ]
    
    @staticmethod
    def get_standard_optimizers() -> List[OptimizerConfig]:
        """Get standard optimizer configurations"""
        return [
            # Adam variants
            OptimizerConfig("adam_001", "adam", 0.001),
            OptimizerConfig("adam_0001", "adam", 0.0001),
            OptimizerConfig("adam_01", "adam", 0.01),
            
            # RMSprop variants
            OptimizerConfig("rmsprop_001", "rmsprop", 0.001),
            OptimizerConfig("rmsprop_0001", "rmsprop", 0.0001),
            
            # SGD variants
            OptimizerConfig("sgd_001", "sgd", 0.001, momentum=0.9),
            OptimizerConfig("sgd_01", "sgd", 0.01, momentum=0.9),
            
            # AdamW
            OptimizerConfig("adamw_001", "adamw", 0.001, decay=0.01),
        ]


class ModelBuilderManager:
    """Main model builder manager that selects appropriate backend"""
    
    def __init__(self, preferred_backend: str = 'auto'):
        """
        Initialize model builder manager
        
        Args:
            preferred_backend: 'tensorflow', 'pytorch', 'sklearn', or 'auto'
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backend = self._select_backend(preferred_backend)
        self.builder = self._create_builder()
        
        self.logger.info(f"Using {self.backend} backend for model building")
    
    def _select_backend(self, preferred: str) -> str:
        """Select best available backend"""
        if preferred == 'tensorflow' and TF_AVAILABLE:
            return 'tensorflow'
        elif preferred == 'pytorch' and TORCH_AVAILABLE:
            return 'pytorch'
        elif preferred == 'sklearn' and SKLEARN_AVAILABLE:
            return 'sklearn'
        elif preferred == 'auto':
            if TF_AVAILABLE:
                return 'tensorflow'
            elif TORCH_AVAILABLE:
                return 'pytorch'
            elif SKLEARN_AVAILABLE:
                return 'sklearn'
        
        available = []
        if TF_AVAILABLE:
            available.append('tensorflow')
        if TORCH_AVAILABLE:
            available.append('pytorch')
        if SKLEARN_AVAILABLE:
            available.append('sklearn')
        
        if not available:
            raise RuntimeError("No suitable backend available. Install tensorflow, pytorch, or scikit-learn")
        
        raise RuntimeError(f"Preferred backend '{preferred}' not available. Available: {available}")
    
    def _create_builder(self) -> BaseModelBuilder:
        """Create appropriate model builder"""
        if self.backend == 'tensorflow':
            return TensorFlowModelBuilder()
        elif self.backend == 'pytorch':
            return PyTorchModelBuilder()
        elif self.backend == 'sklearn':
            return SklearnModelBuilder()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def build_model(self, architecture: ArchitectureConfig,
                   regularization: RegularizationConfig,
                   optimizer: OptimizerConfig,
                   input_dim: int) -> Tuple[Any, Dict]:
        """Build model using selected backend"""
        return self.builder.build_model(architecture, regularization, optimizer, input_dim)
    
    def get_model_info(self, model: Any) -> Dict:
        """Get model information"""
        return self.builder.get_model_info(model)
    
    def get_all_configurations(self) -> Dict[str, List]:
        """Get all standard configurations"""
        return {
            'architectures': ArchitectureFactory.get_standard_architectures(),
            'regularizations': ArchitectureFactory.get_standard_regularizations(),
            'optimizers': ArchitectureFactory.get_standard_optimizers()
        }
    
    def get_backend_info(self) -> Dict:
        """Get information about available backends"""
        return {
            'current_backend': self.backend,
            'tensorflow_available': TF_AVAILABLE,
            'pytorch_available': TORCH_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE
        }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test backend availability
    print("=== Backend Availability ===")
    print(f"TensorFlow: {TF_AVAILABLE}")
    print(f"PyTorch: {TORCH_AVAILABLE}")
    print(f"Scikit-learn: {SKLEARN_AVAILABLE}")
    
    # Test model builder manager
    try:
        manager = ModelBuilderManager('auto')
        print(f"\nUsing backend: {manager.backend}")
        
        # Get configurations
        configs = manager.get_all_configurations()
        print(f"Available architectures: {len(configs['architectures'])}")
        print(f"Available regularizations: {len(configs['regularizations'])}")
        print(f"Available optimizers: {len(configs['optimizers'])}")
        
        # Test simple model creation
        arch = ArchitectureConfig("test", [64, 32], "relu")
        reg = RegularizationConfig("test", dropout_rate=0.2)
        opt = OptimizerConfig("test", "adam", 0.001)
        
        model, info = manager.build_model(arch, reg, opt, input_dim=10)
        print(f"\nModel info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()