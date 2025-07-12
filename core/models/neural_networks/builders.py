#!/usr/bin/env python3
"""
Modular Neural Network Architecture Builder for ANNrun_code
Supports TensorFlow and PyTorch backends with comprehensive architecture options
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import tensorflow as tf

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

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam as TorchAdam, SGD as TorchSGD
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Scikit-learn fallback
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
    
    def build_model(self, architecture: ArchitectureConfig,
                   regularization: RegularizationConfig, 
                   optimizer: OptimizerConfig,
                   input_dim: int) -> Tuple[tf.keras.Model, Dict]:
        """Build TensorFlow model"""
        
        self.logger.debug(f"Building TensorFlow model: {architecture.name}")
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(architecture.layers):
            
            # Dense layer with optional regularization
            kernel_regularizer = self._create_kernel_regularizer(regularization)
            
            model.add(Dense(
                units=units,
                activation=architecture.activation,
                use_bias=architecture.use_bias,
                kernel_initializer=architecture.kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'dense_{i+1}'
            ))
            
            # Batch normalization
            if regularization.batch_norm:
                model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Dropout
            if regularization.dropout_rate and regularization.dropout_rate > 0:
                model.add(Dropout(regularization.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        output_activation = architecture.output_activation or 'linear'
        model.add(Dense(1, activation=output_activation, name='output'))
        
        # Create optimizer
        optimizer_instance = self._create_optimizer(optimizer)
        
        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        # Model info
        model_info = {
            'framework': 'tensorflow',
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'architecture': architecture.to_dict(),
            'regularization': regularization.to_dict(),
            'optimizer': optimizer.to_dict()
        }
        
        return model, model_info
    
    def _create_kernel_regularizer(self, reg_config: RegularizationConfig):
        """Create kernel regularizer"""
        if reg_config.l1_lambda and reg_config.l2_lambda:
            return l1_l2(l1=reg_config.l1_lambda, l2=reg_config.l2_lambda)
        elif reg_config.l1_lambda:
            return l1(reg_config.l1_lambda)
        elif reg_config.l2_lambda:
            return l2(reg_config.l2_lambda)
        return None
    
    def _create_optimizer(self, opt_config: OptimizerConfig):
        """Create optimizer instance"""
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
    """PyTorch model builder"""
    
    def __init__(self):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
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
                    
                    if regularization.dropout_rate and regularization.dropout_rate > 0:
                        layers.append(nn.Dropout(regularization.dropout_rate))
                    
                    in_features = units
                
                # Output layer
                layers.append(nn.Linear(in_features, 1, bias=architecture.use_bias))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return NeuralNetwork()
    
    def _create_pytorch_optimizer(self, model: nn.Module, opt_config: OptimizerConfig):
        """Create PyTorch optimizer"""
        if opt_config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=opt_config.learning_rate,
                betas=(opt_config.beta_1 or 0.9, opt_config.beta_2 or 0.999),
                eps=opt_config.epsilon or 1e-8
            )
        elif opt_config.optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0,
                eps=opt_config.epsilon or 1e-8
            )
        elif opt_config.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum or 0.0
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
            'device': next(model.parameters()).device
        }


class SklearnModelBuilder(BaseModelBuilder):
    """Scikit-learn model builder (fallback)"""
    
    def __init__(self):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
    
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
        
        # Configure regularization
        alpha = regularization.l2_lambda or 0.0001
        
        # Configure optimizer
        solver = 'adam' if optimizer.optimizer_type.lower() == 'adam' else 'lbfgs'
        learning_rate_init = optimizer.learning_rate
        
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=regularization.early_stopping_patience,
            random_state=42
        )
        
        # Calculate approximate parameter count
        param_count = self._estimate_param_count(hidden_layer_sizes, input_dim)
        
        model_info = {
            'framework': 'sklearn',
            'total_params': param_count,
            'trainable_params': param_count,
            'architecture': architecture.to_dict(),
            'regularization': regularization.to_dict(),
            'optimizer': optimizer.to_dict()
        }
        
        return model, model_info
    
    def _estimate_param_count(self, hidden_layer_sizes: Tuple[int, ...], input_dim: int) -> int:
        """Estimate parameter count for sklearn MLP"""
        param_count = 0
        prev_size = input_dim
        
        for size in hidden_layer_sizes:
            param_count += prev_size * size + size  # weights + biases
            prev_size = size
        
        param_count += prev_size * 1 + 1  # output layer
        return param_count
    
    def get_model_info(self, model: MLPRegressor) -> Dict:
        """Get sklearn model information"""
        if hasattr(model, 'coefs_'):
            total_params = sum(coef.size + intercept.size for coef, intercept in zip(model.coefs_, model.intercepts_))
        else:
            total_params = 0
            
        return {
            'framework': 'sklearn',
            'total_params': total_params,
            'trainable_params': total_params,
            'layers': len(model.hidden_layer_sizes) + 1 if hasattr(model, 'hidden_layer_sizes') else 0
        }


class ArchitectureFactory:
    """Factory for creating standard architecture configurations"""
    
    @staticmethod
    def get_standard_architectures() -> List[ArchitectureConfig]:
        """Get predefined standard architectures"""
        return [
            # Single layer architectures
            ArchitectureConfig(name="single_32", layers=[32]),
            ArchitectureConfig(name="single_64", layers=[64]),
            ArchitectureConfig(name="single_128", layers=[128]),
            
            # Double layer architectures
            ArchitectureConfig(name="double_32_16", layers=[32, 16]),
            ArchitectureConfig(name="double_64_32", layers=[64, 32]),
            ArchitectureConfig(name="double_128_64", layers=[128, 64]),
            ArchitectureConfig(name="double_64_64", layers=[64, 64]),
            
            # Triple layer architectures
            ArchitectureConfig(name="triple_128_64_32", layers=[128, 64, 32]),
            ArchitectureConfig(name="triple_64_32_16", layers=[64, 32, 16]),
            ArchitectureConfig(name="triple_32_32_16", layers=[32, 32, 16]),
            
            # Deep architectures
            ArchitectureConfig(name="deep_256_128_64_32", layers=[256, 128, 64, 32]),
            ArchitectureConfig(name="deep_128_128_64_32", layers=[128, 128, 64, 32]),
        ]
    
    @staticmethod
    def get_standard_regularizations() -> List[RegularizationConfig]:
        """Get predefined regularization configurations"""
        return [
            RegularizationConfig(name="none"),
            RegularizationConfig(name="dropout_01", dropout_rate=0.1),
            RegularizationConfig(name="dropout_02", dropout_rate=0.2),
            RegularizationConfig(name="dropout_03", dropout_rate=0.3),
            RegularizationConfig(name="batch_norm", batch_norm=True),
            RegularizationConfig(name="l2_light", l2_lambda=0.001),
            RegularizationConfig(name="l1l2_medium", l1_lambda=0.001, l2_lambda=0.001),
            RegularizationConfig(name="dropout_batch_norm", dropout_rate=0.2, batch_norm=True),
            RegularizationConfig(name="full_regularization", dropout_rate=0.2, l2_lambda=0.001, batch_norm=True)
        ]
    
    @staticmethod
    def get_standard_optimizers() -> List[OptimizerConfig]:
        """Get predefined optimizer configurations"""
        return [
            OptimizerConfig(name="adam_default", optimizer_type="adam", learning_rate=0.001),
            OptimizerConfig(name="adam_low", optimizer_type="adam", learning_rate=0.0001),
            OptimizerConfig(name="rmsprop_default", optimizer_type="rmsprop", learning_rate=0.001),
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
        
        raise RuntimeError("No suitable backend available")
    
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
    
    def count_total_combinations(self) -> int:
        """Count total possible combinations"""
        configs = self.get_all_configurations()
        return (len(configs['architectures']) * 
                len(configs['regularizations']) * 
                len(configs['optimizers']))


# Utility functions
def create_model_builder(backend: str = 'auto') -> ModelBuilderManager:
    """Create model builder manager"""
    return ModelBuilderManager(backend)


def save_model_config(config: Dict, filepath: str):
    """Save model configuration to file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(filepath: str) -> Dict:
    """Load model configuration from file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model builder
    builder_manager = create_model_builder('auto')
    
    print(f"Using backend: {builder_manager.backend}")
    print(f"Total combinations: {builder_manager.count_total_combinations()}")
    
    # Get standard configurations
    configs = builder_manager.get_all_configurations()
    
    print(f"\nAvailable architectures: {len(configs['architectures'])}")
    for arch in configs['architectures'][:3]:
        print(f"  - {arch.name}: {arch.layers}")
    
    print(f"\nAvailable regularizations: {len(configs['regularizations'])}")
    for reg in configs['regularizations'][:3]:
        print(f"  - {reg.name}")
    
    print(f"\nAvailable optimizers: {len(configs['optimizers'])}")
    for opt in configs['optimizers']:
        print(f"  - {opt.name}: {opt.optimizer_type} (lr={opt.learning_rate})")
    
    # Test model building
    try:
        arch = configs['architectures'][0]
        reg = configs['regularizations'][0]
        opt = configs['optimizers'][0]
        
        model, info = builder_manager.build_model(arch, reg, opt, input_dim=10)
        print(f"\nTest model built successfully:")
        print(f"  Framework: {info['framework']}")
        print(f"  Total parameters: {info['total_params']}")
        print(f"  Architecture: {arch.name} - {arch.layers}")
        
    except Exception as e:
        print(f"\nModel building test failed: {e}")
    
    print("\nNeural network builder ready for use!")
            