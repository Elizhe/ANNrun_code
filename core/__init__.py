"""
Core package for ANNrun_code
Robust import handling with graceful fallbacks
"""

import logging
import warnings

# Initialize logger
logger = logging.getLogger(__name__)

# Core imports with error handling
try:
    from .data.data_manager import EnhancedDataManager
    logger.info("Successfully imported EnhancedDataManager")
except ImportError as e:
    logger.warning(f"Could not import EnhancedDataManager: {e}")
    EnhancedDataManager = None

try:
    from .models.neural_networks.builders import ModelBuilderManager
    logger.info("Successfully imported ModelBuilderManager")
except ImportError as e:
    logger.warning(f"Could not import ModelBuilderManager: {e}")
    ModelBuilderManager = None

try:
    from .preprocessing.bias_correction.corrector import BiasCorrector
    logger.info("Successfully imported BiasCorrector")
except ImportError as e:
    logger.warning(f"Could not import BiasCorrector: {e}")
    BiasCorrector = None

try:
    from .utils.logging_config import setup_logging
    logger.info("Successfully imported setup_logging")
except ImportError as e:
    logger.warning(f"Could not import setup_logging: {e}")
    setup_logging = None

# Optional imports
try:
    from .models.lumen.model import LumenModel
    logger.info("Successfully imported LumenModel")
except ImportError as e:
    logger.warning(f"Could not import LumenModel: {e}")
    LumenModel = None

# Build __all__ list with available imports
__all__ = []
for name, obj in [
    ('EnhancedDataManager', EnhancedDataManager),
    ('ModelBuilderManager', ModelBuilderManager),
    ('BiasCorrector', BiasCorrector),
    ('setup_logging', setup_logging),
    ('LumenModel', LumenModel)
]:
    if obj is not None:
        __all__.append(name)

logger.info(f"Core package initialized with: {__all__}")

def check_imports():
    """Check which imports are available"""
    available = {}
    for name in ['EnhancedDataManager', 'ModelBuilderManager', 'BiasCorrector', 'setup_logging', 'LumenModel']:
        available[name] = globals().get(name) is not None
    return available

def get_import_status():
    """Get detailed import status"""
    status = check_imports()
    available_count = sum(status.values())
    total_count = len(status)
    
    return {
        'available': status,
        'available_count': available_count,
        'total_count': total_count,
        'success_rate': available_count / total_count if total_count > 0 else 0
    }
