#!/usr/bin/env python3
"""
Parallel Experiment Runner for ANNrun_code
Supports multiple parallel processing backends:
- Joblib (recommended for local)
- Ray (for distributed)
- Multiprocessing (standard library)
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import json
import psutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Try importing optional dependencies
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Import our modules (these would be in the ANNrun_code structure)
# from core.models.neural_networks.architectures import ArchitectureBuilder
# from core.preprocessing.pipeline import PreprocessingPipeline
# from experiments.base_experiment import BaseExperiment


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    experiment_id: int
    features: List[str]
    normalization: str
    bias_correction: str
    architecture_config: Dict
    regularization_config: Dict
    optimizer_config: Dict
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'experiment_id': self.experiment_id,
            'features': self.features,
            'normalization': self.normalization,
            'bias_correction': self.bias_correction,
            'architecture_config': self.architecture_config,
            'regularization_config': self.regularization_config,
            'optimizer_config': self.optimizer_config,
            'description': self.description
        }


@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    backend: str = 'joblib'  # 'joblib', 'ray', 'multiprocessing'
    n_jobs: int = -1  # -1 for all cores
    batch_size: int = 1
    verbose: int = 1
    memory_limit_gb: float = None
    gpu_ids: List[int] = None
    
    def __post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        if self.gpu_ids is None:
            self.gpu_ids = []


class ResourceMonitor:
    """Monitor system resources during experiments"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_system_info(self) -> Dict:
        """Get current system resource information"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def check_resources(self, config: ParallelConfig) -> bool:
        """Check if system has enough resources"""
        info = self.get_system_info()
        
        # Check memory
        if config.memory_limit_gb:
            if info['memory_available_gb'] < config.memory_limit_gb:
                self.logger.warning(f"Low memory: {info['memory_available_gb']:.1f}GB available, "
                                  f"{config.memory_limit_gb}GB required")
                return False
        
        # Check CPU load
        if info['cpu_percent'] > 90:
            self.logger.warning(f"High CPU usage: {info['cpu_percent']:.1f}%")
            
        return True
    
    def log_system_info(self):
        """Log current system information"""
        info = self.get_system_info()
        self.logger.info(f"System: CPU {info['cpu_percent']:.1f}%, "
                        f"RAM {info['memory_percent']:.1f}% "
                        f"({info['memory_available_gb']:.1f}GB available)")


class JobLibRunner:
    """Joblib-based parallel runner"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_experiments(self, experiment_configs: List[ExperimentConfig], 
                       experiment_func: Callable) -> List[Dict]:
        """Run experiments in parallel using Joblib"""
        
        if not JOBLIB_AVAILABLE:
            raise ImportError("Joblib not available. Install with: pip install joblib")
        
        self.logger.info(f"Running {len(experiment_configs)} experiments with Joblib")
        self.logger.info(f"Using {self.config.n_jobs} parallel jobs")
        
        # Run experiments
        results = Parallel(
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            backend='loky'  # Most stable backend
        )(
            delayed(experiment_func)(config.to_dict()) 
            for config in experiment_configs
        )
        
        return [r for r in results if r is not None]


class RayRunner:
    """Ray-based distributed runner"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize_ray(self):
        """Initialize Ray cluster"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not available. Install with: pip install ray")
        
        if not ray.is_initialized():
            # Initialize Ray with resource limits
            ray_config = {
                'ignore_reinit_error': True,
                'logging_level': logging.INFO
            }
            
            if self.config.memory_limit_gb:
                ray_config['object_store_memory'] = int(self.config.memory_limit_gb * 1024**3 * 0.3)
            
            ray.init(**ray_config)
            
        self.logger.info(f"Ray initialized with {ray.cluster_resources()}")
    
    def run_experiments(self, experiment_configs: List[ExperimentConfig],
                       experiment_func: Callable) -> List[Dict]:
        """Run experiments using Ray"""
        
        self.initialize_ray()
        
        # Convert function to Ray remote
        @ray.remote
        def ray_experiment_func(config_dict):
            return experiment_func(config_dict)
        
        self.logger.info(f"Running {len(experiment_configs)} experiments with Ray")
        
        # Submit all jobs
        futures = [
            ray_experiment_func.remote(config.to_dict()) 
            for config in experiment_configs
        ]
        
        # Collect results with progress tracking
        results = []
        completed = 0
        
        while futures:
            ready, futures = ray.wait(futures, num_returns=1, timeout=10.0)
            
            for future in ready:
                try:
                    result = ray.get(future)
                    if result is not None:
                        results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        self.logger.info(f"Completed {completed}/{len(experiment_configs)} experiments")
                        
                except Exception as e:
                    self.logger.error(f"Experiment failed: {e}")
                    completed += 1
        
        return results


class MultiprocessingRunner:
    """Standard multiprocessing runner"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_experiments(self, experiment_configs: List[ExperimentConfig],
                       experiment_func: Callable) -> List[Dict]:
        """Run experiments using ProcessPoolExecutor"""
        
        self.logger.info(f"Running {len(experiment_configs)} experiments with ProcessPoolExecutor")
        self.logger.info(f"Using {self.config.n_jobs} processes")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(experiment_func, config.to_dict()): config
                for config in experiment_configs
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_config):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    
                    completed += 1
                    if completed % 10 == 0:
                        self.logger.info(f"Completed {completed}/{len(experiment_configs)} experiments")
                        
                except Exception as e:
                    config = future_to_config[future]
                    self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
        
        return results


class ParallelExperimentRunner:
    """Main parallel experiment runner that supports multiple backends"""
    
    def __init__(self, config: ParallelConfig = None):
        self.config = config or ParallelConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitor = ResourceMonitor()
        
        # Initialize backend runner
        self.runner = self._create_runner()
        
    def _create_runner(self):
        """Create appropriate runner based on backend"""
        if self.config.backend == 'joblib' and JOBLIB_AVAILABLE:
            return JobLibRunner(self.config)
        elif self.config.backend == 'ray' and RAY_AVAILABLE:
            return RayRunner(self.config)
        elif self.config.backend == 'multiprocessing':
            return MultiprocessingRunner(self.config)
        else:
            # Fallback to multiprocessing
            self.logger.warning(f"Backend '{self.config.backend}' not available, using multiprocessing")
            self.config.backend = 'multiprocessing'
            return MultiprocessingRunner(self.config)
    
    def run_experiments(self, experiment_configs: List[ExperimentConfig],
                       experiment_func: Callable,
                       output_dir: str = None) -> List[Dict]:
        """
        Run experiments in parallel
        
        Args:
            experiment_configs: List of experiment configurations
            experiment_func: Function to run single experiment
            output_dir: Output directory for results
            
        Returns:
            List of experiment results
        """
        
        self.logger.info(f"Starting parallel experiment runner")
        self.logger.info(f"Backend: {self.config.backend}")
        self.logger.info(f"Number of experiments: {len(experiment_configs)}")
        self.logger.info(f"Parallel jobs: {self.config.n_jobs}")
        
        # Check system resources
        if not self.monitor.check_resources(self.config):
            self.logger.warning("System resources may be insufficient")
        
        self.monitor.log_system_info()
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Run experiments
        start_time = time.time()
        
        try:
            results = self.runner.run_experiments(experiment_configs, experiment_func)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Experiments completed in {duration:.1f} seconds")
            self.logger.info(f"Successful experiments: {len(results)}/{len(experiment_configs)}")
            
            # Save results if output directory provided
            if output_dir and results:
                self._save_results(results, output_path)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_results(self, results: List[Dict], output_path: Path):
        """Save experiment results"""
        
        # Save as CSV
        results_df = pd.DataFrame(results)
        csv_file = output_path / 'parallel_experiment_results.csv'
        results_df.to_csv(csv_file, index=False)
        self.logger.info(f"Results saved to {csv_file}")
        
        # Save as JSON for full detail
        json_file = output_path / 'parallel_experiment_results.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Detailed results saved to {json_file}")
        
        # Save summary
        summary = {
            'total_experiments': len(results),
            'backend_used': self.config.backend,
            'parallel_jobs': self.config.n_jobs,
            'best_rmse': min(r.get('cv_rmse_mean', float('inf')) for r in results),
            'worst_rmse': max(r.get('cv_rmse_mean', 0) for r in results)
        }
        
        summary_file = output_path / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to {summary_file}")


def create_experiment_configs(plan_file: str, experiment_id: int) -> List[ExperimentConfig]:
    """
    Create experiment configurations from plan file
    This is a placeholder - you would implement based on your experiment plan format
    """
    
    # Example implementation
    df = pd.read_csv(plan_file)
    row = df[df['id'] == experiment_id].iloc[0]
    
    # This would be expanded to create multiple architecture/regularization combinations
    configs = []
    
    # Example: create configs for different architectures
    architectures = [
        {'layers': [32], 'activation': 'relu'},
        {'layers': [64], 'activation': 'relu'},
        {'layers': [64, 32], 'activation': 'relu'}
    ]
    
    regularizations = [
        {'type': 'none'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'l2', 'lambda': 0.001}
    ]
    
    optimizers = [
        {'type': 'adam', 'lr': 0.001},
        {'type': 'adam', 'lr': 0.0001}
    ]
    
    config_id = 0
    for arch in architectures:
        for reg in regularizations:
            for opt in optimizers:
                config = ExperimentConfig(
                    experiment_id=config_id,
                    features=row['features'].split(),
                    normalization=row['normalization'],
                    bias_correction=row['bias_correction'],
                    architecture_config=arch,
                    regularization_config=reg,
                    optimizer_config=opt,
                    description=f"Arch:{arch['layers']}_Reg:{reg['type']}_Opt:{opt['type']}"
                )
                configs.append(config)
                config_id += 1
    
    return configs


def run_single_experiment(config_dict: Dict) -> Optional[Dict]:
    """
    Run a single experiment - this would be implemented based on your experiment logic
    This is a placeholder that shows the expected interface
    """
    
    try:
        # Initialize components (this would use your actual modules)
        # data_manager = EnhancedDataManager()
        # model_builder = ArchitectureBuilder()
        # preprocessor = PreprocessingPipeline()
        
        # Simulate experiment
        import random
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        
        # Return mock result
        return {
            'experiment_id': config_dict['experiment_id'],
            'config_name': config_dict['description'],
            'cv_rmse_mean': random.uniform(0.1, 1.0),
            'cv_rmse_std': random.uniform(0.01, 0.1),
            'train_r2': random.uniform(0.7, 0.95),
            'training_time': random.uniform(10, 60),
            'features_used': config_dict['features'],
            'architecture': config_dict['architecture_config'],
            'regularization': config_dict['regularization_config'],
            'optimizer': config_dict['optimizer_config']
        }
        
    except Exception as e:
        logging.error(f"Experiment {config_dict['experiment_id']} failed: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        backend='joblib',  # or 'ray', 'multiprocessing'
        n_jobs=4,          # Use 4 cores
        verbose=1
    )
    
    # Create experiment runner
    runner = ParallelExperimentRunner(parallel_config)
    
    # Create sample experiment configurations
    configs = create_experiment_configs('experiment_plan.csv', 1)
    
    print(f"Created {len(configs)} experiment configurations")
    print(f"Running with backend: {parallel_config.backend}")
    
    # Run experiments
    results = runner.run_experiments(
        experiment_configs=configs,
        experiment_func=run_single_experiment,
        output_dir='./parallel_results'
    )
    
    print(f"Completed {len(results)} experiments successfully")
    
    # Show best result
    if results:
        best_result = min(results, key=lambda x: x.get('cv_rmse_mean', float('inf')))
        print(f"Best result: {best_result['config_name']}")
        print(f"RMSE: {best_result['cv_rmse_mean']:.4f}")
