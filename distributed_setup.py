#!/usr/bin/env python3
"""
Distributed Multi-Node Setup for ANNrun_code
Configure and run experiments across multiple nodes/machines
"""

import os
import sys
import json
import argparse
import socket
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

class MultiNodeManager:
    """Manager for multi-node distributed experiments"""
    
    def __init__(self, experiment_plan_path: str, total_nodes: int):
        self.experiment_plan_path = Path(experiment_plan_path)
        self.total_nodes = total_nodes
        self.base_output_dir = Path("./distributed_results")
        
        # Load experiment plan
        self.experiment_plan = pd.read_csv(experiment_plan_path)
        self.total_experiments = len(self.experiment_plan)
        
        print(f"Loaded {self.total_experiments} experiments for {total_nodes} nodes")
    
    def split_experiments(self) -> List[Dict]:
        """Split experiments across nodes"""
        experiments_per_node = self.total_experiments // self.total_nodes
        remainder = self.total_experiments % self.total_nodes
        
        node_assignments = []
        start_idx = 0
        
        for node_id in range(self.total_nodes):
            # Distribute remainder among first nodes
            current_count = experiments_per_node + (1 if node_id < remainder else 0)
            end_idx = start_idx + current_count
            
            node_experiments = self.experiment_plan.iloc[start_idx:end_idx].copy()
            
            assignment = {
                'node_id': node_id + 1,
                'start_experiment': start_idx + 1,
                'end_experiment': end_idx,
                'experiment_count': current_count,
                'experiments': node_experiments
            }
            
            node_assignments.append(assignment)
            start_idx = end_idx
        
        return node_assignments
    
    def create_node_configs(self, node_assignments: List[Dict]) -> Dict[str, Path]:
        """Create configuration files for each node"""
        config_dir = self.base_output_dir / "node_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_files = {}
        
        for assignment in node_assignments:
            node_id = assignment['node_id']
            
            # Create node-specific experiment plan
            node_plan_file = config_dir / f"experiment_plan_node_{node_id}.csv"
            assignment['experiments'].to_csv(node_plan_file, index=False)
            
            # Create node configuration
            node_config = {
                'node_id': node_id,
                'total_nodes': self.total_nodes,
                'experiment_plan_file': str(node_plan_file),
                'output_dir': str(self.base_output_dir / f"node_{node_id}_results"),
                'start_experiment': assignment['start_experiment'],
                'end_experiment': assignment['end_experiment'],
                'experiment_count': assignment['experiment_count']
            }
            
            config_file = config_dir / f"node_{node_id}_config.json"
            with open(config_file, 'w') as f:
                json.dump(node_config, f, indent=2)
            
            config_files[f"node_{node_id}"] = config_file
            
            print(f"Node {node_id}: {assignment['experiment_count']} experiments "
                  f"(IDs {assignment['start_experiment']}-{assignment['end_experiment']})")
        
        return config_files
    
    def generate_run_scripts(self, node_assignments: List[Dict], 
                           backend: str = 'joblib', 
                           gpu_enabled: bool = True) -> Dict[str, Path]:
        """Generate run scripts for each node"""
        scripts_dir = self.base_output_dir / "run_scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_files = {}
        
        for assignment in node_assignments:
            node_id = assignment['node_id']
            
            # Determine GPU settings
            gpu_args = ""
            if gpu_enabled:
                gpu_args = f"--gpu-ids 0 --backend pytorch"
            
            # Create run script
            script_content = f"""#!/bin/bash
# Multi-node experiment runner for Node {node_id}
# Generated automatically - do not edit manually

set -e  # Exit on any error

echo "=========================================="
echo "Starting Node {node_id} of {self.total_nodes}"
echo "Experiments: {assignment['start_experiment']}-{assignment['end_experiment']} ({assignment['experiment_count']} total)"
echo "=========================================="

# Check environment
echo "Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {{torch.__version__}}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {{tf.__version__}}')"

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
fi

# Set output directory
export NODE_OUTPUT_DIR="{self.base_output_dir / f"node_{node_id}_results"}"
mkdir -p "$NODE_OUTPUT_DIR"

# Run experiments
echo "Starting experiments..."
python main.py parallel \\
    node_configs/experiment_plan_node_{node_id}.csv \\
    1 \\
    --parallel-backend {backend} \\
    --n-jobs -1 \\
    --output-dir "$NODE_OUTPUT_DIR" \\
    {gpu_args} \\
    --log-level INFO \\
    2>&1 | tee "$NODE_OUTPUT_DIR/node_{node_id}_run.log"

echo "=========================================="
echo "Node {node_id} completed successfully!"
echo "Results saved to: $NODE_OUTPUT_DIR"
echo "=========================================="
"""
            
            script_file = scripts_dir / f"run_node_{node_id}.sh"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_file, 0o755)
            
            script_files[f"node_{node_id}"] = script_file
        
        return script_files
    
    def generate_slurm_scripts(self, node_assignments: List[Dict],
                             partition: str = "gpu",
                             time_limit: str = "24:00:00",
                             memory: str = "32GB") -> Dict[str, Path]:
        """Generate SLURM scripts for HPC clusters"""
        slurm_dir = self.base_output_dir / "slurm_scripts"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        
        slurm_files = {}
        
        for assignment in node_assignments:
            node_id = assignment['node_id']
            
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name=ann_node_{node_id}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_node_{node_id}_%j.out
#SBATCH --error=slurm_node_{node_id}_%j.err

# Environment setup
module load python/3.10
module load cuda/11.8
source .venv/bin/activate

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run the node script
echo "Running Node {node_id} on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

bash run_scripts/run_node_{node_id}.sh

echo "Node {node_id} completed on $(hostname)"
"""
            
            slurm_file = slurm_dir / f"submit_node_{node_id}.slurm"
            with open(slurm_file, 'w') as f:
                f.write(slurm_content)
            
            slurm_files[f"node_{node_id}"] = slurm_file
        
        # Create submit all script
        submit_all_content = """#!/bin/bash
# Submit all SLURM jobs

echo "Submitting all node jobs..."
"""
        
        for node_id in range(1, self.total_nodes + 1):
            submit_all_content += f"""
echo "Submitting Node {node_id}..."
sbatch submit_node_{node_id}.slurm
"""
        
        submit_all_content += """
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel -u $USER"
"""
        
        submit_all_file = slurm_dir / "submit_all.sh"
        with open(submit_all_file, 'w') as f:
            f.write(submit_all_content)
        os.chmod(submit_all_file, 0o755)
        
        return slurm_files
    
    def create_monitoring_script(self) -> Path:
        """Create monitoring script for tracking progress"""
        monitor_script = self.base_output_dir / "monitor_progress.py"
        
        monitor_content = f'''#!/usr/bin/env python3
"""
Monitor progress of distributed experiments
"""

import os
import time
import json
from pathlib import Path
import pandas as pd

def monitor_progress():
    """Monitor progress across all nodes"""
    base_dir = Path("{self.base_output_dir}")
    total_experiments = {self.total_experiments}
    total_nodes = {self.total_nodes}
    
    print(f"Monitoring {{total_nodes}} nodes, {{total_experiments}} total experiments")
    print("=" * 60)
    
    while True:
        completed_total = 0
        
        for node_id in range(1, total_nodes + 1):
            node_dir = base_dir / f"node_{{node_id}}_results"
            
            if node_dir.exists():
                # Count completed experiments (look for result files)
                result_files = list(node_dir.glob("**/results.json"))
                completed_node = len(result_files)
                completed_total += completed_node
                
                # Check log file for status
                log_files = list(node_dir.glob("*.log"))
                status = "Running" if log_files else "Not started"
                
                print(f"Node {{node_id:2d}}: {{completed_node:3d}} completed - {{status}}")
            else:
                print(f"Node {{node_id:2d}}: Not started")
        
        progress_pct = (completed_total / total_experiments) * 100
        print(f"Total Progress: {{completed_total}}/{{total_experiments}} ({{progress_pct:.1f}}%)")
        print("=" * 60)
        
        if completed_total >= total_experiments:
            print("🎉 All experiments completed!")
            break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\\nMonitoring stopped by user")
'''
        
        with open(monitor_script, 'w') as f:
            f.write(monitor_content)
        os.chmod(monitor_script, 0o755)
        
        return monitor_script
    
    def setup_distributed_run(self, backend: str = 'joblib', 
                            gpu_enabled: bool = True,
                            create_slurm: bool = False) -> Dict[str, Any]:
        """Complete setup for distributed run"""
        print(f"Setting up distributed run for {self.total_nodes} nodes...")
        
        # Split experiments
        node_assignments = self.split_experiments()
        
        # Create configurations
        config_files = self.create_node_configs(node_assignments)
        
        # Create run scripts
        script_files = self.generate_run_scripts(node_assignments, backend, gpu_enabled)
        
        # Create SLURM scripts if requested
        slurm_files = {}
        if create_slurm:
            slurm_files = self.generate_slurm_scripts(node_assignments)
        
        # Create monitoring script
        monitor_script = self.create_monitoring_script()
        
        setup_info = {
            'total_experiments': self.total_experiments,
            'total_nodes': self.total_nodes,
            'experiments_per_node': [a['experiment_count'] for a in node_assignments],
            'config_files': config_files,
            'script_files': script_files,
            'slurm_files': slurm_files,
            'monitor_script': monitor_script,
            'output_directory': self.base_output_dir
        }
        
        # Save setup summary
        summary_file = self.base_output_dir / "setup_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({k: str(v) if isinstance(v, Path) else v 
                      for k, v in setup_info.items()}, f, indent=2)
        
        return setup_info


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup distributed multi-node experiments")
    parser.add_argument('experiment_plan', help='Experiment plan CSV file')
    parser.add_argument('nodes', type=int, help='Number of nodes')
    parser.add_argument('--backend', choices=['joblib', 'ray', 'multiprocessing'],
                       default='joblib', help='Parallel backend')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--slurm', action='store_true', help='Generate SLURM scripts')
    
    args = parser.parse_args()
    
    # Create manager
    manager = MultiNodeManager(args.experiment_plan, args.nodes)
    
    # Setup distributed run
    setup_info = manager.setup_distributed_run(
        backend=args.backend,
        gpu_enabled=not args.no_gpu,
        create_slurm=args.slurm
    )
    
    print(f"\\n🎯 Distributed setup completed!")
    print(f"Output directory: {setup_info['output_directory']}")
    print(f"Total experiments: {setup_info['total_experiments']}")
    print(f"Nodes: {setup_info['total_nodes']}")
    print(f"Experiments per node: {setup_info['experiments_per_node']}")
    
    print(f"\\n📋 Next steps:")
    print(f"1. Review configuration files in: distributed_results/node_configs/")
    print(f"2. Run on each node:")
    for i in range(1, args.nodes + 1):
        print(f"   Node {i}: bash distributed_results/run_scripts/run_node_{i}.sh")
    
    if args.slurm:
        print(f"\\n🖥️  For SLURM clusters:")
        print(f"   cd distributed_results/slurm_scripts/")
        print(f"   bash submit_all.sh")
    
    print(f"\\n📊 Monitor progress:")
    print(f"   python distributed_results/monitor_progress.py")


if __name__ == "__main__":
    main()
