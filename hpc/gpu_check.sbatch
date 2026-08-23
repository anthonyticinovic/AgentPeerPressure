#!/bin/bash
# Two-minute GPU sanity job. Runs before any weights exist, so it is the fastest
# possible answer to: does the venv see CUDA, is it an H100, is bf16 native, and
# does the CUDA branch of device.py return what the run assumes?
#
#   mkdir -p logs/slurm && sbatch hpc/gpu_check.sbatch
#SBATCH --job-name=gpu_check
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi
echo
python -c "
import sys; sys.path.insert(0, 'src')
from pressure import device as dv
import json; print(json.dumps(dv.describe(), indent=1))
"
echo
python scripts/17_cluster_preflight.py --skip-model
