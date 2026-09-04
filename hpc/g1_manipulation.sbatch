#!/bin/bash
# G1 — content-arm manipulation check (docs/STATE.md). Single-turn,
# no tool loop, ~208 generations of <=80 tokens each. Cheap: minutes of GPU time,
# not hours. Partition chosen by a fresh --test-only sweep on 2026-09-02
# (gpu-h100/gpu-l40s/gpu-a100 all multi-day; gpu-l40s-preempt started same-hour).
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root:
#   mkdir -p logs/slurm && sbatch hpc/g1_manipulation.sbatch
#SBATCH --job-name=g1_manip
#SBATCH -p gpu-l40s-preempt
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi

python -u scripts/27_g1_manipulation_check.py --out results/g1_manipulation_check.json

echo "Generation done. Fetch and grade on the laptop:"
echo "  bash hpc/fetch.sh 'g1_manipulation_check*'"
echo "  uv run python scripts/28_g1_grade.py --in results/g1_manipulation_check.json"
