#!/bin/bash
# G1 retry with a stronger referent (docs/STATE.md, 2026-09-02). Same cost profile
# as hpc/g1_manipulation.sbatch: single-turn, ~208 short generations, minutes of
# GPU time.
#
#   mkdir -p logs/slurm && sbatch hpc/g1_strong_referent.sbatch
#SBATCH --job-name=g1_strong
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

python -u scripts/30_g1_strong_referent_check.py --out results/g1_strong_referent.json

echo "Generation done. Fetch and grade on the laptop:"
echo "  bash hpc/fetch.sh 'g1_strong_referent*'"
echo "  uv run python scripts/28_g1_grade.py --in results/g1_strong_referent.json"
