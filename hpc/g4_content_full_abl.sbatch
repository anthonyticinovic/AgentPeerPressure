#!/bin/bash
# G4 -- content-arm full run, ablated arm (refusal direction ablated). Same as
# G3 (hpc/g3_content_full_base.sbatch) plus --ablate. Submitted in parallel with
# G3 to different --out paths on independent GPUs, per docs/PLAN_content_arm.md
# #5's parallel-arms decision.
#
#   mkdir -p logs/slurm && sbatch hpc/g4_content_full_abl.sbatch
#SBATCH --job-name=g4_abl
#SBATCH -p gpu-a100-preempt
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi
python scripts/17_cluster_preflight.py

python -u scripts/12_peer_loop.py \
  --conditions C8 C8b C9 C8n \
  --max-turns 16 \
  --max-new-tokens 1536 \
  --monitor \
  --ablate \
  --no-judge \
  --out results/content_arm_full_abl.json

echo "Generation done. Fetch and grade on the laptop:"
echo "  bash hpc/fetch.sh 'content_arm_full_abl*'"
echo "  uv run python scripts/15_regrade.py --in results/content_arm_full_abl.json --out results/content_arm_full_abl_judged.json"
