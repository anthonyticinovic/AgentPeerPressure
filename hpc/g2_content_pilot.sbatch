#!/bin/bash
# G2 -- content-arm pilot (docs/STATE.md). 52 items (--one-per-grader),
# 4 new conditions, base arm only (no --ablate). Gated on G1 passing first.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root, after a fresh --test-only sweep (partition queues move fast --
# see the note in hpc/g1_manipulation.sbatch and STATE.md's partition history):
#   mkdir -p logs/slurm && sbatch hpc/g2_content_pilot.sbatch
#SBATCH --job-name=g2_pilot
#SBATCH -p gpu-l40s-preempt
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
  --one-per-grader \
  --max-turns 16 \
  --monitor \
  --no-judge \
  --out results/content_arm_pilot_base.json

echo "Generation done. Fetch and grade on the laptop:"
echo "  bash hpc/fetch.sh 'content_arm_pilot_base*'"
echo "  uv run python scripts/15_regrade.py --in results/content_arm_pilot_base.json --out results/content_arm_pilot_base_judged.json"
echo "  uv run python scripts/29_content_arm_analysis.py --content-base results/content_arm_pilot_base_judged.json"
