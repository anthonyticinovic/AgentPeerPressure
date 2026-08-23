#!/bin/bash
# Gate P — Qwen3.5-9B, 208 items x 4 conditions, 832 rows.
#
# Grading runs WITHOUT the semantic judge here, because compute nodes have no
# outbound network. That is not a compromise: transcripts are stored, so the judge
# is applied afterwards for a few cents with scripts/15_regrade.py. Never analyse
# the file this job writes — analyse the regraded one.
#
# Resumable. 12_peer_loop checkpoints every 10 rows and skips any row that already
# has a transcript, so re-submitting the identical command continues where it
# stopped; --requeue makes a pre-emption restart do that by itself.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root:
#   mkdir -p logs/slurm && sbatch hpc/gate_p.sbatch
#SBATCH --job-name=gatep_9b
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi
python scripts/17_cluster_preflight.py

python -u scripts/12_peer_loop.py \
  --conditions C0 C1b C2 C6 \
  --max-turns 16 \
  --no-judge \
  --out results/peer_loop_9b.json

echo "Generation done. On the laptop, with the judge key:"
echo "  uv run python scripts/15_regrade.py --in results/peer_loop_9b.json --out results/peer_loop_9b_judged.json"
echo "  uv run python scripts/13_loop_analysis.py --in results/peer_loop_9b_judged.json"
