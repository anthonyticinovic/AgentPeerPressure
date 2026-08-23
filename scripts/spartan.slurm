#!/bin/bash
# Gate P on Spartan: Qwen3.5-9B, 208 items x 4 conditions.
#
# Grading runs WITHOUT the semantic judge here, because compute nodes have no
# outbound network. That is not a compromise: transcripts are stored, so the judge
# is applied afterwards on the laptop with scripts/15_regrade.py for a few cents.
# Never analyse the file this job writes — analyse the regraded one.
#
# The job is resumable. 12_peer_loop checkpoints every 10 rows and skips any row
# that already has a transcript, so re-submitting the identical command continues
# where it stopped. --requeue below makes a pre-emption restart do that itself.
#
# CONFIRM BEFORE FIRST SUBMISSION (marked <<<):
#   - partition and account names
#   - that the GPU is A100/H100, not V100: sm70 has no bf16 and the preflight fails
#   - the walltime, once the smoke test reports s/item

#SBATCH --job-name=gatep-9b
#SBATCH --account=punim0000              # <<< your project code
#SBATCH --partition=gpu-a100             # <<< confirm on Spartan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # <<< set from the smoke test
#SBATCH --requeue
#SBATCH --output=logs/gatep-%j.out
#SBATCH --error=logs/gatep-%j.err

set -euo pipefail

REPO="${REPO:-$HOME/AgentPeerPressure}"
cd "$REPO"
mkdir -p logs results

# Weights must already be in this cache — download them from a login node, which
# has network, before submitting. HF_HUB_OFFLINE turns a silent network hang into
# an immediate, legible error.
export HF_HOME="${HF_HOME:-/data/scratch/projects/$SLURM_JOB_ACCOUNT/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

module purge
module load GCCcore/11.3.0 Python/3.11.3 CUDA/12.2.0   # <<< confirm module names
source .venv/bin/activate

nvidia-smi
python scripts/17_cluster_preflight.py                  # exits non-zero on any failure

python scripts/12_peer_loop.py \
  --conditions C0 C1b C2 C6 \
  --max-turns 16 \
  --no-judge \
  --out results/peer_loop_9b.json

echo "generation done — copy results/peer_loop_9b.json to the laptop and run:"
echo "  uv run python scripts/15_regrade.py --in results/peer_loop_9b.json --out results/peer_loop_9b_judged.json"
