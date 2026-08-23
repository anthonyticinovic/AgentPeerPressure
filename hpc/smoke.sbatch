#!/bin/bash
# Smoke test — one item per grader, one condition, 52 rows. Run this first.
#
# It answers the two things nothing local can: does the CUDA branch of device.py
# work, and what is the real s/item at 9B? Set --time in gate_p.sbatch from the
# rate this prints.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root:
#   mkdir -p logs/slurm && sbatch hpc/smoke.sbatch
#SBATCH --job-name=gatep_smoke
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi

# Exercises the CUDA branch of device.py and the Phase 1 intervention seams, then
# exits non-zero on any failure so the run below never starts on a broken node.
python scripts/17_cluster_preflight.py --phase1

python -u scripts/12_peer_loop.py \
  --one-per-grader \
  --conditions C1b \
  --max-turns 16 \
  --no-judge \
  --out results/smoke_9b.json

echo "Job finished."
