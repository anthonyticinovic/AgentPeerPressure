#!/bin/bash
# Gate A1 — rebuild all three directions at 9B.
#
# Gate B (i* = -7, l* = 12) and Gate B2 are 4B results. Nothing in Gate A works
# until r_arditi, r_harm and r_ref exist at the eval scale.
#
# The trailing preflight is the gate, not a report. It fails the job if the
# directions just built do not load, disagree on model, leave a residual writer
# outside embed_tokens/attn/mlp, or overlap r_arditi too far to be monitorable.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root:
#   mkdir -p logs/slurm && sbatch hpc/gate_a1.sbatch
#SBATCH --job-name=gate_a1
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi
python scripts/17_cluster_preflight.py

python -u scripts/02_dual_directions.py
python -u scripts/03_arditi_selection.py
python -u scripts/04_arditi_generation_check.py

# The gate. Reads the artefacts the three steps above just wrote.
python scripts/17_cluster_preflight.py --gate-a

echo "Gate A1 done. On the laptop:"
echo "  bash hpc/fetch.sh"
echo "Then read the four stop conditions in the plan before submitting gate_a2."
