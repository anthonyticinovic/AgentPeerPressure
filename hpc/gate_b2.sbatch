#!/bin/bash
# Gate B2 at 9B — causal-steering validation of r_harm/r_ref/r_arditi.
#
# Never run at 9B before. Result 1 of the write-up currently shows this table
# at 4B, unlabelled, which two independent adversarial reviewers flagged
# 2026-08-31 (see docs/STATE.md). This closes that gap for real.
#
# One env var controls scope, same pattern as gate_a.sbatch:
#   GATE_B2_N=5    smoke test - a few minutes, validates correctness + gets a
#                  real per-batch timing measurement before committing to the
#                  full run (no historical 9B timing for this script exists)
#   GATE_B2_N=50   the real run (script default, matches the original 4B scale)
#
#   mkdir -p logs/slurm
#   sbatch --export=ALL,GATE_B2_N=5  hpc/gate_b2.sbatch   # smoke test
#   sbatch --export=ALL,GATE_B2_N=50 hpc/gate_b2.sbatch   # full run
#
# No checkpointing in 06_inversion_sweep.py - results are written once, at the
# end of the full sweep. A preemption mid-run loses all progress, unlike
# 12_peer_loop.py's per-10-row checkpointing. Submit the smoke test to whatever
# partition is fastest (a lost smoke test costs nothing); submit the full run
# to a non-preemptible partition once its real timing is known from the smoke
# test, not to a preemptible one - see docs/STATE.md for the partition sweep
# behind this choice.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root.
#SBATCH --job-name=gate_b2
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

: "${GATE_B2_N:=50}"

nvidia-smi
python scripts/17_cluster_preflight.py

echo "=== inversion template preflight (never run at 9B - results/inversion_preflight.json didn't exist) ==="
python -u scripts/05_inversion_preflight.py

echo "=== calibrate (9B, fresh - confirming or overturning the frozen coeff=2.0) ==="
time python -u scripts/06_inversion_sweep.py --stage calibrate

echo "=== sweep (n=${GATE_B2_N} held-out per class) ==="
time python -u scripts/06_inversion_sweep.py --stage sweep --n "${GATE_B2_N}"

echo "=== analysis ==="
python -u scripts/07_inversion_analysis.py

echo "Gate B2 (9B, n=${GATE_B2_N}) done. On the laptop: bash hpc/fetch.sh"
