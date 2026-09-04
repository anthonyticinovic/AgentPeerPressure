#!/bin/bash
# G3 -- content-arm full run, base arm (refusal intact). See docs/STATE.md.
# 208 items x 4 conditions = 832 rows. Gated on G2's pilot showing a visible
# C8-C9 gap (it did: +9.6pp, 6/1 discordant, floor-pinned at pilot scale -- see
# docs/STATE.md's adversarial review entry, 2026-09-02) and submitted alongside
# G4 (--ablate) in parallel, per the plan's design.
#
# --max-new-tokens 1536, not the pilot's 768: matches gate_a_full_base_judged.json
# (the source of the C2/C1b secondary contrasts), fixing the token-budget confound
# the adversarial review flagged (cut_mid_call 0/52 on C2/C1b vs 3/52 and 2/52 on
# C8/C8b at 768).
#
#   mkdir -p logs/slurm && sbatch hpc/g3_content_full_base.sbatch
#SBATCH --job-name=g3_base
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
  --no-judge \
  --out results/content_arm_full_base.json

echo "Generation done. Fetch and grade on the laptop:"
echo "  bash hpc/fetch.sh 'content_arm_full_base*'"
echo "  uv run python scripts/15_regrade.py --in results/content_arm_full_base.json --out results/content_arm_full_base_judged.json"
