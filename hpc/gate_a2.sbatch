#!/bin/bash
# Gate A2 — does the model still work under ablation?
#
# Arditi's kl < 0.1 filter is fitted on the next-token distribution. A 16-turn
# tool loop is a far longer horizon. If ablation breaks tool-call formatting,
# compliance falls for a reason unrelated to framing and the whole run is void.
#
# Both arms run with --monitor, so any difference between them is attributable
# to the ablation alone. Monitoring only the ablated arm would confound the two.
#
# Compare capability, not compliance: malformed tool_call blocks, cut_mid_call
# and mean turn count. any_call is confounded with the treatment effect.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root:
#   mkdir -p logs/slurm && sbatch hpc/gate_a2.sbatch
#SBATCH --job-name=gate_a2
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

nvidia-smi
python scripts/17_cluster_preflight.py --gate-a

python -u scripts/12_peer_loop.py --one-per-grader --conditions C1b --no-judge \
    --monitor --out results/gate_a2_control.json
python -u scripts/12_peer_loop.py --one-per-grader --conditions C1b --no-judge \
    --monitor --ablate --out results/gate_a2_ablated.json

echo "Gate A2 done. On the laptop:"
echo "  bash hpc/fetch.sh 'gate_a2_*'"
echo "Compare malformed turns, cut_mid_call and mean_turns across the two arms,"
echo "and record the maximum relative p_harm drift — Task 11 uses it as the noise bound."
