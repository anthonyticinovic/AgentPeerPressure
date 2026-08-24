#!/bin/bash
# Gate A — six conditions x two ablation levels, with direction monitoring.
#
# One level per job. Two environment variables select what this job runs:
#
#   GATE_A_SCOPE=pilot   52 items, one balanced variant per cluster  (~5h)
#   GATE_A_SCOPE=full    the full 208-item corpus                    (~18h)
#   GATE_A_ABLATE=0      no intervention
#   GATE_A_ABLATE=1      ablate r_arditi with ablate_all_components
#
#   mkdir -p logs/slurm
#   sbatch --export=ALL,GATE_A_SCOPE=pilot,GATE_A_ABLATE=0 hpc/gate_a.sbatch
#   sbatch --export=ALL,GATE_A_SCOPE=pilot,GATE_A_ABLATE=1 hpc/gate_a.sbatch
#
# Gate P measured 40.6 s/item without monitoring or ablation hooks; budget 50.
# Resumable: 12_peer_loop checkpoints every 10 rows and skips any row that
# already has a transcript, so a walltime kill or a pre-emption continues where
# it stopped when the identical command is re-submitted. --requeue makes a
# pre-emption do that by itself.
#
# --no-judge because the DeepSeek key never goes to Spartan. Grade locally from
# the stored transcripts with scripts/15_regrade.py; never analyse this file.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root.
#SBATCH --job-name=gate_a
#SBATCH -p gpu-h100
#SBATCH --account=comp90055
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "${PRESSURE_PROJECT_DIR:-/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}/hpc/cluster_env.sh"
pressure_setup_environment

: "${GATE_A_ABLATE:=0}"
: "${GATE_A_SCOPE:=pilot}"

nvidia-smi
python scripts/17_cluster_preflight.py --gate-a

CONDS="C0 C1b C2 C3 C6 C7"
case "${GATE_A_SCOPE}" in
    pilot) SCOPE_ARGS="--sample-per-cluster"; TAG="pilot" ;;
    full)  SCOPE_ARGS="";                     TAG="full"  ;;
    *)     echo "GATE_A_SCOPE must be 'pilot' or 'full', got '${GATE_A_SCOPE}'" >&2; exit 1 ;;
esac
case "${GATE_A_ABLATE}" in
    0) ABL_ARGS="";          LEVEL="base" ;;
    1) ABL_ARGS="--ablate";  LEVEL="abl"  ;;
    *) echo "GATE_A_ABLATE must be '0' or '1', got '${GATE_A_ABLATE}'" >&2; exit 1 ;;
esac
echo "scope=${GATE_A_SCOPE} ablate=${GATE_A_ABLATE} -> results/gate_a_${TAG}_${LEVEL}.json"

# shellcheck disable=SC2086
python -u scripts/12_peer_loop.py --monitor --no-judge ${ABL_ARGS} ${SCOPE_ARGS} \
    --conditions ${CONDS} --out "results/gate_a_${TAG}_${LEVEL}.json"

echo "Generation done. On the laptop, with the judge key:"
echo "  bash hpc/fetch.sh 'gate_a_${TAG}_*'"
echo "  uv run python scripts/15_regrade.py --in results/gate_a_${TAG}_${LEVEL}.json --out results/gate_a_${TAG}_${LEVEL}_judged.json"
