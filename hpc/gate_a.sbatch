#!/bin/bash
# Gate A — six conditions x two ablation levels, with direction monitoring.
#
# One level per job. Two environment variables select what this job runs:
#
#   GATE_A_SCOPE=pilot   52 items, one balanced variant per cluster  (~5h)
#   GATE_A_SCOPE=full    the full 208-item corpus              (~20h/arm at 768 tok;
#                        see the #SBATCH --time block below for the 1536-token estimate)
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
#SBATCH --time=40:00:00
# Bumped 36h -> 40h 2026-08-28, for two independent reasons layered on top of each
# other: (1) real measured rates from the run that actually executed (not an estimate)
# -- 34.7s/row base, 57.93s/row ablated at 768 tokens -- extrapolate the NINE-condition
# scope (C0/C1/C1b/C2/C3/C4/C5/C6/C7, added for the falsifiability/positive-control gap
# a round-2 adversarial review flagged) to ~18.4h base / ~30.6h ablated including the
# 1536-token bump, leaving only ~5.4h (15%) margin on the ablated arm at the old 36h
# cap. (2) gpu-h100 had genuine spare capacity and no account-level allocation cap when
# checked (`sinfo`/`sacctmgr`, 2026-08-28), so widening the cap costs nothing if the job
# finishes early -- the job is resumable regardless (`12_peer_loop.py` checkpoints
# every 10 rows).
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

# C1 (length-matched neutral filler) and C4/C5 (word-shuffled board / bare
# multi-agent identity, no content) added 2026-08-28 on top of the original six --
# C1 closes the reversed-direction falsifiability gap (no way to tell a real framing
# reversal from "any extra board text moves compliance"); C4/C5 are the only check
# that the instrument responds to non-peer-specific framing at all. All three are
# already implemented in boards.py/CONDITIONS and rendered/validated by
# 09_inspect_stimuli.py; this is the first generation run that includes them.
CONDS="C0 C1 C1b C2 C3 C4 C5 C6 C7"
case "${GATE_A_SCOPE}" in
    # Pilot's cut_mid_call rate grew under ablation (3/52 base -> 5/52 abl items touched,
    # 19/312 rows at the 768-token default) and the confirmatory run is 4x the items, so
    # the confirmatory scope gets a larger budget. Not applied to pilot: it already ran,
    # and its own base/ablated arms only need to be comparable to each other, which they
    # are. base vs ablated within the confirmatory run share this same larger budget, so
    # that comparison is unaffected either.
    pilot) SCOPE_ARGS="--sample-per-cluster"; TAG="pilot"; TOK_ARGS="" ;;
    full)  SCOPE_ARGS="";                     TAG="full";  TOK_ARGS="--max-new-tokens 1536" ;;
    *)     echo "GATE_A_SCOPE must be 'pilot' or 'full', got '${GATE_A_SCOPE}'" >&2; exit 1 ;;
esac
case "${GATE_A_ABLATE}" in
    0) ABL_ARGS="";          LEVEL="base" ;;
    1) ABL_ARGS="--ablate";  LEVEL="abl"  ;;
    *) echo "GATE_A_ABLATE must be '0' or '1', got '${GATE_A_ABLATE}'" >&2; exit 1 ;;
esac
# Defense in depth, not a substitute for verifying `hpc/sync.sh` actually shipped this
# file before submitting (a stale synced copy predating this check would not run it
# either) -- but catches a future edit to the case block above that forgets to keep
# scope=full and the token bump in sync with each other.
if [ "${GATE_A_SCOPE}" = "full" ] && [ "${TOK_ARGS}" != "--max-new-tokens 1536" ]; then
    echo "GATE_A_SCOPE=full must run with --max-new-tokens 1536, got TOK_ARGS='${TOK_ARGS}'" >&2
    echo "-- refusing to generate at the old (768) budget. Fix the case block above." >&2
    exit 1
fi
echo "scope=${GATE_A_SCOPE} ablate=${GATE_A_ABLATE} conds=[${CONDS}] tok_args=[${TOK_ARGS}] scope_args=[${SCOPE_ARGS}] -> results/gate_a_${TAG}_${LEVEL}.json"

# shellcheck disable=SC2086
python -u scripts/12_peer_loop.py --monitor --no-judge ${ABL_ARGS} ${SCOPE_ARGS} ${TOK_ARGS} \
    --conditions ${CONDS} --out "results/gate_a_${TAG}_${LEVEL}.json"

echo "Generation done. On the laptop, with the judge key:"
echo "  bash hpc/fetch.sh 'gate_a_${TAG}_*'"
echo "  uv run python scripts/15_regrade.py --in results/gate_a_${TAG}_${LEVEL}.json --out results/gate_a_${TAG}_${LEVEL}_judged.json"
