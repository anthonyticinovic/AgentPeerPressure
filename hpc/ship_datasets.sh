#!/bin/bash
# Ship the HuggingFace *dataset* caches from the laptop to the cluster.
#
# fetch_weights.sh deliberately does not fetch datasets: AdvBench and AgentHarm are
# both gate-on-click, so pulling them on the cluster would need an HF token there.
# Shipping the laptop's cache avoids the token and, more importantly, guarantees both
# machines read byte-identical corpora — `harmful_prompts()` falls back to a mirror if
# the canonical AdvBench is unavailable, and a silent fallback on one machine only
# would change which prompts built the directions.
#
# ~45 MB. Run from the repository root on the laptop:
#   bash hpc/ship_datasets.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SPARTAN_HOST:=spartan}"
: "${SPARTAN_HF:=/data/gpfs/projects/COMP90055/aticinovic/hf}"
: "${LOCAL_HF:=${HF_HOME:-$HOME/.cache/huggingface}}"

# Every corpus src/pressure/data.py can load, by its cache directory name.
SETS=(
    walledai___adv_bench                 # harmful_prompts, canonical
    kelly8tom___advbench_orig            # harmful_prompts, mirror fallback
    tatsu-lab___alpaca                   # harmless_prompts
    JailbreakBench___jbb-behaviors       # matched_pairs, selection split
    ai-safety-institute___agent_harm     # the evaluation corpus
)

for s in "${SETS[@]}"; do
    if [ ! -d "${LOCAL_HF}/datasets/${s}" ]; then
        echo "missing locally: ${LOCAL_HF}/datasets/${s}" >&2
        echo "load it once on this machine, then re-run." >&2
        exit 1
    fi
done

ssh "${SPARTAN_HOST}" "mkdir -p ${SPARTAN_HF}/datasets"
for s in "${SETS[@]}"; do
    rsync -az "${LOCAL_HF}/datasets/${s}/" "${SPARTAN_HOST}:${SPARTAN_HF}/datasets/${s}/"
    echo "shipped ${s}"
done
echo "done — ${#SETS[@]} dataset caches at ${SPARTAN_HOST}:${SPARTAN_HF}/datasets"
