#!/bin/bash
# Pull results back from the cluster. Mirror of sync.sh, which is push-only and
# ships tracked files only — results/*.pt and the Gate A JSON are untracked, so
# nothing comes back without this.
#
#   bash hpc/fetch.sh                 # everything under results/
#   bash hpc/fetch.sh 'gate_a_*'      # a subset

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SPARTAN_HOST:=spartan}"
: "${SPARTAN_DIR:=/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}"

pattern="${1:-*}"
mkdir -p results
rsync -az --prune-empty-dirs \
    --include='*/' --include="${pattern}" --exclude='*' \
    "${SPARTAN_HOST}:${SPARTAN_DIR}/results/" results/
echo "fetched '${pattern}' from ${SPARTAN_HOST}"
