#!/bin/bash
# Ship the repo to the cluster. Tracked files only, so .env and its API key never
# leave this machine, and the git sha travels in .git_sha since .git does not.
#
#   bash hpc/sync.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SPARTAN_HOST:=spartan}"
: "${SPARTAN_DIR:=/data/gpfs/projects/COMP90055/aticinovic/AgentPeerPressure}"

git rev-parse --short HEAD > .git_sha
trap 'rm -f .git_sha .filelist' EXIT
{ git ls-files; echo .git_sha; } > .filelist

rsync -az --files-from=.filelist ./ "${SPARTAN_HOST}:${SPARTAN_DIR}/"
echo "synced $(cat .git_sha) to ${SPARTAN_HOST}:${SPARTAN_DIR}"
