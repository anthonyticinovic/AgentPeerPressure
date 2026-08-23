#!/bin/bash
# Download the Qwen checkpoints into HF_HOME. LOGIN node only — compute nodes have
# no outbound network, and the job sets HF_HUB_OFFLINE=1 so a missing file fails
# loudly rather than hanging on a connection timeout.
#
# ~26 GB for both. Run under nohup and watch the log:
#   nohup bash hpc/fetch_weights.sh > logs/fetch_weights.log 2>&1 &

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

export PATH="${UV_INSTALL_DIR}:${PATH}"
# shellcheck disable=SC1091
source "${PRESSURE_VENV}/bin/activate"
cd "${PRESSURE_PROJECT_DIR}"

export HF_HOME
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

# The 4B is included so the smoke test has a cheap fallback if the 9B misbehaves.
# Datasets are NOT fetched here: AgentHarm is gated, so its cache is shipped from
# the laptop instead of requiring a token on the cluster.
for repo in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"):
    print(f"--- fetching {repo}", flush=True)
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    total = sum(f.stat().st_size for f in __import__("pathlib").Path(p).rglob("*") if f.is_file())
    print(f"--- {repo}: {total / 2**30:.1f} GiB at {p}", flush=True)
PY
echo "done"
