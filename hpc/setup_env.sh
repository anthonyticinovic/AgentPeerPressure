#!/bin/bash
# One-time setup, on a LOGIN node — it needs outbound network, which compute
# nodes do not have. Creates the virtualenv and pre-fetches the weights.
#
#   bash hpc/setup_env.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

if command -v module >/dev/null 2>&1; then
    module purge
    for mod in ${PRESSURE_MODULES}; do module load "${mod}"; done
fi
cd "${PRESSURE_PROJECT_DIR}"

python -m venv "${PRESSURE_VENV}"
# shellcheck disable=SC1091
source "${PRESSURE_VENV}/bin/activate"
pip install --upgrade pip
pip install -e .

# flash-attn must NOT be installed: device.py would then select flash_attention_2
# while the 4B run used sdpa, confounding model size with attention kernel.
pip uninstall -y flash_attn flash-attn 2>/dev/null || true

export HF_HOME
mkdir -p "${HF_HOME}"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"):
    print(f"fetching {repo}")
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
PY

echo
echo "Verifying the environment checks that do not need a GPU:"
python scripts/17_cluster_preflight.py --skip-model || true
echo
echo "Done. Submit with: sbatch hpc/smoke.sbatch"
