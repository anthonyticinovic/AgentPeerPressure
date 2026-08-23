#!/bin/bash
# One-time environment build, on a LOGIN node — it needs outbound network, which
# compute nodes do not have. Fast (a few minutes). Weights are a separate step:
# see hpc/fetch_weights.sh, which is slow and can run in the background.
#
#   bash hpc/setup_env.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

if command -v module >/dev/null 2>&1; then
    module purge
    for mod in ${PRESSURE_MODULES}; do module load "${mod}"; done
fi
cd "${PRESSURE_PROJECT_DIR}"

echo "Building venv at ${PRESSURE_VENV} from $(command -v python)"
python -m venv "${PRESSURE_VENV}"
# shellcheck disable=SC1091
source "${PRESSURE_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .

# flash-attn must NOT be installed: device.py prefers it when present, while the
# 4B run used sdpa. That would confound model size with attention kernel.
python -m pip uninstall -y flash_attn flash-attn 2>/dev/null || true

echo
echo "=== Versions ==="
python -c "import torch, transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__}')"
echo "(cuda is False here — login nodes have no GPU. hpc/gpu_check.sbatch tests that.)"
echo
echo "=== Environment checks that do not need a GPU ==="
python scripts/17_cluster_preflight.py --skip-model || true
echo
echo "Next: bash hpc/fetch_weights.sh   (~26 GB, slow)"
