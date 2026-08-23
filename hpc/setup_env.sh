#!/bin/bash
# One-time environment build, on a LOGIN node — it needs outbound network, which
# compute nodes do not have. Weights are a separate, slower step: hpc/fetch_weights.sh
#
#   bash hpc/setup_env.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"
export UV_INSTALL_DIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
mkdir -p "${UV_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}"
cd "${PRESSURE_PROJECT_DIR}"

if ! command -v "${UV_INSTALL_DIR}/uv" >/dev/null 2>&1; then
    echo "Installing uv into ${UV_INSTALL_DIR}"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${UV_INSTALL_DIR}" sh
fi
export PATH="${UV_INSTALL_DIR}:${PATH}"
export UV_PYTHON_PREFERENCE UV_PYTHON
uv --version

# Install uv's own CPython into project storage. Without this uv reaches for the
# login node's /usr/bin/python3.12, which compute nodes do not have.
uv python install "${UV_PYTHON}"
uv python find "${UV_PYTHON}"

# Rebuild rather than reuse: a venv bound to the wrong interpreter cannot be
# repaired in place.
rm -rf "${PRESSURE_VENV}"

# --frozen resolves nothing: it installs exactly what uv.lock pins, so the cluster
# runs the same dependency set as the laptop rather than a fresh resolution.
echo "Syncing dependencies from uv.lock"
uv sync --frozen --python "${UV_PYTHON}"

# flash-attn must NOT be present: device.py prefers it when installed, while the
# 4B run used sdpa. That would confound model size with attention kernel.
uv pip uninstall flash-attn flash_attn 2>/dev/null || true

# shellcheck disable=SC1091
source "${PRESSURE_VENV}/bin/activate"
echo
echo "=== Versions ==="
echo "interpreter: $(readlink -f "${PRESSURE_VENV}/bin/python")"
python -V
python -c "import torch, transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__}')"
echo "(cuda is False on a login node — no GPU here. hpc/gpu_check.sbatch tests that.)"
echo
echo "Next: bash hpc/fetch_weights.sh"
