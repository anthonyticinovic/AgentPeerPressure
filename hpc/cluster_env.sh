#!/bin/bash
# Cluster-specific settings, sourced by every sbatch script under hpc/.
#
# Override any of them by exporting before submitting:
#
#   PRESSURE_BASE=/some/other/path sbatch hpc/gate_p.sbatch
#
# The #SBATCH headers in each script — partition, GPU type, memory, walltime —
# cannot be read from here, because SLURM parses them before any shell runs.
# Those must be edited per cluster. See hpc/README.md.

# Project storage, not home: home is capped at 50 GB and the 9B checkpoint alone
# is ~18 GB.
: "${PRESSURE_BASE:=/data/gpfs/projects/COMP90055/aticinovic}"
: "${PRESSURE_PROJECT_DIR:=$PRESSURE_BASE/AgentPeerPressure}"
: "${HF_HOME:=$PRESSURE_BASE/hf}"

# uv owns the toolchain, and no environment module is loaded.
#
# Spartan's Anaconda base is Python 3.11.7 and pyproject requires >=3.12, so the
# system interpreter cannot build this project at all. uv downloads a standalone
# CPython 3.12 and reproduces uv.lock exactly, giving the cluster the same
# dependency set as the laptop. It is also self-contained: the previous project's
# venvs all died with "libpython3.11.so.1.0: cannot open shared object file" when
# the module they were built against was withdrawn. Nothing here can be withdrawn.
: "${UV_INSTALL_DIR:=$PRESSURE_BASE/uv}"
# only-managed is load-bearing. Left to its own devices uv picked the login node's
# /usr/bin/python3.12, which does not exist on the compute nodes: .venv/bin/python
# became a dangling symlink, `source activate` still succeeded because it only edits
# PATH, and bash skipped the unexecutable link and silently fell through to
# Anaconda's 3.11.7. A managed interpreter lives in project storage, which every
# node mounts.
: "${UV_PYTHON_PREFERENCE:=only-managed}"
: "${UV_PYTHON:=3.12}"

# Spartan's driver is 570.211.01 / CUDA 12.8. PyPI's torch 2.13.0 is built against
# CUDA 13.0 and refuses to initialise on it: torch.cuda.is_available() returns
# False and everything silently runs on CPU. cu126 is the newest build of the same
# torch version the lock pins, and a 12.6 build runs on any 12.x driver >= 12.6, so
# only the CUDA runtime differs from the laptop — not the torch version.
: "${PRESSURE_TORCH_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${PRESSURE_TORCH_SPEC:=torch==2.13.0+cu126}"
: "${UV_CACHE_DIR:=$PRESSURE_BASE/uv-cache}"
: "${UV_PYTHON_INSTALL_DIR:=$PRESSURE_BASE/uv-python}"
: "${PRESSURE_VENV:=$PRESSURE_PROJECT_DIR/.venv}"

pressure_setup_environment() {
    export UV_INSTALL_DIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_PREFERENCE UV_PYTHON
    export PATH="${UV_INSTALL_DIR}:${PATH}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found at ${UV_INSTALL_DIR}. Run hpc/setup_env.sh on a login node." >&2
        exit 1
    fi
    if [ ! -f "${PRESSURE_VENV}/bin/activate" ]; then
        echo "No virtualenv at ${PRESSURE_VENV}. Run hpc/setup_env.sh on a login node." >&2
        exit 1
    fi
    if [ ! -d "${PRESSURE_PROJECT_DIR}" ]; then
        echo "No checkout at ${PRESSURE_PROJECT_DIR}. Set PRESSURE_PROJECT_DIR." >&2
        exit 1
    fi
    cd "${PRESSURE_PROJECT_DIR}" || exit 1
    # shellcheck disable=SC1091
    source "${PRESSURE_VENV}/bin/activate"

    export HF_HOME
    # Compute nodes have no outbound network. Without these a cache miss hangs on a
    # connection timeout instead of failing with the name of the missing file.
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1

    # Verify the interpreter is the venv's, not something PATH fell through to. A
    # dangling venv symlink is invisible to `source activate`, and the job that hid
    # it reported COMPLETED with exit 0 while every Python step failed.
    if [ "$(command -v python)" != "${PRESSURE_VENV}/bin/python" ]; then
        echo "python resolved to $(command -v python), not ${PRESSURE_VENV}/bin/python." >&2
        echo "The venv is broken — rebuild it with hpc/setup_env.sh." >&2
        exit 1
    fi
    if ! python -c "import torch" 2>/dev/null; then
        echo "torch does not import in ${PRESSURE_VENV}. Rebuild with hpc/setup_env.sh." >&2
        exit 1
    fi

    echo "=== Environment ==="
    echo "Project:  ${PWD}"
    echo "Python:   $(command -v python) $(python -V 2>&1)"
    echo "HF_HOME:  ${HF_HOME}"
    python -c "import torch, transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__} | cuda {torch.cuda.is_available()}')"
    echo "==================="
}
