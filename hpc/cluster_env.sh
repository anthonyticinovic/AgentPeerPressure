#!/bin/bash
# Cluster-specific settings, sourced by every sbatch script under hpc/.
#
# Override any of them by exporting before submitting:
#
#   PRESSURE_VENV="$HOME/envs/pressure" sbatch hpc/gate_p.sbatch
#
# The #SBATCH headers in each script — partition, GPU type, memory, walltime —
# cannot be read from here, because SLURM parses them before any shell runs.
# Those must be edited per cluster. See hpc/README.md.

# Project storage, not home: home is capped at 50 GB and the two Qwen
# checkpoints alone are ~26 GB.
: "${PRESSURE_BASE:=/data/gpfs/projects/COMP90055/aticinovic}"
: "${PRESSURE_PROJECT_DIR:=$PRESSURE_BASE/AgentPeerPressure}"
: "${PRESSURE_VENV:=$PRESSURE_BASE/venv-pressure}"
: "${HF_HOME:=$PRESSURE_BASE/hf}"

# Anaconda only. There is no Python/3.11.x module on Spartan any more — the one
# the previous project used has been removed, which is why every venv in that home
# directory died with "libpython3.11.so.1.0: cannot open shared object file". The
# venv below is built from this interpreter, so this module must be loaded before
# activating it, and the version must stay pinned.
#
# No CUDA module: the pip torch wheel bundles its own CUDA runtime.
: "${PRESSURE_MODULES:=Anaconda3/2024.02-1}"

pressure_setup_environment() {
    if command -v module >/dev/null 2>&1; then
        module purge
        for mod in ${PRESSURE_MODULES}; do
            module load "${mod}"
        done
    else
        echo "No 'module' command found — assuming the environment is already set up."
    fi

    if [ ! -f "${PRESSURE_VENV}/bin/activate" ]; then
        echo "No virtualenv at ${PRESSURE_VENV}. Run hpc/setup_env.sh on a login node." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${PRESSURE_VENV}/bin/activate"

    if [ ! -d "${PRESSURE_PROJECT_DIR}" ]; then
        echo "No checkout at ${PRESSURE_PROJECT_DIR}. Set PRESSURE_PROJECT_DIR." >&2
        exit 1
    fi
    cd "${PRESSURE_PROJECT_DIR}" || exit 1

    export HF_HOME
    # Compute nodes have no outbound network. Without this a cache miss hangs on a
    # connection timeout instead of failing with the name of the missing file.
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1

    echo "=== Environment ==="
    echo "Project:  ${PWD}"
    echo "Python:   $(command -v python)"
    echo "HF_HOME:  ${HF_HOME}"
    python -c "import torch, transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__} | cuda {torch.cuda.is_available()}')"
    echo "==================="
}
