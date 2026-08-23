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

# Repository checkout. Also where the scripts cd to before running Python.
: "${PRESSURE_PROJECT_DIR:=$HOME/AgentPeerPressure}"

# Virtualenv holding the dependencies. hpc/setup_env.sh creates it.
: "${PRESSURE_VENV:=$PRESSURE_PROJECT_DIR/.venv}"

# Environment modules to load, in order.
: "${PRESSURE_MODULES:=GCCcore/11.3.0 Python/3.11.3 CUDA}"

# Weight cache. Must live somewhere compute nodes can read and login nodes can
# write — home is usually both, scratch is usually faster.
: "${HF_HOME:=$PRESSURE_PROJECT_DIR/.hf}"

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
    # Compute nodes have no outbound network. Without this, a cache miss hangs on a
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
