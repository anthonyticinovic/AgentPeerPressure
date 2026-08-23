# SLURM job scripts

Gate P on Spartan: Qwen3.5-9B against the full AgentHarm harmful corpus, four
conditions. Written for A100/H100 nodes — **treat the headers as a starting
point, not a drop-in**.

## Running

```bash
bash hpc/setup_env.sh                 # login node, once — venv + weights
mkdir -p logs/slurm
sbatch hpc/smoke.sbatch               # 52 rows, one per grader. Run this first.
sbatch hpc/gate_p.sbatch              # 832 rows, the real run
```

Logs land in `logs/slurm/<job-name>_<job-id>.{out,err}`.

## What to adapt

### 1. Paths and environment — `hpc/cluster_env.sh`

Every script sources this file. Edit the defaults there, or export overrides at
submission time:

| Variable | Default | What it is |
|---|---|---|
| `PRESSURE_PROJECT_DIR` | `$HOME/AgentPeerPressure` | Repository checkout |
| `PRESSURE_VENV` | `$PRESSURE_PROJECT_DIR/.venv` | Virtualenv, created by `setup_env.sh` |
| `PRESSURE_MODULES` | `GCCcore/11.3.0 Python/3.11.3 CUDA` | Environment modules, loaded in order |
| `HF_HOME` | `$PRESSURE_PROJECT_DIR/.hf` | Weight cache, populated on a login node |

### 2. Scheduler directives — the `#SBATCH` header in each script

Partition name, `--gres`, `--mem` and `--time` are cluster-specific and **cannot**
be moved into `cluster_env.sh`: SLURM parses the header before any shell runs.
Edit them in place.

Set `gate_p.sbatch`'s `--time` from the s/item the smoke test prints, times 832,
plus half again.

## Why the job looks like this

- **The GPU must be A100 or H100.** Below compute capability 8.0 there is no
  native bf16, `device.py` falls back to fp16, and the 9B numbers would differ
  from the 4B run by precision as well as scale. The preflight fails hard on it.
- **flash-attn must not be installed.** `device.py` prefers it when present, and
  the 4B run used sdpa. `setup_env.sh` uninstalls it; the preflight checks.
- **The judge is off on the node.** Compute nodes have no outbound network.
  Transcripts are stored, so grading is a separate local step costing cents —
  see the closing lines of `gate_p.sbatch`. Analysing the file the job writes
  directly would report judge-disabled numbers under a judged label.
- **`HF_HUB_OFFLINE=1`.** Without it a cache miss hangs on a connection timeout
  instead of naming the missing file.
- **One job at a time, no arrays.** Tool execution `chdir`s into `vendor/agentharm`
  and several vendored tools write fixture files there; two jobs sharing the
  checkout would race.
- **Preflight gates every run.** `scripts/17_cluster_preflight.py` exercises the
  CUDA branch of `device.py` — which has never executed — and, with `--phase1`,
  the four seams where a CPU-held direction meets an on-device activation.
