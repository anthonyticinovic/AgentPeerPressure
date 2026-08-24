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

Logs land in `logs/slurm/<job-name>_<job-id>.{out,err}`. Create that directory
first — SLURM does not, and a job whose `--output` path is unwritable fails at
launch with nothing to read.

`sync.sh` pushes the repo to the cluster and is tracked-files-only, so `.env` and
the judge key never leave this machine. `fetch.sh` is the other direction: run
outputs are untracked, so nothing comes back without it.

```bash
bash hpc/sync.sh                      # push
bash hpc/fetch.sh                     # pull everything under results/
bash hpc/fetch.sh 'gate_a_*'          # pull a subset
```

Both read `SPARTAN_HOST` (default `spartan`) and `SPARTAN_DIR`.

## Gate A — the ablation runs

Five jobs in order. Each one gates the next, and two of them can stop the
experiment before the expensive job is submitted.

| # | job | what it settles |
|---|---|---|
| 1 | `gate_a1.sbatch` | Rebuild `r_arditi`, `r_harm`, `r_ref` at 9B. Gate B and B2 are 4B results. |
| 2 | `gate_a2.sbatch` | Does the model still call tools under ablation? Also fixes the `p_harm` noise bound. |
| 3 | identity check | Does the monitor's extra forward pass perturb greedy generation? Must be all-identical. |
| 4 | `gate_a.sbatch` pilot | 52 balanced items, both levels. Decision gate. |
| 5 | `gate_a.sbatch` full | 208 items, both levels. The confirmatory run. |

```bash
bash hpc/sync.sh
mkdir -p logs/slurm
sbatch hpc/gate_a1.sbatch             # ~10h. Read its four stop conditions before continuing.
sbatch hpc/gate_a2.sbatch             # ~3h. Compare capability, not compliance.
```

Step 3 is an `srun` one-liner rather than a job — see Task 9 of the plan. It
re-runs eight Gate P items with `--monitor` and diffs the transcripts against
`results/peer_loop_9b.json`. Decoding is greedy, so anything short of identical
means a hook was left registered. Stop and fix.

The pilot and the confirmatory run share one script. `GATE_A_SCOPE` picks the
corpus, `GATE_A_ABLATE` picks the level, and one job runs one level:

```bash
sbatch --export=ALL,GATE_A_SCOPE=pilot,GATE_A_ABLATE=0 hpc/gate_a.sbatch
sbatch --export=ALL,GATE_A_SCOPE=pilot,GATE_A_ABLATE=1 hpc/gate_a.sbatch
# read the pilot, then:
sbatch --export=ALL,GATE_A_SCOPE=full,GATE_A_ABLATE=0 hpc/gate_a.sbatch
sbatch --export=ALL,GATE_A_SCOPE=full,GATE_A_ABLATE=1 hpc/gate_a.sbatch
```

The leading `ALL` is load-bearing. Naming any variable in `--export` replaces
SLURM's default of exporting the whole submitting environment, so
`--export=GATE_A_SCOPE=full` alone would drop `PRESSURE_PROJECT_DIR` and every
other override, and the job would silently run on the defaults. Output goes to
`results/gate_a_<scope>_<level>.json`.

Both scopes fit inside one `--time=20:00:00`, the full corpus with about 15%
headroom at the budgeted 50 s/item. That is thinner than `gate_p.sbatch`'s
margin and is deliberate: the run checkpoints every 10 rows and skips rows that
already have a transcript, so re-submitting the identical command after a
walltime kill resumes rather than restarts.

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
