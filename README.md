# AgentPeerPressure

MATS 12.0 project. **Thesis:** under peer-endorsement framing, the refusal *signal* and
refusal *behaviour* come apart while the model's internal *harmfulness belief* is preserved.
If a monitor watches the refusal direction, social framing can slip a harmful action past it
without the model ever "deciding" the task is safe.

## State (2026-08-22)

**Read [`docs/STATE.md`](docs/STATE.md) first** — it is the durable handoff: established
results, frozen parameters, measurement traps, and what comes next.

Phase 1 (directions) is complete on the local 4B. Both preconditions hold: refusal is
causally mediated (**Gate B**), and the harmfulness belief is a *separate* direction
(**Gate B2**). The main peer-framing experiment has not started.

- **Research log** — `artifacts/inspection.html` (`scripts/05_build_log.py`). The working
  record: every decision, dead end and retraction, in order.
- **Write-up** — `artifacts/writeup.html` (`scripts/08_build_writeup.py`). Findings only,
  for an interp-literate reader. Implementation history stays in the log.
- **Plan** — `docs/superpowers/plans/2026-08-19-safety-signal-social-framing.md`. Phases,
  gates, and "Findings from Checkpoint N" sections. Checkpoint 6 is the current front.

Key finding so far — **three functions, three different vectors**: the direction that
**separates** harmful from benign is not the one that **causes** refusal, and neither is
the one that **holds the harmfulness belief**. Which of the three a monitor tracks
determines what a manipulation would have to defeat.

## Pipeline

Run in order; `--iter` uses the 4B iteration model, no flag uses the 9B eval model.

| # | Script | Produces |
|---|---|---|
| 01 | `extract_directions.py` | `r_ref` diff-of-means + layer/position AUROC sweep |
| 02 | `dual_directions.py` | `r_ref`/`r_harm` at both token positions, Gate B2 |
| 03 | `arditi_selection.py` | 288-candidate grid, selected `(i*, l*)` by bypass/induce/kl |
| 04 | `arditi_generation_check.py` | 256-token behavioural confirmation + controls |
| 05 | `build_log.py` | regenerates `artifacts/inspection.html` |
| 06 | `inversion_sweep.py` | Zhao reply-inversion, `--stage calibrate` then `sweep` |
| 07 | `inversion_analysis.py` | sign-consistency, content gap, artefact test |
| 08 | `build_writeup.py` | regenerates `artifacts/writeup.html` |

```bash
uv run python scripts/03_arditi_selection.py --iter
```

`scripts/archive/` holds superseded scripts that produced retracted results — kept for the
audit trail, not to be cited. See `scripts/archive/README.md`.

## Layout

- `src/pressure/` — library. `arditi.py` (selection), `causal.py` (interventions + controls),
  `directions.py` (diff-of-means, sweep, freeze), `data.py` (three-role corpora),
  `hooks.py` (residual capture), `device.py` (MPS/CUDA portability), `grade.py` (compliance).
- `results/` — run outputs, gitignored. `_superseded/` holds pruned detour outputs.
- `tests/` — `pytest`.

## Portability / Spartan

All device logic is in `device.py` (dtype, attention impl, memory guard) so the same scripts
run on Apple Silicon (MPS) and A100/H100 (CUDA). The 4B run is local; the 9B eval run is the
first step that needs the cluster (`model.py` refuses 9B on MPS by design). No code change
should be needed — set `PRESSURE_EVAL_MODEL` / `PRESSURE_DEVICE` and run without `--iter`.

## Measurement discipline

Compliance and judgement have been mis-measured **five** times, every time by a proxy:
substring refusal markers; an over-strict judge; a self-invented prefix objective; Chinese
refusals scored unparsed; and `"Yes, …"` verdicts scored unparsed (723 replies).
**The number that stands is always the hand-check.** Any new metric must be validated
against blind labels before its value is quoted — see `docs/STATE.md` §3 for the analysis
traps already fixed, which are easy to reintroduce.
