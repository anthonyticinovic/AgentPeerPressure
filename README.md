# AgentPeerPressure

MATS 12.0 project. **Thesis:** under peer-endorsement framing, the refusal *signal* and
refusal *behaviour* come apart while the model's internal *harmfulness belief* is preserved.
If a monitor watches the refusal direction, social framing can slip a harmful action past it
without the model ever "deciding" the task is safe.

## State (2026-08-21)

Phase 1 (directions) is done on the local 4B model. **Gate B passes:** a diff-of-means
direction, selected by Arditi's real criterion, causally ablates refusal
(1.00 → 0.04, 0.84 harmful compliance by hand). Full narrative and numbers:

- **Research log** — `artifacts/inspection.html` (built by `scripts/05_build_log.py`). The
  canonical record: ledger of what is established, dated entries E1–E11, Gate B table.
- **Plan** — `docs/superpowers/plans/2026-08-19-safety-signal-social-framing.md`. Phases,
  gates, and "Findings from Checkpoint N" sections. Checkpoint 5 is the current front.

Key nuance the thesis rests on: the direction that **separates** harmful from benign (the
monitor) is *not* the direction that **ablates** refusal (the mediator) — detector ≠ cause.

## Pipeline

Run in order; `--iter` uses the 4B iteration model, no flag uses the 9B eval model.

| # | Script | Produces |
|---|---|---|
| 01 | `extract_directions.py` | `r_ref` diff-of-means + layer/position AUROC sweep |
| 02 | `dual_directions.py` | `r_ref`/`r_harm` at both token positions, Gate B2 |
| 03 | `arditi_selection.py` | 288-candidate grid, selected `(i*, l*)` by bypass/induce/kl |
| 04 | `arditi_generation_check.py` | 256-token behavioural confirmation + controls |
| 05 | `build_log.py` | regenerates `artifacts/inspection.html` from `results/*.json` |

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

Compliance has been mis-measured three times, each time by a token/prefix proxy (substring
refusal markers; two over-strict judges). **The number that stands is always the hand-check.**
Any new compliance metric must be validated against hand labels before its value is quoted.
