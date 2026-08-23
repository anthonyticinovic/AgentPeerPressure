# AgentPeerPressure

MATS 12.0 project. **Thesis:** under peer-endorsement framing, the refusal *signal* and
refusal *behaviour* come apart while the model's internal *harmfulness belief* is preserved.
If a monitor watches the refusal direction, social framing can slip a harmful action past it
without the model ever "deciding" the task is safe.

## State (2026-08-23)

**Read [`docs/STATE.md`](docs/STATE.md) first** — the durable handoff: established results,
frozen parameters, measurement traps, power, open items.

- **Phase 1 (directions) — complete and holding.** Refusal is causally mediated (**Gate B**),
  and the harmfulness belief is a *separate* direction (**Gate B2**).
- **Phase 2 (peer framing) — pipeline correct, results null on the 4B.** Primary contrast
  C2 − C1b = **+1.7pp, p = 0.69** over 115 items / 29 independent scenarios. Three earlier
  Phase-2 results were retracted; see STATE §2.
- **Next:** 9B on Spartan with the full 208-item corpus.

Key finding so far — **three functions, three different vectors**: the direction that
**separates** harmful from benign is not the one that **causes** refusal, and neither is the
one that **holds the harmfulness belief**.

## Pipeline

`--iter` uses the local 4B; no flag uses the 9B eval model.

| # | Script | Produces |
|---|---|---|
| 01 | `extract_directions.py` | `r_ref` diff-of-means + layer/position AUROC sweep |
| 02 | `dual_directions.py` | `r_ref`/`r_harm` at both token positions, Gate B2 |
| 03 | `arditi_selection.py` | 288-candidate grid, selected `(i*, l*)` |
| 04 | `arditi_generation_check.py` | 256-token behavioural confirmation + controls |
| 05 | `build_log.py` | regenerates `artifacts/inspection.html` |
| 05 | `inversion_preflight.py` | template/coefficient checks before the sweep |
| 06 | `inversion_sweep.py` | Zhao reply-inversion, `--stage calibrate` then `sweep` |
| 07 | `inversion_analysis.py` | sign-consistency, content gap, artefact test |
| 08 | `build_writeup.py` | regenerates `artifacts/writeup.html` |
| 09 | `inspect_stimuli.py` | `artifacts/stimuli.html` — every condition rendered |
| 12 | `peer_loop.py` | **Gate P** — multi-turn tool loop over AgentHarm, graded |
| 13 | `loop_analysis.py` | exact McNemar, family-wide Holm, cluster-level tests |
| 14 | `contrast_audit.py` | decomposes a contrast to the criterion responsible |
| 15 | `regrade.py` | re-score stored transcripts with the judge |
| 16 | `grader_split.py` | regenerates the judge-dependence split |
| — | `diag_inversion.py` | 49-template scan at the logit level |

```bash
uv run python scripts/12_peer_loop.py --iter --conditions C0 C1b C2 C6
uv run python scripts/13_loop_analysis.py --in results/peer_loop.json
```

**Grading is separable from generation.** Transcripts are stored, so re-scoring costs API
calls and seconds — never another model run.

`scripts/archive/` holds superseded scripts that produced **retracted** results, kept for
the audit trail and never cited. See its README.

## Layout

- `src/pressure/` — library.
  - **Phase 1:** `arditi.py` (selection), `causal.py` (interventions + controls),
    `directions.py` (diff-of-means, sweep), `inversion.py` (Zhao reply-inversion),
    `hooks.py` (residual capture).
  - **Phase 2:** `boards.py` (condition renderers + stimulus invariants),
    `tools.py` (AgentHarm tool schemas), `loop.py` (multi-turn tool loop),
    `grading.py` (their rubrics), `judge.py` (DeepSeek semantic judge + fallbacks).
  - **Shared:** `config.py`, `data.py` (corpora + leakage assertions), `model.py`,
    `device.py` (MPS/CUDA portability), `grade.py` (legacy substring grader), `plots.py`.
- `vendor/agentharm`, `vendor/zhao` — authors' released code, verbatim. Never reimplement
  from a paper's prose; every deviation here has cost a retraction.
- `boards/` — 10 hand-written peer boards, 4 variants each.
- `results/` — run outputs, gitignored. `tests/` — `pytest`.

## Portability / Spartan

Device logic lives in `device.py` (dtype, attention impl, memory guard) so the same scripts
run on Apple Silicon and A100/H100. The 4B runs locally; the 9B needs the cluster
(`model.py` refuses 9B on MPS by design). **The CUDA path has never executed** — smoke-test
on the cluster before committing to a long job.

## Measurement discipline

Compliance has been mis-measured **eight** times, always by a proxy or a silently-wrong code
path. Four rules now hold:

1. The number that stands is the hand-check.
2. No number is quoted unless committed code reproduces it from a committed data file.
3. Never re-derive a stored verdict in an analysis script — read it.
4. Cold subagent review before trusting any metric. It has caught something every time.

See `docs/STATE.md` §3 for the traps already fixed, which are easy to reintroduce.
