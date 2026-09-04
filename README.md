# AgentPeerPressure

MATS 12.0 project. **Thesis:** under peer-endorsement framing, the refusal *signal* and
refusal *behaviour* come apart while the model's internal *harmfulness belief* is preserved.
If a monitor watches the refusal direction, social framing can slip a harmful action past it
without the model ever "deciding" the task is safe.

## Status (2026-09-05)

**Read [`docs/writeup.md`](docs/writeup.md) first** — the narrative: what was asked, what
was measured, what it means. **Read [`docs/STATE.md`](docs/STATE.md) for everything else** —
the working record: every wrong turn, every retraction, the full parameter trail. The two
are deliberately separate; corrections stay out of the write-up by design.

All five results are complete and independently checked against source data:

1. **Three separable directions, not one.** Ablating `r_arditi` collapses refusal
   (1.00 → 0.04) without moving `r_harm`; causal steering confirms only `r_harm` tracks
   ground-truth harmfulness.
2. **With refusal intact, peer identity alone does nothing** — across the full 208-item,
   9-condition AgentHarm corpus.
3. **With refusal ablated, the pre-registered primary interaction is nominally
   significant** (+5.77pp, p=0.0408, power 26%).
4. **A monitoring blind spot exists once refusal is ablated** — 10.5% of complying rows
   still internally flag the request as harmful (p<0.0001, exploratory).
5. **A follow-on condition (board asserts a verdict on the specific item, not just peer
   identity) replicates the pattern at full scale** (+6.9pp, p=0.027 cluster), confirmed
   through three rounds of adversarial review.

Draft is ready for a final authorial pass, not yet the submission version — see the banner
at the top of `docs/writeup.md`.

## Pipeline

`--iter` uses the local 4B; no flag uses the 9B eval model.

| # | Script | Produces |
|---|---|---|
| 01 | `extract_directions.py` | `r_ref` diff-of-means + layer/position AUROC sweep |
| 02 | `dual_directions.py` | `r_ref`/`r_harm` at both token positions, Gate B2 |
| 03 | `arditi_selection.py` | 288-candidate grid, selected `(i*, l*)` for `r_arditi` |
| 04 | `arditi_generation_check.py` | behavioural confirmation of the selected direction |
| 05 | `inversion_preflight.py` | can the model do reply-inversion at all, and with which template |
| 06 | `inversion_sweep.py` | Zhao reply-inversion sweep — does `r_harm` reverse judgement |
| 07 | `inversion_analysis.py` | sign-consistency, content gap, artefact test |
| 12 | `peer_loop.py` | **Gate P** — multi-turn tool loop over AgentHarm, graded |
| 13 | `loop_analysis.py` | exact McNemar, family-wide Holm, cluster-level tests |
| 14 | `contrast_audit.py` | does the metric manufacture contrasts |
| 15 | `regrade.py` | re-score stored transcripts with the judge |
| 16 | `grader_split.py` | regenerates the judge-dependence split |
| 17 | `cluster_preflight.py` | everything that can only fail on the cluster, checked cheaply |
| 18 | `power.py` | what effect size this design actually excludes |
| 19 | `ablation_analysis.py` | **Gate A** — did ablation restore dynamic range, does framing matter now |
| 20 | `grade_ablation.py` | grade ablation completions with the judge, full reply |
| 21 | `interaction_power.py` | power for the actual interaction test |
| 22 | `turn1_lockin.py` | is compliance already decided at turn 1, before framing bites |
| 23 | `calibrate_tau_orth.py` | frozen threshold on `p_harm_orth`, real-ablation calibrated |
| 24 | `blindspot_crosstab.py` | the monitoring blind-spot cross-tab (Result 4) |
| 25 | `merge_cluster_rerun.py` | merges a targeted single-cluster re-run into a judged file |
| 26 | `g0_content_arm_check.py` | renders/validates the content-arm's 4 new conditions |
| 27 | `g1_manipulation_check.py` | does the model actually attribute a board note to its own item |
| 28 | `g1_grade.py` | grades script 27's output |
| 29 | `content_arm_analysis.py` | pre-registered analysis for the content arm (Result 5) |
| 30 | `g1_strong_referent_check.py` | G1 retry with a stronger referent, after name-only failed |
| 30 | `make_figures.py` | static figures for the write-up |

```bash
uv run python scripts/12_peer_loop.py --iter --conditions C0 C1b C2 C6
uv run python scripts/13_loop_analysis.py --in results/peer_loop.json
```

**Grading is separable from generation.** Transcripts are stored, so re-scoring costs API
calls and seconds — never another model run.

Two script folders sit outside the numbered pipeline, each for a different reason:

- `scripts/diagnostics/` — one-off investigations that answered a real question and were
  never wrong. See its README.
- `scripts/archive/` — superseded scripts whose result was later **retracted**, kept for
  the audit trail and never cited. See its README.

## Layout

- `src/pressure/` — library.
  - **Phase 1 (directions):** `arditi.py` (selection), `causal.py` (interventions +
    controls), `directions.py` (diff-of-means, sweep), `inversion.py` (Zhao
    reply-inversion), `hooks.py` (residual capture), `monitor.py` (per-turn `r_harm`/`r_ref`
    projection, orthogonalised against `r_arditi`).
  - **Phase 2 (peer framing):** `boards.py` (condition renderers + stimulus invariants),
    `tools.py` (AgentHarm tool schemas), `loop.py` (multi-turn tool loop), `grading.py`
    (their rubrics, structural/semantic split), `judge.py` (DeepSeek semantic judge +
    fallbacks).
  - **Shared:** `config.py`, `data.py` (corpora + leakage assertions), `model.py`,
    `device.py` (MPS/CUDA portability), `grade.py` (legacy substring grader), `plots.py`,
    `stats.py` (paired-binary tests shared by every contrast), `provenance.py` (guards
    against mixing artefacts from different model scales).
- `vendor/agentharm`, `vendor/zhao` — authors' released code, verbatim. Never reimplement
  from a paper's prose; every deviation here has cost a retraction.
- `boards/` — 10 hand-written peer boards, 4 variants each.
- `results/` — run outputs, gitignored (467MB+ locally). One exception committed:
  `gate_p_9b.json`. `tests/` — `pytest`.

## Portability / Spartan

Device logic lives in `device.py` (dtype, attention impl, memory guard) so the same scripts
run on Apple Silicon and A100/H100. The 4B runs locally for method development; all
headline numbers are the 9B on Spartan (A100/H100), including the full 208-item, 9-condition
corpus and the content-arm follow-on.

## Measurement discipline

Compliance has been mis-measured **fourteen** times, always by a proxy or a silently-wrong
code path. Four rules now hold:

1. The number that stands is the hand-check.
2. No number is quoted unless committed code reproduces it from a committed data file.
3. Never re-derive a stored verdict in an analysis script — read it.
4. Cold subagent review before trusting any metric. It has caught something every time.

See `docs/STATE.md` for the traps already fixed, which are easy to reintroduce.
