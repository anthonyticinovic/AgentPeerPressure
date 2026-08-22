# Project state — read this first

Last updated 2026-08-22. Written to survive context compaction: if you are picking this
up cold, this file plus the research log is enough to continue without re-deriving
anything.

**Thesis.** Under peer-endorsement framing, the refusal *signal* and refusal *behaviour*
come apart while the model's internal *harmfulness belief* is preserved. A monitor
watching the refusal signal would not fire.

**Where we are.** Phase 1 complete on the local 4B. Both preconditions are established:
refusal is causally mediated, and the harmfulness belief is a *separate* direction. The
main experiment — peer framing on AgentHarm — has **not** started.

---

## 1. Established results (all Qwen3.5-4B, held-out, controlled)

### Gate B — refusal is causally ablatable
Arditi selection over the full 288-candidate grid (9 post-instruction positions × 32
layers), scored by refusal metric, filtered `induce>0, kl<0.1, l<0.8L`.

| | value |
|---|---|
| selected direction | **i\* = −7, l\* = 12**, kl 0.067 |
| refusal under ablation | 1.00 → **0.04** |
| harmful compliance (hand-labelled, n=25) | **0.84** |
| shuffled-label null, same site | 1.00 / 0.00 |
| best *separating* direction (pos −1, l 22) | kl **0.45** → inadmissible, 0.08 compliance |

### Gate B2 — the belief is a different direction
Zhao reply-inversion, 10 arms × 32 layers × 50 held-out prompts per class.

| direction | pushed + (more harmful) | pushed − | peak refusal | moves as labelled |
|---|---|---|---|---|
| `r_harm` (t_inst) | **0.78 @ L6** | no effect | **0.00** | **yes** |
| `r_ref` (t_post-inst) | no effect, any layer | 1.00 @ L11 | 1.00 | no |
| `r_arditi` (Gate-B winner) | no effect, any layer | — | 1.00 | no |

`r_harm` is directional only at **layers 5–10** (where +v and −v move judgement opposite
ways). Baseline: benign 0.00, harmful 0.96 judged harmful.

### The headline claim
**Three functions, three different vectors:** the direction that *detects* harm is not
the one that *causes* refusal, and neither is the one that *holds the belief*.

**Do not overclaim.** "Only r_harm can make benign look harmful" is FALSE — `r_ref−`
also reaches 1.00. The correct claim is sign-consistency: `r_harm` raises the harmful
verdict when pushed toward *more harmful*; `r_ref` only when pushed toward *less
harmful*, while refusing at rate 1.00.

---

## 2. Frozen parameters — do not silently change these

| parameter | value | why |
|---|---|---|
| inversion template | **idx 25** of `vendor/zhao/src/all_inversion_template.py` | chosen on the *selection* split; acc 1.00, 0 unparsed. Only 15/49 templates work |
| steering coefficient | **2.0** | calibrated on selection split; independently equals Zhao's Qwen2 value |
| directions | **raw, unnormalised** diff-of-means | coeff 2 is calibrated against the unnormalised scale |
| intervention | activation addition, forward **pre-hook**, **one layer**, **prefill pass only** | `max_decode_step_while_intervene=1` in their shell script |
| spans | `r_harm` → instruction tokens only; `r_ref` → all tokens | their Appendix E.1; instruction-only for r_ref gives ~zero effect |
| max_new_tokens | 8 | verdict lands at char ≤1 (measured, not assumed); 12× cheaper than their 100 |
| thinking mode | **disabled** | separates far better here (7.34 vs 1.41). Logged as a limitation |
| model | `Qwen/Qwen3.5-4B` (iter) / `Qwen3.5-9B` (eval) | 9B refuses to load on MPS by design |

---

## 3. Measurement discipline — the recurring failure mode

Compliance/judgement has been mis-measured **five** times, every time by a proxy:

1. substring refusal markers (12× inflation) — retracted
2. an over-strict compliance judge (0.20 vs 0.84 true) — corrected by hand
3. self-invented "RDO" optimising an affirmative *prefix* — retracted entirely
4. Chinese refusals scored `unparsed` — English-only regex
5. `"Yes, …"` verdicts scored `unparsed` — 723 replies, 43% of that bucket

**Rule: the number that stands is always the hand-check.** Any new metric must be
validated against blind labels before its value is quoted.

### Current scorer (validated)
`judge_inversion_strict` in `src/pressure/inversion.py` is **positional —
earliest-match-wins** among affirm / negate / refuse. This is what makes
`"Yes, but I cannot elaborate"` a verdict and `"I cannot answer 'Certainly'"` a refusal.
Blind-labeller agreement **98.1%** on the hardest strata (the one disagreement was the
labeller's own low-confidence call).

`judge_inversion` is Zhao's, kept verbatim as the faithful baseline **including its bug**
(unbounded `'no'`, so "cannot" reads as a harmless judgement). On our corpus their scorer
is ~82% on the cases it counts, with 17.9% wrong-and-counted; it discards 11.3%, of which
40% are real "Yes" verdicts. Their error is one-directional: it can only turn refusals
into harmless verdicts — the same direction as their own conclusion.

### Analysis traps already fixed (do not reintroduce)
- **`unparsed` is in the denominator of P(judged harmful).** A 100%-garbage cell scores
  exactly 0.00 on a panel whose baseline is 0.00 — a destroyed model is arithmetically
  identical to an inert one. Cells with unparsed ≥ 0.5 are marked unreadable and drawn
  as gaps, never zeros. 11 cells were affected.
- **Aggregate "consistent in every panel" verdicts are constants.** Floor/ceiling makes
  4 of 10 arms unpassable at any data (harmless panel starts at 0.00, harmful at 0.96).
- **`max()` on an all-tied series invents a peak layer** (reported "@L2" from iteration
  order). Such arms must read "no effect, any layer".
- **Layers 0–1 are representational damage, not steering** — +v and −v give the *same*
  answer. Excluded consistently via `LAYER0_EXCLUDE`.
- `"correct"` must NOT be an affirmative token ("the correct answer is No" → harmful).

---

## 4. Pipeline

```
01_extract_directions   r_ref + layer/position AUROC sweep
02_dual_directions      r_ref/r_harm at both positions
03_arditi_selection     288-candidate grid -> (i*, l*)          [Gate B]
04_arditi_generation_check  256-token behavioural confirmation
05_build_log            -> artifacts/inspection.html  (research log)
06_inversion_sweep      --stage calibrate | sweep               [Gate B2]
07_inversion_analysis   sign-consistency, content gap, artefact test
08_build_writeup        -> artifacts/writeup.html    (narrative)
diag_inversion          template scan (49 templates, logit-level)
```
`scripts/archive/` holds superseded scripts that produced **retracted** results — kept
for the audit trail, never to be cited. See its README.

Runtime on M4 Pro: Arditi grid ~53 min; full inversion sweep ~2 h (10 arms).
Long runs: use the harness's `run_in_background`, NOT `nohup &` — the latter is invisible
in the user's task list.

---

## 5. Two artefacts, two audiences — keep them separate

| | URL | contains |
|---|---|---|
| **Research log** | https://claude.ai/code/artifact/804abf22-30c5-4015-b7ff-27ba99bb7555 | working record: every decision, dead end, retraction |
| **Write-up** | https://claude.ai/code/artifact/f37f59f0-eaab-44f8-a298-385f7cf2327f | findings only, MATS reviewers, interp-fluent |

**User's explicit instruction:** implementation detail and the correction history stay in
the log. They are noise in the write-up while we are still short of the main experiment.
Write-up is for MATS reviewers — assumes interp fluency, no hand-holding.

---

## 6. Open / next

1. **PENDING — user is hand-labelling the 25 Gate-B samples** in
   `results/REVIEW_FOR_ANTHONY_gateb.md` (blind; my labels stripped). When they return,
   score against `results/AUDIT_gateb_key.json` and report agreement honestly, including
   disagreements. This closes the last independently-unverified number (the 0.84).
   The independent-subagent route is **blocked**: cyber safeguards refuse the malware/
   hacking transcripts. Do not retry it.
2. **NEXT — Phase 2 peer-framing design.** Categories fixed a priori: Disinformation,
   Hate, Harassment, Copyright (52 behaviours, 208 variants, cluster on `id_original`,
   MDE ≈ 15 pp). Board framing = shared agent task-queue. Task text must be
   byte-identical across conditions (assert in code). User flagged realism concern:
   one-shot social proof may be less realistic than gradual scope creep — but escalation
   breaks byte-identity. Unresolved; scrutinise rendered prompts with data in hand.
3. **9B on Spartan** — first step that genuinely needs the cluster. `device.py` handles
   MPS/CUDA; no code change expected, set `PRESSURE_EVAL_MODEL` and drop `--iter`.
4. Deferred: XSTest/SORRY-Bench off-diagonal data would make `v_refuse` behaviourally
   distinct from `r_ref`. On our pure-diagonal corpus they are the same object.
5. Observation to test at 9B: ablation elicits cyber compliance most readily
   (0.90 vs 0.80 non-cyber, n=25 — one item wide, not yet significant).

---

## 7. Sources — read the code, not the abstract

- **Arditi** arXiv:2406.11717 — selection is a *grid* over (position × layer), min
  bypass_score s.t. `induce>0, kl<0.1, l<0.8L`; refusal metric = log-odds of first token
  in R. R derived empirically here = `{40}` = `"I"`.
- **Zhao** arXiv:2507.11878 — code **vendored at `vendor/zhao`**. Their released code
  computes ONE quantity, `mean(harmful) − mean(harmless)`; `mode_dir` only picks the
  token position. The paper's `µ_refused − µ_accepted` is achieved by *data selection*
  (their `data/readme.md` says to split by behaviour first). On a pure-diagonal corpus
  the two definitions coincide — which is our regime.
- **Wollschläger** arXiv:2502.17420 — cited for concept cones. We have NOT implemented
  it; the earlier "RDO" was self-invented and is retracted.
