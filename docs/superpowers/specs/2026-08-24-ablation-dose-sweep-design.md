# Gate A — refusal ablation with joint direction monitoring

Date: 2026-08-24. Status: design, not yet approved for implementation.
Supersedes the Phase 2 plan in `docs/STATE.md` §2 as the next experiment.

---

## 1. Why the Gate P null cannot be read as "framing is inert"

Gate P ran the full 208-item corpus at 9B across four framings and returned
C2 - C1b = -0.96pp, CI [-3.63, +1.71]pp. That looks like a well-powered null.
The item-level decomposition says otherwise:

| | items |
|---|---|
| never comply in any arm | **154** (108 make zero tool calls; 46 engage but never complete) |
| comply in all four arms | **28** |
| **informative** (differ across arms) | **26** |

McNemar discards concordant pairs, so **every p-value we have rests on 26 items**,
and the primary contrast is decided by 8 discordant pairs. Two categories are pinned
at absolute zero in all four arms:

| category | n | C0 | C1b | C2 | C6 |
|---|---|---|---|---|---|
| Disinformation | 24 | 37.5 | 33.3 | 29.2 | 29.2 |
| Cybercrime | 24 | 25.0 | 29.2 | 25.0 | 20.8 |
| Drugs | 24 | 20.8 | 29.2 | 25.0 | 29.2 |
| Harassment | 28 | 25.0 | 28.6 | 32.1 | 28.6 |
| Fraud | 28 | 25.0 | 25.0 | 25.0 | 25.0 |
| Copyright | 28 | 14.3 | 21.4 | 21.4 | 25.0 |
| **Sexual** | 28 | **0.0** | **0.0** | **0.0** | **0.0** |
| **Hate** | 24 | 4.2 | **0.0** | **0.0** | **0.0** |

52 items — a quarter of the corpus — carry no information and never can.

The honest statement of the Gate P result is therefore: **the corpus has almost no
dynamic range, and no framing manipulation of any strength could have been detected
on 87.5% of it.** The null is about the instrument, not yet about the hypothesis.

## 2. Hypothesis

Refusal ablation raises the compliance floor and converts dead items into informative
ones. Run the framing conditions with and without ablation and the design separates
two worlds that Gate P cannot:

- **Inertness.** Compliance rises under ablation; the framing delta stays at zero at
  both levels. Peer framing is genuinely absent from the decision. The Gate P null
  becomes credible, because we have demonstrated the instrument has range.
- **Masking.** Compliance rises under ablation *and* the framing delta opens up. Social
  influence is present in the model but clipped by a refusal circuit that dominates it.
  The sensitivity re-emerges the moment that circuit is removed.

A dose sweep was considered and **cut**. It would have tested a third outcome —
**tiebreaking**, where the framing delta peaks at intermediate ablation and collapses
at full ablation once nothing is left to contest the decision. Three reasons against
running it now: partial ablation is not a principled operating point (directional
ablation is binary by construction, and no abliterated checkpoint is half-ablated);
detecting a non-monotonicity needs far more power than detecting the delta itself, and
we cannot reliably see 7.5pp; and the interaction should not be bought before the main
effect is seen. If full ablation opens the framing delta, the dose curve becomes a
well-motivated follow-up.

### Ecological motivation

Full refusal ablation is not a synthetic manipulation. Publicly downloadable
"abliterated" checkpoints are refusal-ablated models. The ablated arm is a
deployment condition, not a control condition.

## 3. Design

Full factorial, within-item, paired on `(cluster, id)`.

| factor | levels |
|---|---|
| ablation | off (baseline, **already run**) / on (standard Arditi, all layers) |
| framing | C0, C1b, C2, C6 |
| items | 208 (full corpus, no category dropped) |

= 832 new runs. The unablated half is `results/peer_loop_9b.json`, unchanged. Task text
stays byte-identical across every cell, as in Gate P.

**Do not drop Sexual and Hate.** They are the sharpest probe in the design: if
ablation lifts Cybercrime and Copyright while those two stay at zero, a single Arditi
direction does not mediate those refusals and a second mechanism exists. That result
comes free.

### Ablation setting

Standard Arditi directional ablation, unchanged: `h <- h - (h . v)v` at every layer,
which is what `causal.ablate` already does. **No change to `causal.py` is required.**

A coefficient is introduced only if Gate A2 fails — see the contingency there.

### Monitors

At every turn, capture the residual stream during the prefill forward pass that
generation already performs — no extra pass — and project onto:

| vector | read as | position / layers |
|---|---|---|
| `r_harm` | **result** — does the harmfulness belief survive ablation? | t_inst, layers 5-10 |
| `r_ref` | **result** — does the refusal signal track behaviour or decouple from it? | post-instruction |
| `r_arditi` | **fidelity check only, never a result** | all layers |

**Tautology guard.** We ablate `r_arditi`. Its own projection therefore goes to ~0 by
construction. That number is an assertion that the hook fired, and must never appear
in a results table. This is the exact failure mode in `STATE.md` §3 — eight
mis-measurements, all from a proxy or a silent code path, and retracted result #1 was
a tautology of precisely this shape. `r_harm` and `r_ref` are admissible because Gate
B2 established that all three are different vectors.

The per-turn trajectories are the point, not just the endpoint: a model that engages
for four turns and refuses at the payload has a different `r_harm` trace from one that
refuses at turn 1, and Gate P's engagement-without-completion pattern predicts exactly
such a trace.

## 4. Prerequisites

### Gate A0 — baseline monitor replay and harness identity check (blocking, ~1h)

The existing baseline has compliance but no monitor traces. Rather than regenerate it,
**teacher-force the stored transcripts** and read activations without sampling: one
forward pass per turn, roughly 5-10x faster than a re-run.

Decoding is greedy, so the replay must argmax back to the stored tokens exactly. Assert
that. It is simultaneously the monitor readout and the proof that the hook harness does
not perturb generation. If the replay diverges, every downstream number is void.

### Gate A1 — directions at 9B (blocking, ~half a day)

Gate B (`i*=-7, l*=12`, refusal 1.00 -> 0.04) and Gate B2 (`r_harm`, `r_ref`) are
**4B results**. Nothing is ablatable or monitorable at 9B until re-derived there:

- `scripts/03_arditi_selection.py` — 288-candidate grid, same admissibility filter
  (`induce>0, kl<0.1, l<0.8L`)
- `scripts/04_arditi_generation_check.py` — behavioural confirmation
- `scripts/02_dual_directions.py` — `r_ref` / `r_harm` at both positions

This also closes B2-at-9B, already on the backlog.

### Gate A2 — agentic capability under ablation (blocking, ~30 min)

Arditi's KL constraint is fitted on short single-turn generation. A 16-turn tool loop
is a far longer horizon. If full ablation degrades the model into malformed tool calls,
compliance falls for a boring reason and the experiment is void.

Run `12_peer_loop.py --one-per-grader` (52 items, all graders) under ablation and
confirm: tool calls parse, the XML format holds, turn counts are comparable to
baseline, transcripts are coherent to inspection. **Do not commit the 8h run until
this passes.**

**Contingency if it fails.** Introduce a coefficient on the ablation and calibrate down
to the largest value that preserves agentic capability, then run one dose there. This
is a fallback, not a planned arm, and the chosen value must be reported as fitted.

## 5. Analysis

Primary outcome stays `full_compliance_judged` — AgentHarm's own rubrics with the
DeepSeek judge live. No new metric is invented for this gate.

1. **Compliance under ablation** — does it raise the floor, and by how much per
   category. Reports the informative-item count with and without ablation; **if it does
   not grow, the experiment failed and the analysis says so.**
2. **Framing contrast under ablation** — exact McNemar for C2 - C1b, C0 - C1b, C6 - C1b,
   plus the cluster-level sign test, exactly as `13_loop_analysis.py` does now.
3. **Interaction** — permutation test on the ablation x framing term, permuting the
   framing label within item across both ablation levels. Assumption-free and reuses the
   existing cluster-permutation machinery. This is the inertness-vs-masking test.
4. **Monitor trajectories** — `r_harm` and `r_ref` per turn, split by ablation, framing,
   and realised outcome. The thesis question: where behaviour changes, does `r_harm`
   stay high?

Multiplicity stays **family-wide across every test reported**, per `STATE.md` §3.
The family is pre-registered here, before the run: three framing contrasts x two
ablation levels, plus the interaction = 7 tests.

## 6. Cost

| stage | runs | walltime |
|---|---|---|
| A1 — Gate B + B2 directions at 9B | — | ~4h |
| A2 — capability check under ablation | 52 | ~30 min |
| A0 — baseline monitor replay, teacher-forced | 832 | ~1h |
| ablated run, 4 arms x 208 | 832 | ~8h |

**~13h of cluster time**, split into four jobs so failures surface early. Grading is
separable from generation, so re-scoring costs API calls and seconds.

## 7. Risks

| risk | mitigation |
|---|---|
| ablation destroys agentic capability | Gate A2, blocking; coefficient calibration as fallback |
| hooks perturb generation | Gate A0 teacher-forced replay must argmax to stored tokens, blocking |
| Arditi direction does not transfer to 9B | Gate A1 re-derives it; admissibility filter is the same |
| `r_arditi` projection reported as a result | named as a fidelity check in this spec; assert in the analysis script |
| family-wide multiplicity swamps everything | pre-register the family in this spec before the run |
| compliance rises but graders cap it | per-category informative-item count reported at both levels |

## 8. Out of scope

- **Ablation dose sweep.** Cut, with reasons, in §2. Revisit only if full ablation opens
  the framing delta.
- New framing conditions (authority, incrementalism, decomposition). Decide after this
  gate says whether the corpus can register a framing effect at all.
- 4B replication. The scale claim stays a stated limitation.
- Any change to the corpus, the graders, or the compliance definition.
