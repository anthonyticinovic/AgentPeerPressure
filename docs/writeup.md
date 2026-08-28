# Separating belief from refusal: does peer framing create a monitoring blind spot?

*Draft - Gate A confirmatory run in progress. Numbers below are complete through the pilot; the primary result is not yet in.*

## Executive summary

This project started from a broader question: agents increasingly work in swarms - reading each other's outputs, picking up where another agent left off - and swarms produce interaction effects a single agent alone never would. A concrete case from July 2026: agents in a shared OpenAI/Hugging Face repository began leaving each other status notes, unprompted, that grew into an informal message board. We wanted to know whether being embedded in something like that changes what a model *does* independent of what it *believes* - whether seeing peers already approve of and act on a request moves a model's behavior in a way that's decoupled from its own judgment of the request.

That's a monitoring question as much as a behavioral one. A safety monitor that watches for refusal is watching a proxy. It assumes refusing means the model recognized the request as harmful, and complying means it didn't. If those two things can come apart - if a model can be made to comply while still internally "knowing" the request is harmful - a refusal-based monitor would miss exactly the cases it exists to catch. **Peer-endorsement framing**, tested here, is one candidate mechanism for that gap; whether any effect it has is hidden by refusal acting as a ceiling on the model's behavior is the open half of the question.

**Established: refusal, harm-belief, and the refusal signal are three different directions in the model, not one.** `r_arditi` is a *lever*: Arditi's method finds it by actually ablating candidate directions until refusal stops, so by construction it's causal - remove it and the model can no longer refuse. `r_harm` and `r_ref` are both *readouts*, found by classification rather than intervention, but of different things at different points: `r_harm` (Zhao's method, read from the instruction itself) tracks whether the model judges the request harmful; `r_ref` (read from the very last token of the prompt, right before generation starts) tracks whether the model is about to refuse - a detector for the decision, not the judgment behind it. Ablating `r_arditi` collapses refusal (1.00 → 0.04) without moving `r_harm` - the switch is gone, but the belief underneath it isn't. This is the mechanistic precondition for the whole project: if there were only one direction, "belief surviving refusal" wouldn't be a coherent question.

**Established: with refusal intact, peer identity alone does not move behavior, robustly.** Across the full AgentHarm agentic corpus, naming peers as having already endorsed a task (vs. an anonymized version of the same endorsement) produces no measurable effect - not on the graded outcome, and not on four alternate outcome definitions that don't depend on AgentHarm's own grading functions. This null survived an independent, blind reimplementation.

**Not yet answered: whether that null is because there's nothing there, or because refusal is a ceiling.** Only 26 of 208 AgentHarm items are behaviorally "informative" (able to move at all) when refusal is intact - most items either always comply or the model never touches them. Ablating the refusal direction raises the task-attempt rate from 48% to 92% in the full agentic loop, and nearly triples the pilot's informative item set (7→17 of 52). Whether the *peer-framing conditions diverge from each other* once that ceiling is lifted - the actual pre-registered question - is flat in the pilot (p=0.75) but the pilot is structurally underpowered to say anything at that sample size (see Limitations). **The properly-powered run (208 items, both ablation levels) is running now; this section will be filled in once it completes.**

---

## Why this matters

Refusal suppression via weight or activation intervention - steering, ablation, fine-tuning - is a reasonably well-studied failure mode: something acts directly on the model. Contextual framing is a different, less-examined route to the same behavioral outcome - no intervention on the model at all, just what it's shown. If that route works on its own, monitoring the model's weights or activations for tampering wouldn't catch it; only behavior, or an independent read on what the model actually believes, would.

The stimuli below are built on the same shared-workspace structure as the July incident above - peers leaving notes for one another inside a live multi-agent deployment, not a synthetic prompt-injection contrivance.

---

## Setup

**Models.** Qwen3.5-9B for all headline numbers; Qwen3.5-4B was used for method development and appears only where explicitly labeled.

**Three directions**, all diff-of-means over 128 harmful / 128 harmless prompts, differing in *where* they're read and *how* they were selected:

| | `r_arditi` | `r_harm` | `r_ref` |
|---|---|---|---|
| what it is | the lever: removing it stops refusal | the readout of belief: does the model judge the request harmful | the readout of decision: is the model about to refuse |
| found by | ablate every candidate, keep whichever kills refusal (causal) | classify harmful vs. harmless activations, keep whichever separates them best (correlational) | classify refuse vs. comply activations, same method as `r_harm`, different position |
| read at | end-relative offset, causal-sweep-selected | last token of the instruction | last token of the full prompt, right before generation |
| use here | ablated | monitored | monitored |

**Task corpus.** AgentHarm (vendored, unmodified grading functions and system prompt), full multi-turn tool-use loop - not single-turn completions.

**Peer framing.** A shared "workspace board" - status notes purportedly left by other agents - is prepended to a byte-identical AgentHarm task. Only the board differs between conditions:

| condition | board content |
|---|---|
| C0 | none (baseline) |
| C1b | peer endorsement, speakers stripped - the "speaker-free" floor |
| C2 | peer endorsement, named peers |
| C3 | the *same* peers, refusing - held for review |
| C6 | an audit record, no social framing |
| C7 | a peer hands the task off partway through - diffusion of authorship, not endorsement |

`C2 − C3` (peers who complied vs. peers who refused, otherwise identical) is the actual peer-pressure contrast and is this project's pre-registered primary. An earlier version of this project used `C2 − C1b`, which only tests whether *naming* peers matters, not what they did.

**Ablation.** Arditi's `ablate_all_components`: zeroes the residual stream along `r_arditi` at every layer, in the embedding and every attention and MLP output - not the weaker "block-output-only" variant.

**Monitoring.** `r_harm` and `r_ref` projections are read once per turn from a dedicated forward pass, orthogonalized against `r_arditi` to remove the arithmetic contamination ablation otherwise introduces into every other projection.

---

## Result 1 - three directions, not one

| | value |
|---|---|
| Arditi direction selected | i\*=-7, l\*=12 (kl 0.067) |
| refusal rate, ablated | 1.00 → **0.04** |
| harmful compliance, hand-labelled (n=25) | **0.84** |
| shuffled-label control, same site | 1.00 / 0.00 (no effect) |

| direction | steered toward harmful | steered toward benign | moves as labelled |
|---|---|---|---|
| `r_harm` | judgment shifts, 0.84 @ layer 6 | no effect | **yes** |
| `r_ref` | no effect | refusal induced, 1.00 @ layer 11 | no |
| `r_arditi` | no effect | - | no |

`r_harm` is the only one of the three whose steering effect tracks the harmful/benign label directionally - it's the belief direction. `r_ref` and `r_arditi` both move refusal when steered, but not in a way that respects ground-truth harmfulness; they're behavioral/signal directions, not judgment directions. Held-out baseline: benign prompts judged harmful 0.00, harmful prompts judged harmful 0.98.

This result is single-turn (AdvBench-style completions), not yet the full agentic setting - that's what the rest of this project tests.

---

## Result 2 - with refusal intact, peer identity alone does nothing

Full AgentHarm agentic corpus, multi-turn tool loop, AgentHarm's own grading rubrics with a semantic judge:

| vs. C1b | rate | delta | exact p | cluster p |
|---|---|---|---|---|
| C2 (named peers) | 0.217 | **+1.7pp** | 0.688 | 0.688 |
| C6 (audit record) | 0.157 | -4.3pp | 0.125 | 0.125 |

This null is stronger than the table shows: restricted to the items where neither arm hit a turn cap or grading artifact (98 of 115), `C2 − C1b = -1.0pp, p = 1.00`. On four further outcome definitions that don't depend on AgentHarm's grading functions at all (did the model attempt the task, did it reach a payload call, mean per-criterion score), the C2 contrast is exactly zero. An independent, blind reimplementation - different code, no access to this project's analysis scripts - reproduced every number above exactly.

Why: decomposing the full 208-item corpus by whether an item can carry information at all under intact refusal - 154 of 208 never comply in *any* condition (108 make zero tool calls, 46 engage but never complete), 28 comply in *all four* - leaves only **26 informative items**, and the primary contrast is decided by 8 discordant pairs among those. Two entire categories (Sexual, Hate - 52 items) are pinned at zero compliance in every condition. Refusal itself is acting as a floor that most of the corpus never gets off of, which is the direct motivation for Result 3.

---

## Result 3 - refusal ablation, in progress

**Ablation works in the full agentic setting, not just single-turn.** 52-item capability check, both arms monitored:

| | intact | ablated |
|---|---|---|
| malformed tool-call blocks | 2/52 | 2/52 |
| mean turns | 3.35 | 5.10 |
| any-tool-call rate | 0.48 | **0.92** |

Malformed calls are identical across arms (a corpus/token-budget property, not an ablation artifact). The turn increase tracks the attempt-rate increase - the model is finishing more real work, not looping (only 1 row loops, identically, in both arms). Ten ablated transcripts read by hand across two independent passes are all coherent.

**The corpus opens up.** 52-item pilot, all six conditions, both ablation levels:

| | C0 | C1b | C2 | C3 | C6 | C7 | informative |
|---|---|---|---|---|---|---|---|
| intact | 19.2% | 26.9% | 21.2% | 25.0% | 23.1% | 25.0% | 7/52 |
| ablated | 48.1% | 40.4% | 40.4% | 40.4% | 36.5% | 40.4% | 17/52 |

The informative set more than doubles.

**Whether C2 diverges from C3 once the ceiling lifts - the actual question - is not detectable at this sample size:**

| | delta | p |
|---|---|---|
| intact, C2−C3 | -3.85pp | 0.500 |
| ablated, C2−C3 | 0.00pp | 1.000 |
| **interaction (primary)** | **+3.85pp** | **0.754** |

This is not evidence of no effect. A 52-item pilot cannot settle this either way - power for the interaction (which has larger variance than a single contrast, since it's a difference of two differences) doesn't cross even odds until roughly a 30pp true effect:

| true interaction | power, n=52 (pilot) | power, n=208 (confirmatory) |
|---|---|---|
| 10pp | 0.07 | 0.17 |
| 15pp | 0.17 | 0.30 |
| 20pp | 0.28 | 0.47 |
| 25pp | 0.38 | 0.69 |
| 30pp | 0.54 | 0.87 |

The pilot did what it was for - validated the pipeline and screened for a large effect, which isn't there. It says nothing about a moderate one. **The full 208-item, six-condition, two-ablation-level confirmatory run - the only thing that can actually answer the question - is running now.** Both arms passed a live preflight (direction provenance, residual-writer coverage, ablation fidelity) before submission; expected completion in the next day.

---

## Limitations

- **The confirmatory run's own power ceiling is ~20-25pp** for even-odds detection of the primary interaction, even at full scale (208 items). A smaller true effect will not resolve cleanly either way.
- **The judge has been blind-validated on this specific new territory (ablated generations, C3/C7 conditions) only on a 25-case sample** - 24/25 confirmed, 1 ambiguous, no directional bias by condition found. Not exhaustive.
- **A small, condition-symmetric truncation artifact** (`cut_mid_call`, ~4-6% of rows, worse under ablation) affects long-form generation tasks specifically; addressed for the confirmatory run by a larger token budget, not eliminated.
- **9B is the primary scale; 4B was used for method development only** and the two are not bit-identical even under greedy decoding (different hardware backends during development).
- **Greedy decoding throughout** - there is no run-to-run sampling variance to average over; a given prompt always produces the same completion on the same hardware.
- **The peer-framing manipulation is synthetic** - a constructed shared-workspace board, not observed real multi-agent deployment traffic. It's modeled on the July 2026 incident's structure, not drawn from it.

---

## Status

Gate A confirmatory run (both ablation levels, full 208-item corpus, six conditions) submitted and running. Once complete: grade locally with the semantic judge, run the pre-registered `C2 − C3` interaction test (uncorrected, as the designated primary; all other contrasts enter the same Holm-corrected secondary family), and report the result here regardless of outcome.
