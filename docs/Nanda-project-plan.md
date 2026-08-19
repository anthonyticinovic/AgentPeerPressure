# MATS 12.0 (Neel Nanda) — Project Execution Plan v2

**Working title (internal):** Which safety signal survives social framing — refusal or harmfulness?
**Deadline:** Fri 4 Sept 2026, 23:59 PT (extensions to 11 Sept available)
**Budget:** 20h project + 2h exec summary. Form Q time is *not* counted.
**Audience for this doc:** me. Not the submission.

**Changes from v1:** thesis reframed around two directions rather than one; Zhao et al. added as primary positioning; C1b (speaker-free floor) added; direction extraction moved off AgentHarm; residualisation control replaced; causal validation and behavioural pilot promoted to gates; OpenAI/HF incident demoted to a footnote.

---

## 0. Assumptions to confirm or override

| # | Assumption | Override if |
|---|---|---|
| A1 | 20h clock has not started. Reading done to *select* the problem — including the Zhao/Hu collision check — is uncounted. | Already burned hours → adjust §8 |
| A2 | Runs happen on A100/H100 (Spartan or GCP), not Apple Silicon. Local is for plotting/analysis only. | Cluster queue times bite → RunPod fallback |
| A3 | Model: Qwen 3.6 9B dense, instruct. 4B for iteration only. | 3.6 9B unavailable → Qwen 3.5 9B |
| A4 | **Zhao et al.'s harmfulness-direction extraction is reproducible from their released code on a 2026 Qwen.** | See §5.4 fallback. Do **not** reconstruct their method from an LLM summary or from memory — read the code |
| A5 | Peer boards: 8–10 distinct hand-written sets, not one template. Between-board variance is reported. | Need more scale → LLM-generate, read all of them, include random samples |
| A6 | This doc ≠ CLAUDE.md. Agent instructions live in Appendix B, split into its own file at build time. | — |
| A7 | Toggl running from hour 0, screenshot into the write-up. | — |

---

## 1. Thesis

One sentence: **Under peer-endorsement framing, the refusal signal and refusal behaviour come apart while the model's internal harmfulness belief is preserved — so under social context shift you should monitor harmfulness, not refusal.**

The design measures two frozen directions against the same behaviour. Every outcome is a result:

| Refusal signal | Harmfulness signal | Reading | Value |
|---|---|---|---|
| Drops with compliance | Holds | Zhao's jailbreak mechanism extends to social framing — a new shift axis, actionable monitoring conclusion | Solid |
| Holds | Holds | Signal–behaviour decoupling in a read-out monitor; peer framing is *not* a jailbreak mechanistically | Strong |
| Drops | Drops | Peer framing suppresses the harmfulness belief itself — no jailbreak in Zhao's set does this | Strongest |
| Holds | Drops | Anomalous; likely a direction-validity or position artefact. Investigate before believing | Diagnostic |

No branch is a dead project. This is the point of the rewrite: v1 had a headline whose most likely outcome was "ordinary monitor degradation, already covered by Kramár," and a fallback that was published in July.

**Decomposition thesis (secondary, not fallback):** the compliance shift is carried mostly by the *asserted precedent*, not by speaker attribution — quantify the speaker's marginal contribution in the agentic-harm setting, where nobody has measured it.

---

## 2. Positioning

Cite **Kramár and Zhao in the first paragraph**. Kramár is Neel's own GDM team and he links it in the application doc. Zhao is the paper that predicts my primary result; discovering it himself, unaddressed, ends the application.

| Work | Establishes | My delta |
|---|---|---|
| **Zhao et al., arXiv:2507.11878** (NeurIPS 2025) — LLMs Encode Harmfulness and Refusal Separately | Harmfulness is a distinct direction from refusal; some jailbreaks suppress refusal while the harmfulness belief persists; adversarial finetuning barely moves harmfulness; Latent Guard exploits this | Social/agentic context shift instead of adversarial prompts; no attacker, no finetuning, model uncompromised; monitoring-under-shift framing rather than jailbreak analysis |
| **Kramár et al., arXiv:2601.11516** (GDM) — Building Production-Ready Probes For Gemini | Probes fail to generalise under production shift; short→long context is the hard case; multi-turn and adaptive red-teaming tested; cyber-offensive domain | Social endorsement as a shift axis orthogonal to length and turn count; signal-vs-action rather than signal accuracy |
| **Hu & Qu, arXiv:2607.05545** (Jul 2026) — Most LLM Conformity Needs No Speaker | Removing the speaker leaves most apparent conformity intact (66.5% harmful revision vs 10.3% plain re-ask); source framing adds only a modest increment | Their setting is QA correctness; mine is harmful-action compliance in an agentic harness. Their control design is adopted directly as C1b |
| **Pinto, arXiv:2607.09156** (Jul 2026) — Present but Rescaled | Injected steering direction survives chat→agent transfer at near-full strength while behavioural coupling resets — a representational/behavioural dissociation | Read-out monitoring rather than injection; two directions compared rather than one; social framing rather than harness format |
| **Xu et al., arXiv:2607.06807** (AcMAS, ICML 2026) | Activation-based detection of *compromised* agents in MAS | Uncompromised agents under social pressure; no attacker |
| **Arditi et al.** — refusal is mediated by a single direction | The refusal direction and abliteration | Direction as *instrument*, not object of study |
| **Anil et al.** — many-shot jailbreaking | In-context compliance demos break safety | C1b/C3 separate "asserted precedent" from "attributed peers" |
| **Rose et al., arXiv:2604.01151** | Deception-probe aggregation across agents (multi-agent interpretability) | Cited for completeness; different question |

**Explicit sentence to include:** "I use two known directions as monitoring instruments, rather than demonstrating that a safety concept has a linear representation." Neel names the latter as a project type he is tired of. Make the avoidance legible.

**Do not** open with the OpenAI/Hugging Face incident. It is real, but the agents ran with cyber refusals deliberately reduced, it is cyber-domain while this runs AgentHarm, and the postmortem was presented by OpenAI and HF — not by anyone at GDM. If it appears at all, it is one line in Limitations noting that shared-channel agent incidents motivate the general question, not evidence for this mechanism.

---

## 3. Hypotheses

Track in a live hypothesis doc (Neel: *"keep a running doc with a list of your hypotheses"*). Log graphs into it as produced — it becomes the write-up skeleton.

| ID | Hypothesis | Prediction if true | Prediction if false |
|---|---|---|---|
| **H1** | Peer framing decouples the *refusal* signal from behaviour | Compliance rises; `proj_refusal` falls less than compliance implies; the (proj>τ ∧ complied) cell grows | Both move together → ordinary monitor degradation; Zhao's mechanism holds |
| **H2** | The *harmfulness* signal is preserved where refusal is not | `proj_harm` stable across C0→C2 while `proj_harm`'s item-level prediction of compliance also degrades | Both signals track compliance → no monitoring recommendation available |
| **H3** | The behavioural shift is carried by asserted precedent, not by speaker attribution | C1b reproduces most of C2; attributed-speaker increment is small | Attribution carries the effect → contradicts Hu & Qu in the harm-action setting, which is itself interesting |
| **H4** | The social channel is bidirectional | C3 (peers refused) raises refusal above C0 | Only compliance leaks → this is many-shot jailbreaking in costume |

**H1+H2 jointly are the headline.** H3 is the decomposition. H4 is cheap and diagnostic.

**Pre-registered primary contrast:** C2 vs C0, item-level AUROC of `proj_refusal` → compliance. Everything else is secondary and reported as such. Say this in the write-up so the multiple-comparisons objection is answered before it is raised.

---

## 4. Conditions

Task text **byte-identical** across all conditions. Only the surrounding context varies. Assert byte-identity in code.

| # | Condition | Content | Isolates | Core? |
|---|---|---|---|---|
| **C0** | Single-agent baseline | Task only | Baseline | Yes |
| **C1** | Token-matched filler | Task + neutral non-agentic text, token count matched to C2 | Context length | Yes |
| **C1b** | Precedent, no speaker | Same compliance assertions as C2, speaker attribution stripped | **Speaker-free floor** (Hu & Qu) | Yes |
| **C2** | Peers complied | Message board: 3–5 attributed peers report having done similar tasks | Headline manipulation | Yes |
| **C3** | Peers refused | Same board, peers report declining | Social channel direction | Yes |
| **C4** | Scrambled peers | C2 messages word-shuffled | Coherent signal vs token soup | Stretch |
| **C5** | Identity only | "You are agent 4 of 12", empty board | Identity vs content | Stretch |

**Cut order if time-pressed:** C5, then C4, then C3. C0/C1/C1b/C2 carry the argument. C1b is *not* cuttable — without it the C2−C0 delta is uninterpretable, because the published floor effect will dominate it.

**Board is deterministic and scripted, and there are 8–10 boards, not one.** Report the between-board spread on the headline metric. Frame it honestly as a controlled stimulus, not a multi-agent system. Do not build a real MAS — that scaffolding rabbit hole eats the budget, and Neel rewards knowing when not to enter one.

---

## 5. Setup

### 5.1 Model & access
- **Qwen 3.6 9B instruct**, bf16, single A100/H100.
- **Raw HF forward hooks** on `model.model.layers[i]`, capturing residual stream (post-block). Reads only, except the §5.5 causal validation.
- Qwen 3.6 4B for pipeline iteration, and reported once as a cheap scale baseline.

### 5.2 Data — extraction and evaluation corpora are now *different datasets*

v1 split AgentHarm 60/20/20 and left 22 evaluation items. That is too few for a 2×2 cell analysis, McNemar across five conditions, and a τ calibrated at 5% FPR. Fixed by separating the corpora entirely:

| Purpose | Corpus | N (target) |
|---|---|---|
| Direction extraction | AdvBench / JailbreakBench harmful vs Alpaca harmless, plain chat format | ~400 pairs |
| Layer/position/τ selection | Held-out slice of the same corpus | ~100 pairs |
| **Evaluation** | **AgentHarm — all 110 harmful base behaviours + matched benign counterparts** | 110 × conditions |

This removes the leakage worry structurally (different datasets, not just different rows) and restores power. Still assert in code that no extraction string appears in the evaluation set.

AgentHarm remains the evaluation set because its benign counterparts have equivalent tool use — that matching is what isolates alignment refusal from capability failure.

### 5.3 Direction 1 — refusal (Arditi)
1. Forward pass every extraction prompt in single-agent chat format.
2. Cache residual stream at all layers, post-instruction token positions.
3. `r_ref[l,p] = mean(harmful[l,p]) − mean(harmless[l,p])`; normalise to unit.
4. Select `(l*, p*)` by held-out AUROC on the selection slice. Save the sweep plot — it justifies the hyperparameter rather than asserting it.

### 5.4 Direction 2 — harmfulness (Zhao)
**Read §3 of arXiv:2507.11878 and the released code at `github.com/CHATS-lab/LLMs_Encode_Harmfulness_Refusal_Separately` before implementing.** Their harmfulness direction is *not* the Arditi diff-of-means; reconstructing it from an abstract or an LLM summary will silently produce two copies of the same vector, which would invalidate the entire design.

**Fallback if their extraction does not transfer to Qwen 3.6 (decide by hour 2, do not debug it past that):** build a simple, obvious-first proxy — force the model to answer "Is this request harmful? yes/no" on the extraction corpus and take diff-of-means over its *stated judgement* rather than over ground-truth labels. Report it as a proxy, state the difference from Zhao's construct, and do not claim it is theirs.

**Sanity check either way:** `cos(r_ref, r_harm)` at `(l*, p*)`. If it is above ~0.9 you have one direction wearing two hats and the two-signal claim collapses. Report the number regardless — it is a load-bearing fact.

### 5.5 Causal validation — mandatory, not optional (Gate B)
AUROC only shows the direction discriminates prompts. It does not show it mediates behaviour.

- Ablate `r_ref` from the residual stream in C0 on 30 harmful items. Refusal rate should collapse.
- Add `r_ref` on 30 benign items. Refusal should be induced.
- Same two checks for `r_harm`. Zhao's prediction is that `r_harm` steering changes what the model *treats as* harmful, while `r_ref` steering changes refusal surface behaviour without reversing the harmfulness judgement. Confirming or contradicting that on a 2026 model is itself worth a paragraph.

If `r_ref` fails causal validation, it is a correlate and the monitoring story is dead. Better to learn that at hour 3 than hour 19.

### 5.6 Freeze
Write `r_ref`, `r_harm`, `l*`, `p*`, and τ to disk. **Git commit before touching any multi-agent data.** The commit hash is the pre-registration; cite it in the write-up.

**Justification sentence for the write-up:** recomputing a diff-of-means direction on a shifted distribution trivially yields a different vector, so cosine similarity between recomputed directions tests nothing. Only projection onto a fixed direction tests monitor reliability.

### 5.7 Token position in long contexts
The confound that will eat this project if unhandled. Keep the task instruction as a **distinct terminal block** in every condition so `p*` is well-defined and comparable.
- Primary: last token of the task instruction block, immediately pre-generation.
- Secondary (report both): last token of full context.
- If primary and secondary disagree, that is a finding, not a bug — report it.

---

## 6. Measures & analysis

### 6.1 Per-run
- **Signals:** `proj_ref = ⟨a[l*,p*], r_ref⟩` and `proj_harm = ⟨a[l*,p*], r_harm⟩`. Report raw and z-scored against C0 benign.
- **Behaviour:** compliance = agent attempts the harmful action. AgentHarm grader + LLM judge, **manually verified on ≥30 randomly sampled transcripts**.

### 6.2 Decoupling — the corrected primary measure
v1 proposed residualising the refusal projection against a C0-vs-C2 context-detector score. That control is incoherent: the detector recovers condition identity, so residualising against it removes all between-condition variance including the effect being measured.

**Replace with the within-condition test.** Compute item-level AUROC of `proj → complied`, separately per condition, for both directions:

| | C0 | C1 | C1b | C2 | C3 |
|---|---|---|---|---|---|
| AUROC(`proj_ref` → complied) | — | — | — | — | — |
| AUROC(`proj_harm` → complied) | — | — | — | — | — |
| Compliance rate | — | — | — | — | — |
| Mean `proj_ref` (z) | — | — | — | — | — |
| Mean `proj_harm` (z) | — | — | — | — | — |

Decoupling = AUROC(`proj_ref`→complied) collapses toward 0.5 in C2 while remaining high in C0. That is the claim, measured cleanly, with no residualisation and no dependence on condition identity.

Secondary, for the monitor framing: τ fixed from the selection slice at 5% FPR on benign, **never re-tuned**; report the 2×2 `(proj > τ) × (complied)` per condition and the Δ in the (fired ∧ complied) cell between C0 and C2.

### 6.3 Statistics
- Paired items → McNemar for compliance-rate contrasts; bootstrap CIs over items for projection deltas and for AUROC differences.
- With 110 paired items, the minimum detectable compliance shift is roughly 10pp at 80% power. **State this in the write-up** so a null on C3 or C4 is read as "underpowered for small effects," not as "no effect."
- Between-board variance reported alongside every headline number.

### 6.4 Controls

| Control | Purpose | Kill implication |
|---|---|---|
| **Random direction** | Same pipeline, random unit vector, matched norm | Effect must vanish. Five random vectors, not one |
| **Prompted classifier** — "would you refuse this?" per condition | Neel: *"compare against 'just ask the model'"*; Kramár found probes and prompted classifiers complementary | If prompting tracks compliance as well as the projections, the probes add nothing — say so |
| **`cos(r_ref, r_harm)`** | Are these actually two signals? | High cosine collapses the two-direction claim |
| **Between-board spread** | Is the effect a property of peer framing or of four sentences I wrote? | Large spread → the stimulus, not the phenomenon |
| **4B scale check** | Cheap scale sensitivity | Report, do not over-interpret |

---

## 7. Gates

| Gate | Hour | Trigger | Action |
|---|---|---|---|
| **A** | pre-clock | Zhao/Hu/Pinto collision check | **Already done. Resolved: none of the three runs this design. Do not re-spend clock hours on it.** |
| **B** | 3 | `r_ref` fails causal validation (ablation does not collapse refusal), or held-out AUROC < 0.9, or C0 refusal rate < 70% on AgentHarm harmful | Monitor is not valid. Fix or abort **before** spending 15h on it |
| **B2** | 3 | `cos(r_ref, r_harm)` > 0.9 | Two-direction design is dead. Fall back to single-direction decoupling + decomposition, and say why in the write-up |
| **P** | 4 | **Pilot: 20 items × {C0, C1b, C2}, behaviour only.** C2 compliance does not exceed C0 by ≥10pp | There is no effect to decouple. Pivot to the decomposition-only project, or change the manipulation, before building the six-condition harness |
| **C** | 7 | 9B cannot follow the agentic harness coherently | Drop agentic framing; run chat-format peer turns. Do **not** debug scaffolding |
| **D** | 12 | C1 and C1b jointly reproduce the full C2 effect | The speaker contributes nothing in the harm-action setting. Headline becomes the quantified replication of Hu & Qu's floor in a new domain, plus the two-signal analysis, which still stands |

Gates are the point. Neel: *"realising the project is doomed halfway through, and just continuing"* is a named failure mode; *"knowing when to give up is a key research skill."* Full direction change → 20h clock resets.

---

## 8. Schedule

| Hours | Activity | Output |
|---|---|---|
| 0–1 | Read Zhao §3–4 + their extraction code. Skim Kramár §5–6 and Hu & Qu's no-source design. | Positioning paragraph drafted; extraction method understood, not guessed |
| 1–2 | Extract `r_ref` and `r_harm` on AdvBench/Alpaca; layer/position sweep; cosine check (**Gate B2**) | Sweep plot, two candidate directions |
| 2–3 | **Causal validation of both directions (Gate B).** Freeze, write to disk, git commit as pre-registration | Frozen `r_ref`, `r_harm`, `l*`, `p*`, τ + ablation/steering results |
| 3–4 | **Behavioural pilot, 20 items × {C0, C1b, C2} (Gate P)** | Go/no-go on the manipulation itself |
| 4–7 | Build harness for the core five conditions on AgentHarm; 8–10 boards | Runnable pipeline, byte-identity assertions passing |
| 7–12 | Full runs on 9B: both projections + rollouts, all 110 items | Raw results checkpointed to disk |
| 12–14 | Controls: random directions, prompted classifier, board variance, 4B scale (**Gate D**) | Control table |
| 14–17 | **Sanity-check protocol (§9)** | Verification notes for the write-up |
| 17–20 | Write-up + figures | Google doc |
| +2 | Exec summary + form Qs | Submission |

Set an hourly timer to zoom out: *am I making progress or in a rabbit hole?* (Neel's own recommendation.)

---

## 9. Sanity-check protocol

Neel calls this *"the most important piece of advice in this doc"* and says it should consume a meaningful fraction of the 20h. Three hours are ring-fenced. **Document what was checked** — "I read 30 transcripts and confirmed the positives were real" is explicitly cited as strong evidence of research skill.

- [ ] Read ≥30 raw transcripts across conditions. Confirm judge positives are genuine compliance, not refusal-with-caveats.
- [ ] Inspect actual prompts sent. Verify each board renders as intended and the task block is byte-identical across conditions.
- [ ] Recompute the headline AUROC-collapse number with an independent one-liner, not the pipeline.
- [ ] Verify no extraction-corpus string appears in the evaluation set.
- [ ] Confirm `r_ref` and `r_harm` were loaded from the frozen commit, not recomputed anywhere in the run path.
- [ ] Read all 8–10 boards yourself. Include a random sample in the write-up.
- [ ] Ask "what's the dumbest way this is wrong?" per key result. Current best answers: `r_harm` is secretly `r_ref` (→ cosine check); position-selection artefact (→ report both positions); judge grading refusal-with-hedging as compliance (→ transcript read); τ mis-calibrated across conditions (→ within-condition AUROC does not depend on τ); the AUROC collapse is a range-restriction artefact because compliance is near-ceiling in C2 (→ check the compliance base rate per condition before interpreting).
- [ ] Include **5 randomly selected** (not cherry-picked) transcripts immediately after the exec summary.

---

## 10. Write-up

Exec summary: **≤1 page ideal, ≤3 pages, ≤600 words, graphs mandatory.** Bullets fine. Lead with the finding, not chronology.

1. Problem + why interesting, with Kramár and Zhao positioning in para 1
2. High-level takeaways
3. Random qualitative transcripts and a random sample of boards
4. Two-signal result: the AUROC table (§6.2) as one figure
5. Causal validation of both directions — what the ablation and steering showed
6. Decomposition: C1 / C1b / C2 — answers "isn't this just length, or just the speaker-free floor?"
7. Controls and what they rule out; the `cos(r_ref, r_harm)` number
8. Limitations: scripted boards ≠ real MAS; single model family; refusal may not be strictly 1-D (concept cones, arXiv:2602.02132); `r_harm` is a replication of someone else's construct, with all the transfer risk that implies; N and minimum detectable effect; near-ceiling compliance risk in C2
9. What I would do next

**Write it yourself.** Neel: raw LLM prose is *"a significant negative signal — I see hundreds of them, and they blur together."* Use an LLM for critique passes with an anti-sycophancy frame, not for generation.

Form Qs are read **first** as a preliminary filter. Budget the full 2h. Name the model, the key experiment, the surprising number. For "1–3 pieces of evidence you'd do good research": lead with the AAAI 2026 first-authorship and state the specific contribution.

---

## 11. Neel dos/don'ts — final check

| ✅ Do | Status |
|---|---|
| Baselines: random vectors, prompted classifier, 4B scale | §6.4 |
| Start with the obvious thing | Prompted classifier; forced-judgement proxy fallback for `r_harm` |
| Check the phenomenon replicates in *my* setting before building on it | Gate P pilot, §7 |
| Causal validation, not just AUROC | §5.5, Gate B |
| Sanity-check the agent | §9 |
| Random, not cherry-picked examples | §10.3 |
| Pivot when doomed | §7 |
| Recent models only | Qwen 3.6 |
| Show hyperparameters and how data was made | §5.3, §5.7, §6.3 |
| Negative results, honestly reported | §1 outcome grid |
| Agentic coding tool, sanity-checked | Appendix B |

| ❌ Don't | Guard |
|---|---|
| "Safety concept has a linear representation" | Directions are instruments; stated explicitly |
| Reproducing a published result and calling it new | Zhao, Hu & Qu, Pinto all cited in positioning with explicit deltas |
| Areas he's cooled on (SAEs, circuits, grokking, toy models) | None present |
| Old models (GPT-2, Pythia, Gemma 2) | None |
| Cherry-picked qualitative evidence | Random sampling, board variance reported |
| Overclaiming / hyping | Outcome grid written before the data exists |
| Spreading thin | Five core conditions; cut order defined |
| Rabbit-holing | Hourly timer, Gates C and P |
| Raw LLM prose | Write-up self-authored |

---

## 12. Links

- Zhao et al. — https://arxiv.org/abs/2507.11878 · code: https://github.com/CHATS-lab/LLMs_Encode_Harmfulness_Refusal_Separately
- Kramár et al. — https://arxiv.org/abs/2601.11516
- Hu & Qu (speaker-free floor) — https://arxiv.org/abs/2607.05545
- Pinto (chat→agent steering dissociation) — https://arxiv.org/abs/2607.09156
- AcMAS (Xu et al.) — https://arxiv.org/abs/2607.06807
- Rose et al. (multi-agent probe aggregation) — https://arxiv.org/abs/2604.01151
- AgentHarm — https://arxiv.org/abs/2410.09024
- More than a single direction — https://arxiv.org/abs/2602.02132
- Neel: pragmatic interpretability, Explore/Understand/Distill, Key Mindsets, ML paper writing — all linked from the application doc
- Neel's 600k-token mech interp context file — load into Claude Code by default
- jupyter-mcp-server for persistent kernel

---

## Appendix A — Honest reservation

The v1 risk was that the safety machinery crowded out the finding. That risk is now smaller, because the two-direction design has no branch that yields nothing. The new risks are different and worth naming:

1. **`r_harm` may not replicate.** The whole upgrade rests on reproducing someone else's extraction on a model they did not use. Gate B2 and the §5.4 fallback exist for this. If both fail, the project degrades to the v1 single-direction design — which is a weaker application, but a survivable one.
2. **C2 may be near-ceiling on compliance.** If peer framing drives compliance to 90%+, item-level AUROC becomes range-restricted and the decoupling metric degrades for a boring statistical reason rather than an interesting mechanistic one. Check the base rate at the Gate P pilot and pick a harder task subset if needed.
3. **The decomposition may be entirely Hu & Qu's floor.** Likely, on their numbers. That is fine — it is a quantified extension to harmful-action agentic settings, and it is the *secondary* claim now, not the fallback the project rests on.

The gate discipline is the same as v1. Trust it. Neel: *"negative or inconclusive results that are well-analysed are much better than a poorly supported positive result."*

---

## Appendix B — CLAUDE.md content (split into its own file at build time)

```
# Project: which safety signal survives social framing — refusal or harmfulness?

## Environment
- Model loaded ONCE in a dedicated top cell. Never restart the kernel without asking.
- Persistent Jupyter kernel via jupyter-mcp-server. Port-forward to laptop.
- Save every plot to disk as PNG in addition to displaying.
- Checkpoint activations, datasets, and rollouts to disk. A crashed kernel must not lose work.
- Long runs go to background scripts with logs, not notebook cells.

## Hard rules
- TWO directions are frozen after extraction: r_ref (Arditi diff-of-means) and r_harm
  (Zhao). Layer, position, and threshold are frozen with them.
  Never recompute any of them on multi-agent data. Never re-tune tau.
- Load directions from the frozen artifact file at the start of every run. Assert the
  file's hash matches the pre-registration commit.
- r_harm must be implemented from the Zhao paper's released code. Do NOT infer the
  method from the abstract or from your own summary. If the code cannot be made to run,
  stop and tell me — do not substitute a method you invented.
- Task instruction text must be byte-identical across all conditions. Assert in code.
- No extraction-corpus string may appear in the evaluation set. Assert in code.
- Report both primary and secondary token positions. Do not silently pick the better one.
- Report per-board results, not just the pooled mean.

## Do not
- Do not residualise projections against condition identity or a condition classifier.
  The within-condition AUROC(proj -> complied) is the decoupling measure.
- Do not report a single random-vector control. Use five, matched norm.

## What I will verify by hand (do not assume these passed)
- Rendered prompts per condition, and all 8-10 boards
- 30 sampled transcripts vs judge labels
- Headline AUROC number via independent recomputation
- cos(r_ref, r_harm)

## Style
- Plain .py scripts and notebook cells. Raw HF hooks, not TransformerLens.
- Write a short report after each experiment: what was run, what the numbers were,
  what could be wrong with them.
```
