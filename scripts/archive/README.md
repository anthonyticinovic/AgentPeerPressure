# Archived scripts — superseded, kept for the audit trail

Every script here produced a result that was later **retracted**. They are retained so
the reasoning behind the retraction is reproducible, not because their numbers stand.
The findings that replaced them live in the plan's "Findings from Checkpoint 5" section
and in the research log (`artifacts/inspection.html`).

Do not cite outputs from these scripts. The working pipeline is `scripts/01`–`04`.

| File | Was | Why retired |
|---|---|---|
| `causal_validation.py` | first Gate B attempt | verdict threshold sat on noise (+0.025 vs +0.007 null) |
| `inversion_sweep.py` | all-layer steering sweep | steering method wrong: one vector pushed into all layers |
| `audit_rerun.py` | post-audit ablation rerun | `lay[-1] // 2` bug — ablated layer 15, not the selected l\* |
| `ablation_fixed.py` | ablation at selected layers | selection still AUROC/position−1 based, not Arditi's grid |
| `matched_fit.py` | fit on JBB matched pairs | corpus-axis diagnostic; superseded for selection, but its confound finding still informs r_ref-as-monitor |
| `arditi_sweep_layers_only.py` | "Arditi" causal sweep | swept 32 of 288 candidates (layers only, position −1), judge-graded, no filters |
| `rdo_lite.py` | self-invented "RDO-lite" | optimised −log P("Sure, here"), a prefix proxy; not Wollschläger et al.'s method; unnecessary once diff-of-means passed Gate B |
