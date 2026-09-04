# Diagnostics — one-off investigations, not superseded results

Not part of the numbered pipeline, and not `scripts/archive/`: nothing here produced a
retracted result. Each answers a specific question raised during development and is kept
because the answer is still useful context, not because it needs to be re-run.

| File | Question it answers |
|---|---|
| `diag_inversion.py` | Is the model's harmfulness judgement absent, or just unread by greedy decoding? |
| `diag_orth_identity.py` | Does the cross-arm `p_harm_orth`/`p_harm` identity claimed by `23_calibrate_tau_orth.py` actually hold? |
