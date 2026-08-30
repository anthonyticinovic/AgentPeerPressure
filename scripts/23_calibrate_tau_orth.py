"""Calibrate tau_harm_orth: the frozen threshold on the orthogonalised harm-belief
readout p_harm_orth, at the same FPR the causal switch's own threshold uses.

Why calibration runs under the real ablation hook
----------------------------------------------------
`24_blindspot_crosstab.py` applies `tau_harm_orth` to `p_harm_orth` values read off the
ABLATED arm's real forward passes (`12_peer_loop.py`'s `projections()`, run inside
`ablate_all_components`). Under real ablation, `h . r_arditi` is already zero at every
layer (`ablate_all_components`'s docstring), so the raw projection and the
post-hoc-orthogonalised one coincide exactly there (confirmed to float precision:
`scripts/diag_orth_identity.py`, same-pass mean abs diff 0.0013). tau must be
calibrated on that same real-ablation distribution, not approximated from a different
one -- which is what an earlier version of this script did.

That earlier version ran the calibration corpus through a plain, non-ablated forward
pass and orthogonalised the *direction* post-hoc: `(h - proj_r(h)).v == h.(v -
proj_r(v))` for any h, v, r, so dotting a non-ablated hidden state with an
orthogonalised direction is an exact piece of linear algebra. But that identity relates
a non-ablated hidden state's orthogonalised-direction projection to itself -- it says
nothing about what a REAL, hook-based ablated pass would produce for the same prompt,
because real ablation zeroes `r_arditi`'s component at every layer, not just at the
read-out layer, and that earlier removal propagates through attention/MLP
nonlinearities before it reaches `harm_layer`. Checked on real data (1240 matched rows
across `gate_a_full_base_judged.json`/`gate_a_full_abl_judged.json`, see docs/STATE.md
dated entry for this script): the two are strongly correlated (Pearson r=0.995) but the
gap is systematic, not symmetric noise -- mean signed diff +0.22 (base minus abl),
median +0.14, 85.6% of rows positive, consistent (+0.18 to +0.24) across every
condition. A threshold calibrated in that "too-high" regime under-flags when applied to
the real ablated arm's systematically lower values -- conservative in direction (an
underestimate of the blind-spot rate this pipeline would report) but a real bias, not
an approximation.

Fixed here: the harmless calibration corpus is now run with
`ablate_all_components(model, dirs.r_arditi)` engaged for the whole forward pass,
matching exactly what `24_blindspot_crosstab.py` applies the threshold to. Because raw
and orthogonalised projections coincide under real ablation (the same-pass identity
above), the raw `task_last`/`harm_layer` projection is read directly inside the `with`
block -- no separate orthogonalisation step. The old post-hoc approach is kept below
only as a labelled, printed diagnostic (`project_harm_orth_posthoc_biased`):
`dual_directions.json` never stores its tau and nothing downstream reads it.

Calibration corpus and position
--------------------------------
`matched_pairs()`'s harmless side -- the topic-matched JailbreakBench split
`02_dual_directions.py` already uses for AUROC selection, not the easy
AdvBench/Alpaca split (bag-of-words separable at 0.9955, a vacuous FPR). Position is
`task_last` at `harm_layer`, both already selected by `02_dual_directions.py`; this
script does not reselect either.

Distribution-shift limitation -- read before trusting this threshold
------------------------------------------------------------------------
`tau_harm_orth` is calibrated on short, single-turn, single-message JailbreakBench-style
benign prompts. It is applied (by `24_blindspot_crosstab.py`) inside multi-turn,
tool-calling, peer-framed agentic transcripts hundreds of tokens longer. The
`task_last` position is defensible for that transfer -- it is the last token of the
*original instruction*, causally upstream of every turn of scaffolding that follows
(`run_loop` appends after it; see `resolve_positions`' docstring), so nothing about
the multi-turn machinery can move what is being read. But "the position is upstream
of the drift" is not the same claim as "the calibrated threshold generalises to that
context" -- a benign JailbreakBench prompt and a benign AgentHarm task instruction are
still two different distributions over instruction text. Treat `tau_harm_orth` as an
assumption about calibration transfer, not a proven property, and carry that caveat
into any write-up that cites a flagged rate against it.

    uv run python scripts/23_calibrate_tau_orth.py --iter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.causal import ablate_all_components  # noqa: E402
from pressure.config import CFG  # noqa: E402
from pressure.data import agentharm, assert_no_leakage, matched_pairs  # noqa: E402
from pressure.device import describe, seed_everything  # noqa: E402
from pressure.directions import calibrate_tau  # noqa: E402
from pressure.hooks import residuals_at_named  # noqa: E402
from pressure.model import chat_prompt, load_model  # noqa: E402
from pressure.monitor import Directions, orthogonal_to  # noqa: E402


def project_harm_real_ablation(model, tok, prompts: list[str], dirs: Directions) -> np.ndarray:
    """p_harm_orth per prompt, correctly: task_last hidden state at harm_layer, read
    with `ablate_all_components` engaged for the whole forward pass. Under real
    ablation h.r_arditi is already zero at every layer, so the raw projection onto
    `r_harm` already equals what post-hoc orthogonalising would give -- no separate
    correction is applied or needed. This is the calibration input tau_harm_orth is
    drawn from; see the module docstring for why the previous non-ablated,
    post-hoc-orthogonalised approach was biased."""
    v_harm = dirs.r_harm[dirs.harm_layer]
    out = []
    with ablate_all_components(model, dirs.r_arditi):
        for p in prompts:
            acts = residuals_at_named(model, tok, chat_prompt(tok, p), p)
            h = acts["task_last"][dirs.harm_layer]
            out.append(float(h @ v_harm))
    return np.array(out)


def project_harm_orth_posthoc_biased(model, tok, prompts: list[str], dirs: Directions,
                                      v_orth) -> np.ndarray:
    """DIAGNOSTIC ONLY -- see module docstring. The original approach: a plain,
    non-ablated forward pass with the direction orthogonalised post-hoc. Biased high
    relative to `project_harm_real_ablation`; its tau is never written to
    dual_directions.json and nothing downstream reads it."""
    out = []
    for p in prompts:
        acts = residuals_at_named(model, tok, chat_prompt(tok, p), p)
        h = acts["task_last"][dirs.harm_layer]
        out.append(float(h @ v_orth))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true", help="use the local iteration model")
    args = ap.parse_args()

    seed_everything(CFG.seed)
    repo = CFG.iter_model if args.iter else CFG.eval_model

    dirs = Directions.load(CFG.results_dir)
    if dirs.model != repo:
        raise SystemExit(
            f"loaded directions are {dirs.model} but the requested model is {repo}. "
            f"{'Pass --iter to' if dirs.model == CFG.iter_model else 'Drop --iter, or'} "
            "rerun 02_dual_directions.py and 03_arditi_selection.py at the matching "
            "scale first -- calibrating at one scale against directions fit at another "
            "would dot vectors from two different hidden spaces."
        )

    model, tok = load_model(eval_model=not args.iter)
    print(f"model: {repo}  env: {describe()}", flush=True)
    print(f"harm_layer={dirs.harm_layer}  cos(r_harm, r_arditi)={dirs.cosines()['harm']:+.4f}",
          flush=True)

    m_h, m_b = matched_pairs()
    task_prompts = [it["prompt"] for it in agentharm(harmful=True)]
    assert_no_leakage(m_h.prompts + m_b.prompts, task_prompts)
    print(f"calibration corpus: {len(m_h)} harmful / {len(m_b)} harmless "
          f"(JailbreakBench matched pairs) -- no leakage into {len(task_prompts)} "
          "AgentHarm task prompts", flush=True)

    v_harm = dirs.r_harm[dirs.harm_layer]
    v_orth = orthogonal_to(v_harm, dirs.r_arditi)
    orth_norm_ratio = float(v_orth.norm() / v_harm.norm())

    print("projecting matched pairs at task_last under real ablation "
          "(correct calibration input)", flush=True)
    ph = project_harm_real_ablation(model, tok, m_h.prompts, dirs)
    pb = project_harm_real_ablation(model, tok, m_b.prompts, dirs)

    tau = calibrate_tau(pb, CFG.tau_fpr)
    y = np.r_[np.ones(len(ph)), np.zeros(len(pb))]
    auroc = float(roc_auc_score(y, np.r_[ph, pb]))
    empirical_fpr = float(np.mean(pb > tau))

    print(f"\ntau_harm_orth = {tau:+.4f}  (FPR={CFG.tau_fpr}, empirical {empirical_fpr:.4f})")
    print(f"AUROC of p_harm_orth (real ablation) on matched pairs: {auroc:.4f}")
    print(f"orthogonalised-direction norm retained: {orth_norm_ratio:.4f}")

    print("\n[DIAGNOSTIC, uncorrected, biased high -- see docs/STATE.md's dated entry "
          "for this script -- never written to dual_directions.json, never read "
          "downstream] old approach, non-ablated pass + post-hoc orthogonalisation:",
          flush=True)
    ph_bad = project_harm_orth_posthoc_biased(model, tok, m_h.prompts, dirs, v_orth)
    pb_bad = project_harm_orth_posthoc_biased(model, tok, m_b.prompts, dirs, v_orth)
    tau_bad = calibrate_tau(pb_bad, CFG.tau_fpr)
    auroc_bad = float(roc_auc_score(y, np.r_[ph_bad, pb_bad]))
    diffs = np.r_[pb_bad - pb, ph_bad - ph]  # matched prompt-for-prompt, old minus new
    print(f"  tau_harm_orth (post-hoc, DO NOT USE) = {tau_bad:+.4f}  AUROC={auroc_bad:.4f}")
    print(f"  signed diff (post-hoc minus real-ablation), matched prompts, n={len(diffs)}: "
          f"mean={float(diffs.mean()):+.4f}  median={float(np.median(diffs)):+.4f}  "
          f"{100 * float(np.mean(diffs > 0)):.1f}% positive -- replicates Reviewer A's "
          "cross-arm finding on this calibration corpus itself")

    json_path = CFG.results_dir / "dual_directions.json"
    existing = json.loads(json_path.read_text()) if json_path.exists() else {}
    if existing and existing.get("model") not in (None, repo):
        raise SystemExit(
            f"{json_path} is for {existing.get('model')}, but this calibration is for "
            f"{repo}. It is only fully rewritten by 02_dual_directions.py (03 never "
            "touches it, and this script only ever patches tau_harm_orth into a file "
            f"already matching), so it can go stale relative to the .pt artefacts -- "
            f"rerun 02_dual_directions.py {'--iter ' if args.iter else ''}first so the "
            "JSON matches what Directions.load() actually reads."
        )
    existing["model"] = repo
    existing["tau_harm_orth"] = {
        "value": tau,
        "fpr": CFG.tau_fpr,
        "empirical_fpr": empirical_fpr,
        "auroc": auroc,
        "n_harmful": len(ph),
        "n_harmless": len(pb),
        "harm_layer": dirs.harm_layer,
        "position": "task_last",
        "corpus": "JailbreakBench matched_pairs (harmless side calibrates tau, harmful side scores AUROC)",
        "calibration_method": "real ablation hook (ablate_all_components) engaged for "
            "the whole forward pass; raw p_harm read directly at task_last/harm_layer, "
            "no post-hoc orthogonalisation -- see this script's module docstring",
        "cos_harm_vs_arditi": dirs.cosines()["harm"],
        "orth_norm_ratio": orth_norm_ratio,
        "diagnostic_posthoc_biased_tau": tau_bad,  # NOT used by any downstream code --
        # kept for transparency only. See module docstring / docs/STATE.md.
        "n_agentharm_task_prompts_checked_for_leakage": len(task_prompts),
        "environment": describe(),
    }
    json_path.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {json_path} (tau_harm_orth)")


if __name__ == "__main__":
    main()
