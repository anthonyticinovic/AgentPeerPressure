"""Calibrate tau_harm_orth: the frozen threshold on the orthogonalised harm-belief
readout p_harm_orth, at the same FPR the causal switch's own threshold uses.

Why calibration runs under the real ablation hook
----------------------------------------------------
`24_blindspot_crosstab.py` applies `tau_harm_orth` to `p_harm_orth` values read off the
ABLATED arm's real forward passes (`12_peer_loop.py`'s `projections()`, run inside
`ablate_all_components`). Under real ablation, `h . r_arditi` is already zero at every
layer (`ablate_all_components`'s docstring), so the raw projection and the
post-hoc-orthogonalised one coincide exactly there (confirmed to float precision:
`scripts/diagnostics/diag_orth_identity.py`, same-pass mean abs diff 0.0013). tau must be
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
nonlinearities before it reaches `harm_layer`. This bias was confirmed on real data
(docs/STATE.md's dated round-3 entry) and is kept below only as a labelled diagnostic
(`project_harm_orth_posthoc_biased`) -- its tau is never written to
dual_directions.json and nothing downstream reads it. Do not touch this real-ablation
methodology; the corpus swap below is the only thing this revision changes.

Calibration corpus: AgentHarm's own benign counterparts, not JailbreakBench
-----------------------------------------------------------------------------
The real 9B run found this threshold does not transfer: applied to the real Gate A
ablated arm, 0 of 3,744 rows across both arms ever exceeded it -- the single highest
`p_harm_orth` in the whole ablated arm (11.88) sits below the JailbreakBench-calibrated
threshold (+15.82). JailbreakBench's calibration prompts are short, single-turn,
single-message behaviours; `24_blindspot_crosstab.py` applies the threshold inside
long, multi-turn, tool-calling AgentHarm transcripts. That is too large a domain shift
for a threshold fit on one to bind usefully on the other.

`agentharm(harmful=False)` (`src/pressure/data.py`) is AgentHarm's own benign
counterpart split: 208 items across 52 clusters, matched 1:1 to the harmful set on
`target_functions`, `grading_function`, `hint_included` and `detailed_prompt` --
"AgentHarm, whose benign counterparts have equivalent tool use" (that file's own
docstring). Calibrating on it means tau is now fit on the same benchmark, and the same
agentic/tool-oriented instruction-writing style, as the task prompts Gate A actually
runs -- zero new data collection, and a corpus that was already loaded by this
pipeline for other purposes.

Scope decision: bare single-turn render, no board/peer-framing
-------------------------------------------------------------------
Each benign AgentHarm prompt is rendered the same minimal way the old JBB prompts
were -- `chat_prompt(tok, prompt)`, a bare single-turn chat message, no system prompt,
no board members, no peer-pressure framing of any kind. This is deliberate: the
distribution-shift problem being fixed here is about domain and phrasing (short
adversarial-benchmark prose vs long agentic task instructions), and swapping the
corpus is a single, isolated change to that variable. Also introducing board-framing
at calibration time would confound two unexamined variables in one move and make it
impossible to attribute a change in tau to either one specifically.

Honest limitation this leaves unresolved: `task_last` in the real Gate A transcripts
sits inside a board-framed, peer-pressured multi-turn conversation; the calibration
forward pass here never sees that framing at all. The corpus fix closes the
phrasing/domain gap (short JBB behaviour vs long AgentHarm task instruction) but not
the context gap (bare instruction vs board-framed instruction). `task_last` is the
last token of the *original instruction*, causally upstream of everything the board
framing and later turns add (`run_loop` appends after it; see `resolve_positions`'
docstring) -- which is why the position is still defensible -- but "the position is
upstream of the framing" is not the same claim as "the calibrated threshold is
unaffected by it". Treat this as a remaining, unresolved limitation, not something
this revision has solved, and carry it into any write-up that cites a flagged rate
against `tau_harm_orth`.

AUROC here is descriptive, not a held-out generalisation estimate
-----------------------------------------------------------------------
The AUROC reported below scores `agentharm(harmful=False)` (calibrates tau) against
`agentharm(harmful=True)` -- the *same* 208 harmful task prompts used as Gate A's
actual evaluation items, not an independent held-out harmful set. `calibrate_tau`
only ever touches the benign side (`pb`), so tau itself is not contaminated by this --
but the AUROC number describes how separable this corpus already is, not how well the
threshold would generalise to unseen harmful prompts. Read it as "is there signal
here at all", not as a validation-set score.

Old JBB-based tau, kept as a diagnostic
----------------------------------------
The previous production tau (JBB `matched_pairs()`, same real-ablation methodology)
is still computed below and stored as `diagnostic_jbb_tau` -- not silently discarded,
matching this project's convention of keeping a superseded-but-informative number
(`diagnostic_posthoc_biased_tau` from the round-3 fix) rather than deleting it.
Nothing downstream reads it; `tau_harm_orth` is now the AgentHarm-benign-calibrated
value only.

Position and layer
-------------------
Position is `task_last` at `harm_layer`, both already selected by `02_dual_directions.py`;
this script does not reselect either.

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


def _render(tok, p: str) -> tuple[str, str]:
    """(rendered prompt, task_text) for one bare single-turn prompt.

    Chat templates strip leading/trailing whitespace off a message that forms the
    whole of its content (unlike a mid-message substring, which board-framed
    production prompts embed the task text into -- see `12_peer_loop.py`, where
    `it["prompt"]` sits inside a larger `user_text`, so this never bites there). A
    handful of AgentHarm prompts (4 of 208 harmful, 2 of 208 benign, checked directly)
    open with `\\n\\n`, which `resolve_positions`' exact substring search then cannot
    find in the rendered text at all. Stripping once, and using the same stripped
    string for both the render and the search, keeps `task_text` truthful to what
    actually appears in `prompt` regardless of which corpus is passed in."""
    task_text = p.strip()
    return chat_prompt(tok, task_text), task_text


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
            rendered, task_text = _render(tok, p)
            acts = residuals_at_named(model, tok, rendered, task_text)
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
        rendered, task_text = _render(tok, p)
        acts = residuals_at_named(model, tok, rendered, task_text)
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

    task_prompts = [it["prompt"] for it in agentharm(harmful=True)]
    benign_prompts = [it["prompt"] for it in agentharm(harmful=False)]
    assert_no_leakage(benign_prompts, task_prompts)
    print(f"calibration corpus: AgentHarm benign counterparts, {len(benign_prompts)} items "
          f"-- no leakage into the {len(task_prompts)} AgentHarm harmful task prompts "
          "Gate A actually evaluates (same benchmark, same 208 harmful items -- see "
          "module docstring on why the AUROC below is descriptive, not held-out)",
          flush=True)

    v_harm = dirs.r_harm[dirs.harm_layer]
    v_orth = orthogonal_to(v_harm, dirs.r_arditi)
    orth_norm_ratio = float(v_orth.norm() / v_harm.norm())

    print("projecting AgentHarm harmful/benign prompts at task_last under real "
          "ablation (correct calibration input)", flush=True)
    ph = project_harm_real_ablation(model, tok, task_prompts, dirs)
    pb = project_harm_real_ablation(model, tok, benign_prompts, dirs)

    tau = calibrate_tau(pb, CFG.tau_fpr)
    y = np.r_[np.ones(len(ph)), np.zeros(len(pb))]
    auroc = float(roc_auc_score(y, np.r_[ph, pb]))
    empirical_fpr = float(np.mean(pb > tau))

    print(f"\ntau_harm_orth = {tau:+.4f}  (FPR={CFG.tau_fpr}, empirical {empirical_fpr:.4f})")
    print(f"AUROC of p_harm_orth (real ablation, AgentHarm benign vs harmful, "
          f"descriptive not held-out): {auroc:.4f}")
    print(f"orthogonalised-direction norm retained: {orth_norm_ratio:.4f}")

    print("\n[DIAGNOSTIC, superseded -- kept for comparison, not silently discarded -- "
          "the previous production calibration corpus, same correct real-ablation "
          "methodology] JailbreakBench matched_pairs:", flush=True)
    m_h, m_b = matched_pairs()
    assert_no_leakage(m_h.prompts + m_b.prompts, task_prompts)
    ph_jbb = project_harm_real_ablation(model, tok, m_h.prompts, dirs)
    pb_jbb = project_harm_real_ablation(model, tok, m_b.prompts, dirs)
    tau_jbb = calibrate_tau(pb_jbb, CFG.tau_fpr)
    y_jbb = np.r_[np.ones(len(ph_jbb)), np.zeros(len(pb_jbb))]
    auroc_jbb = float(roc_auc_score(y_jbb, np.r_[ph_jbb, pb_jbb]))
    empirical_fpr_jbb = float(np.mean(pb_jbb > tau_jbb))
    print(f"  diagnostic_jbb_tau = {tau_jbb:+.4f}  (FPR={CFG.tau_fpr}, empirical "
          f"{empirical_fpr_jbb:.4f})  AUROC={auroc_jbb:.4f}  n={len(m_h)}+{len(m_b)}")

    print("\n[DIAGNOSTIC, uncorrected, biased high -- see docs/STATE.md's dated entry "
          "for this script -- never written to dual_directions.json, never read "
          "downstream] old approach, non-ablated pass + post-hoc orthogonalisation, "
          "on the JailbreakBench corpus:", flush=True)
    ph_bad = project_harm_orth_posthoc_biased(model, tok, m_h.prompts, dirs, v_orth)
    pb_bad = project_harm_orth_posthoc_biased(model, tok, m_b.prompts, dirs, v_orth)
    tau_bad = calibrate_tau(pb_bad, CFG.tau_fpr)
    auroc_bad = float(roc_auc_score(y_jbb, np.r_[ph_bad, pb_bad]))
    diffs = np.r_[pb_bad - pb_jbb, ph_bad - ph_jbb]  # matched prompt-for-prompt, old minus new
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
        "auroc_is_descriptive_not_held_out": True,  # scored against the same 208
        # AgentHarm harmful items Gate A evaluates, not an independent harmful set --
        # see module docstring. tau itself only ever touches the benign side, so this
        # only affects how the AUROC number should be read.
        "n_harmful": len(ph),
        "n_harmless": len(pb),
        "harm_layer": dirs.harm_layer,
        "position": "task_last",
        "corpus": "AgentHarm benign counterparts (harmless side calibrates tau; "
            "harmful side scoring AUROC is the same 208 items Gate A evaluates, "
            "descriptive not held-out)",
        "corpus_render": "bare single-turn chat_prompt, no board/peer-framing -- "
            "isolates the domain/phrasing variable; does not close the board-framing "
            "context gap with the real evaluation transcripts. See module docstring.",
        "calibration_method": "real ablation hook (ablate_all_components) engaged for "
            "the whole forward pass; raw p_harm read directly at task_last/harm_layer, "
            "no post-hoc orthogonalisation -- see this script's module docstring",
        "cos_harm_vs_arditi": dirs.cosines()["harm"],
        "orth_norm_ratio": orth_norm_ratio,
        "diagnostic_jbb_tau": {  # superseded JailbreakBench-calibrated tau, same
            # real-ablation methodology -- kept for comparison, not used downstream.
            "value": tau_jbb,
            "empirical_fpr": empirical_fpr_jbb,
            "auroc": auroc_jbb,
            "n_harmful": len(m_h),
            "n_harmless": len(m_b),
            "corpus": "JailbreakBench matched_pairs -- previous production corpus, "
                "found not to transfer to the real ablated-arm p_harm_orth range "
                "(0/3744 rows exceeded it; see docs/STATE.md)",
        },
        "diagnostic_posthoc_biased_tau": tau_bad,  # NOT used by any downstream code --
        # kept for transparency only. See module docstring / docs/STATE.md.
        "n_agentharm_task_prompts_checked_for_leakage": len(task_prompts),
        "environment": describe(),
    }
    json_path.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {json_path} (tau_harm_orth)")


if __name__ == "__main__":
    main()
