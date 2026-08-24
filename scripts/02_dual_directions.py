"""Extract r_ref and r_harm at Zhao's two token positions, and run Gate B2.

Zhao's harmfulness direction and Arditi's refusal direction are the same diff-of-means
read at different positions (vendor/zhao/src/extract_hidden.py):
    t_inst      = last token of the instruction   -> harmfulness  -> our `task_last`
    t_post-inst = last token of the whole prompt  -> refusal      -> our `context_last`

Gate B2: if cos(r_ref, r_harm) > 0.9 at the selected layer, the two-direction design
is dead and the project falls back to single-direction decoupling plus decomposition.

    uv run python scripts/02_dual_directions.py --iter
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pressure.config import CFG
from pressure.data import extraction_corpus, matched_pairs, split_extract_select
from pressure.device import describe, seed_everything
from pressure.directions import accumulate_mean_named, cosine, diff_of_means
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, layer_types, load_model

POSITIONS = {"task_last": "r_harm (Zhao t_inst)", "context_last": "r_ref (Zhao t_post-inst)"}


def project_named(model, tok, prompts, direction) -> np.ndarray:
    """(n_prompts, n_layers) projections at one named position."""
    out = []
    for p in prompts:
        acts = residuals_at_named(model, tok, chat_prompt(tok, p), p)
        out.append((acts[direction["pos"]] * direction["vec"]).sum(-1).numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    args = ap.parse_args()

    seed_everything(CFG.seed)
    harmful, harmless = extraction_corpus()
    h_ext, _ = split_extract_select(harmful)
    b_ext, _ = split_extract_select(harmless)

    model, tok = load_model(eval_model=not args.iter)
    repo = CFG.iter_model if args.iter else CFG.eval_model
    types = layer_types(model)
    print(f"model: {repo}  env: {describe()}", flush=True)

    print("accumulating means at both named positions", flush=True)
    mh = accumulate_mean_named(model, tok, h_ext.prompts, progress=100)
    mb = accumulate_mean_named(model, tok, b_ext.prompts, progress=100)

    dirs = {pos: diff_of_means(mh[pos], mb[pos]) for pos in POSITIONS}

    # Selection on topic-matched pairs, per Checkpoint 4.
    m_h, m_b = matched_pairs()
    result: dict[str, dict] = {}
    for pos, label in POSITIONS.items():
        print(f"scoring {label} at {pos}", flush=True)
        d = {"pos": pos, "vec": dirs[pos]}
        ph = project_named(model, tok, m_h.prompts, d)
        pb = project_named(model, tok, m_b.prompts, d)
        y = np.r_[np.ones(len(ph)), np.zeros(len(pb))]
        auroc = np.array(
            [roc_auc_score(y, np.r_[ph[:, l], pb[:, l]]) for l in range(len(types))]
        )
        l_star = int(auroc.argmax())
        result[pos] = {
            "label": label,
            "auroc_by_layer": auroc.tolist(),
            "l_star": l_star,
            "l_star_type": types[l_star],
            "auroc": float(auroc[l_star]),
            "tau": float(np.quantile(pb[:, l_star], 1 - CFG.tau_fpr)),
        }
        print(f"  l*={l_star} ({types[l_star]})  AUROC={auroc[l_star]:.4f}", flush=True)

    # --- Gate B2 -------------------------------------------------------------
    r_harm = dirs["task_last"]
    r_ref = dirs["context_last"]
    per_layer = [cosine(r_harm[l].numpy(), r_ref[l].numpy()) for l in range(len(types))]
    l_h, l_r = result["task_last"]["l_star"], result["context_last"]["l_star"]
    cos_matched_layer = per_layer[l_r]
    cos_at_own_best = cosine(r_harm[l_h].numpy(), r_ref[l_r].numpy())

    print("\n--- GATE B2 -------------------------------------------------")
    print(f"cos(r_harm, r_ref) at a matched layer {l_r}      : {cos_matched_layer:.4f}")
    print(f"cos(r_harm@l{l_h}, r_ref@l{l_r}) at own best layers: {cos_at_own_best:.4f}")
    print(f"per-layer cosine  min={min(per_layer):.3f}  max={max(per_layer):.3f}")
    verdict = "PASS - two distinct directions" if cos_matched_layer <= 0.9 else "FAIL - one direction, two hats"
    print(f"verdict: {verdict}")

    out = {
        "model": repo,
        "environment": describe(),
        "positions": result,
        "gate_b2": {
            "cos_matched_layer": cos_matched_layer,
            "cos_at_own_best": cos_at_own_best,
            "cos_per_layer": per_layer,
            "threshold": 0.9,
            "verdict": verdict,
        },
        "layer_types": types,
        "note": "r_harm and r_ref are the same diff-of-means read at different token positions",
    }
    (CFG.results_dir / "dual_directions.json").write_text(json.dumps(out, indent=2))
    torch.save({"r_harm": r_harm, "r_ref": r_ref, "model": repo, **result},
               CFG.results_dir / "dual_raw.pt")
    print(f"\nwrote {CFG.results_dir / 'dual_directions.json'}")


if __name__ == "__main__":
    main()
