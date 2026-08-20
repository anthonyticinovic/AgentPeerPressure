"""Extract r_ref and sweep for (l*, p*).

Local 4B run:   uv run python scripts/01_extract_directions.py --iter
Spartan 9B run: uv run python scripts/01_extract_directions.py

Results land in results/r_ref_sweep.json (numbers, human-readable) and
results/directions_raw.pt (tensors, consumed by Tasks 1.4-1.7).
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pressure.config import CFG
from pressure.data import assert_no_leakage, extraction_corpus, matched_pairs, split_extract_select
from pressure.device import describe, empty_cache, seed_everything
from pressure.directions import (
    accumulate_mean,
    calibrate_tau,
    diff_of_means,
    project_prompts,
    select_best,
    sweep_auroc,
)
from pressure.model import layer_types, load_model
from pressure.plots import sweep_figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true", help="use the local iteration model")
    args = ap.parse_args()

    seed_everything(CFG.seed)
    env = describe()
    print("environment:", env, flush=True)

    harmful, harmless = extraction_corpus()
    h_ext, h_sel = split_extract_select(harmful)
    b_ext, b_sel = split_extract_select(harmless)
    print(f"corpus: {harmful.source} / {harmless.source}", flush=True)
    print(f"  extraction {len(h_ext)}+{len(b_ext)}   selection {len(h_sel)}+{len(b_sel)}", flush=True)

    model, tok = load_model(eval_model=not args.iter)
    repo = CFG.iter_model if args.iter else CFG.eval_model
    types = layer_types(model)
    print(f"model: {repo}  layers={len(types)}", flush=True)

    t0 = time.time()
    print("pass 1/2: accumulating means over the extraction slice", flush=True)
    mean_h = accumulate_mean(model, tok, h_ext.prompts, progress=100)
    mean_b = accumulate_mean(model, tok, b_ext.prompts, progress=100)
    r_ref = diff_of_means(mean_h, mean_b)
    empty_cache(CFG.device)

    print("pass 2/3: projecting the held-out AdvBench/Alpaca slice", flush=True)
    proj_h = project_prompts(model, tok, h_sel.prompts, r_ref)
    proj_b = project_prompts(model, tok, b_sel.prompts, r_ref)
    auroc_unmatched = sweep_auroc(proj_h, proj_b)

    # Selection runs on topic-matched pairs, never on held-out AdvBench/Alpaca.
    # The latter is separable at 0.9955 by bag-of-words, so its AUROC is vacuous.
    print("pass 3/3: projecting the topic-matched JailbreakBench pairs", flush=True)
    m_h, m_b = matched_pairs()
    assert_no_leakage(h_ext.prompts, m_h.prompts)
    proj_mh = project_prompts(model, tok, m_h.prompts, r_ref)
    proj_mb = project_prompts(model, tok, m_b.prompts, r_ref)
    auroc = sweep_auroc(proj_mh, proj_mb)

    l_star, o_star, best = select_best(auroc)
    tau = calibrate_tau(proj_mb[:, l_star, o_star])
    elapsed = time.time() - t0

    offset = CFG.sweep_offsets[o_star]
    print(
        f"\nSELECTED on matched pairs: l*={l_star} ({types[l_star]})  offset={offset}  "
        f"AUROC={best:.4f}  tau={tau:.3f}",
        flush=True,
    )
    print(
        f"  same cell, unmatched AdvBench/Alpaca AUROC={auroc_unmatched[l_star, o_star]:.4f} "
        f"(vacuous: layer 0 scores {auroc_unmatched[0].max():.4f})",
        flush=True,
    )
    print(f"  matched-pair AUROC at layer 0 = {auroc[0].max():.4f}  [{elapsed/60:.1f} min]", flush=True)

    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "r_ref": r_ref,
            "auroc_ref": auroc,
            "auroc_unmatched": auroc_unmatched,
            "proj_mh": proj_mh,
            "proj_mb": proj_mb,
            "offsets": CFG.sweep_offsets,
            "l_ref": l_star,
            "o_ref": o_star,
            "tau": tau,
            "proj_h": proj_h,
            "proj_b": proj_b,
            "model": repo,
            "layer_types": types,
        },
        CFG.results_dir / "directions_raw.pt",
    )
    summary = {
        "model": repo,
        "environment": env,
        "corpus": {"harmful": harmful.source, "harmless": harmless.source},
        "n_extract": len(h_ext),
        "n_select": len(h_sel),
        "l_star": l_star,
        "l_star_type": types[l_star],
        "offset": offset,
        "auroc_matched": best,
        "auroc_unmatched_same_cell": float(auroc_unmatched[l_star, o_star]),
        "auroc_unmatched_layer0": float(auroc_unmatched[0].max()),
        "auroc_matched_layer0": float(auroc[0].max()),
        "tau": tau,
        "elapsed_min": round(elapsed / 60, 2),
        "auroc_grid": auroc.tolist(),
        "auroc_grid_unmatched": auroc_unmatched.tolist(),
        "selection_corpus": [m_h.source, m_b.source],
        "layer_types": types,
        "sweep_offsets": list(CFG.sweep_offsets),
    }
    (CFG.results_dir / "r_ref_sweep.json").write_text(json.dumps(summary, indent=2))
    sweep_figures(auroc, CFG.sweep_offsets, types, l_star, o_star)
    print(f"wrote {CFG.results_dir / 'r_ref_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
