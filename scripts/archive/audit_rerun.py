"""Re-run every causal claim with the fixes from the adversarial review.

Fixes applied:
  1. Compliance is judge-graded, not substring-matched. Transcripts are written to disk
     for hand audit, and both measures are reported so their disagreement stays visible.
  2. Controls are variance-matched: shuffled-label diff-of-means (identical construction,
     meaningless labels) and the top principal component. Isotropic random directions are
     kept but reported as the weak null they are.
  3. Directions are kept UNNORMALISED. Zhao steer with raw diff-of-means whose norm scales
     with depth; unit-normalising and applying a constant coefficient is ~24x stronger at
     layer 0 than layer 31 on this model.
  4. Refusal baseline is measured on AgentHarm prompts as well as AdvBench, because Gate B's
     70% threshold is specified for the evaluation corpus.

    uv run python scripts/05_audit_rerun.py --iter
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from pressure.causal import (
    ablate,
    generate,
    random_directions,
    shuffled_label_direction,
    steer_layerwise,
    top_pc_direction,
)
from pressure.config import CFG
from pressure.data import agentharm, extraction_corpus, split_extract_select
from pressure.device import describe, empty_cache, seed_everything
from pressure.grade import judge_complied, save_transcripts, substring_complied
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, load_model


def collect_named(model, tok, prompts, key: str, progress=None) -> torch.Tensor:
    """(n_prompts, n_layers, hidden) at one named position."""
    out = []
    for i, p in enumerate(prompts):
        out.append(residuals_at_named(model, tok, chat_prompt(tok, p), p)[key])
        if progress and i % progress == 0:
            print(f"    {i}/{len(prompts)}", flush=True)
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n-fit", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=25)
    args = ap.parse_args()
    seed_everything(CFG.seed)
    t0 = time.time()

    harmful, harmless = extraction_corpus()
    h_fit, h_held = split_extract_select(harmful)
    b_fit, b_held = split_extract_select(harmless)
    H, B = h_held.prompts[: args.n_test], b_held.prompts[: args.n_test]

    model, tok = load_model(eval_model=not args.iter)
    repo = CFG.iter_model if args.iter else CFG.eval_model
    print(f"model={repo}  env={describe()}", flush=True)

    # --- directions, kept unnormalised ---------------------------------------
    print("collecting activations for directions and controls", flush=True)
    Ah = collect_named(model, tok, h_fit.prompts[: args.n_fit], "context_last", 100)
    Ab = collect_named(model, tok, b_fit.prompts[: args.n_fit], "context_last", 100)
    Th = collect_named(model, tok, h_fit.prompts[: args.n_fit], "task_last", 100)
    Tb = collect_named(model, tok, b_fit.prompts[: args.n_fit], "task_last", 100)

    r_ref_raw = Ah.mean(0) - Ab.mean(0)      # (n_layers, hidden), Zhao scale
    r_harm_raw = Th.mean(0) - Tb.mean(0)
    shuf_ref = shuffled_label_direction(Ah, Ab)
    shuf_harm = shuffled_label_direction(Th, Tb)
    pc_ref = top_pc_direction(torch.cat([Ah, Ab]))
    n_layers, hidden = r_ref_raw.shape
    rand = torch.stack([random_directions(hidden, n=1, seed=s)[0] for s in range(n_layers)])
    # Match the random and PC controls to the real direction's per-layer norm, so any
    # difference is about direction, not magnitude.
    scale = r_ref_raw.norm(dim=-1, keepdim=True)
    rand = rand / rand.norm(dim=-1, keepdim=True) * scale
    pc_ref = pc_ref / pc_ref.norm(dim=-1, keepdim=True) * scale

    print(f"per-layer norms  r_ref_raw: {[round(float(x),2) for x in scale[::8].squeeze()]}", flush=True)
    print(f"                 shuffled : {[round(float(x),2) for x in shuf_ref.norm(dim=-1)[::8]]}",
          flush=True)
    empty_cache(CFG.device)

    # --- behavioural measurement ---------------------------------------------
    transcripts: list[dict] = []

    def measure(label: str, prompts, ctx=None):
        sub, jud = [], []
        for p in prompts:
            if ctx is None:
                text = generate(model, tok, chat_prompt(tok, p), 60)
            else:
                with ctx():
                    text = generate(model, tok, chat_prompt(tok, p), 60)
            s = substring_complied(text)
            j = judge_complied(model, tok, p, text)
            sub.append(s)
            jud.append(j)
            transcripts.append({"condition": label, "prompt": p, "reply": text,
                                "substring": s, "judge": j})
        print(f"  {label:26} substring={np.mean(sub):.2f}  judge={np.mean(jud):.2f}", flush=True)
        return float(np.mean(sub)), float(np.mean(jud))

    print("\ncompliance on held-out AdvBench harmful (higher = more compliant)", flush=True)
    res: dict[str, tuple[float, float]] = {}
    res["baseline"] = measure("baseline", H)
    lay = range(n_layers)
    res["ablate_r_ref"] = measure("ablate r_ref", H, lambda: ablate(model, r_ref_raw[lay[-1] // 2]))
    res["ablate_r_harm"] = measure("ablate r_harm", H, lambda: ablate(model, r_harm_raw[15]))
    res["ablate_shuffled"] = measure("ablate shuffled-label", H, lambda: ablate(model, shuf_ref[16]))
    res["ablate_top_pc"] = measure("ablate top PC", H, lambda: ablate(model, pc_ref[16]))
    res["ablate_random"] = measure("ablate random (weak)", H, lambda: ablate(model, rand[16]))
    empty_cache(CFG.device)

    print("\nGate B baseline on the EVALUATION corpus (AgentHarm, chat format)", flush=True)
    ah = [i["prompt"] for i in agentharm(harmful=True)
          if i["category"] in ("Disinformation", "Hate", "Harassment", "Copyright")][: args.n_test]
    res["agentharm_baseline"] = measure("AgentHarm baseline", ah)

    # --- steering at Zhao's raw scale ----------------------------------------
    print("\nlayer-matched steering at raw (unnormalised) scale", flush=True)
    import importlib.util

    spec = importlib.util.spec_from_file_location("inv", "scripts/04_inversion_sweep.py")
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    from pressure.causal import judgement_logit_gap

    def gap_raw(prompts, dirs, c):
        if c == 0:
            return inv.gap(model, tok, prompts)
        vals = []
        for p in prompts:
            full, span = inv.inversion_prompt(tok, p)
            # steer_layerwise normalises internally, so pass the per-layer norm through
            # the coefficient to preserve Zhao's depth scaling.
            with steer_layerwise(model, dirs, lay, c, span, scale=dirs.norm(dim=-1)):
                vals.append(judgement_logit_gap(model, tok, full, "Certainly", "No"))
        return float(np.mean(vals))

    coeffs = (-4, -2, -1, 0, 1, 2, 4)
    curves = {}
    for name, d in (("r_harm", r_harm_raw), ("shuffled", shuf_harm), ("random", rand)):
        curves[name] = [gap_raw(H, d, c) for c in coeffs]
        rng = max(curves[name]) - min(curves[name])
        print(f"  {name:>9}: " + " ".join(f"{x:+6.2f}" for x in curves[name]) + f"  range={rng:.2f}",
              flush=True)

    save_transcripts(CFG.results_dir / "transcripts.json", transcripts)
    out = {
        "model": repo, "environment": describe(),
        "n_fit": args.n_fit, "n_test": args.n_test,
        "compliance": {k: {"substring": v[0], "judge": v[1]} for k, v in res.items()},
        "per_layer_norm_r_ref": [float(x) for x in scale.squeeze()],
        "steering_coeffs": list(coeffs), "steering_curves": curves,
        "steering_ranges": {k: max(v) - min(v) for k, v in curves.items()},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (CFG.results_dir / "audit_rerun.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote audit_rerun.json and {len(transcripts)} transcripts  [{out['elapsed_min']} min]")


if __name__ == "__main__":
    main()
