"""Stages 1-3: Zhao's reply-inversion sweep — does r_harm reverse the model's judgement?

Protocol is Zhao et al. (2507.11878) as implemented in `vendor/zhao/src/intervention.py`
and `complete_intervene.sh`: activation addition via a forward PRE-hook on one layer at a
time, raw (unnormalised) per-layer diff-of-means scaled by a coefficient, applied for the
prefill pass only, greedy decoding, scored by substring on the completion.

Their Figure 5 contrast, with our causal arm added:

    harmless instructions   +r_harm   +r_ref   -r_ref   +r_arditi
    harmful  instructions   -r_harm   +r_ref   -r_ref   -r_arditi

The prediction under test: steering the *harmfulness* direction reverses the judgement
(harmless -> 'Certainly', harmful -> 'No'), while steering the *refusal* direction does
not, because it carries surface refusal signals rather than the belief.

Span handling follows their Appendix E.1 and is not symmetric: the harmfulness direction
is applied to the instruction tokens only, the refusal directions to all tokens. Applying
the refusal direction to instruction tokens alone yields ~zero effect in their Figure 10,
which would masquerade as a dissociation.

Pre-registered: the primary arms are r_harm and r_ref under Zhao's exact configuration.
Every other arm is reported but secondary, and the coefficient is calibrated on the
SELECTION split before any held-out prompt is scored.

    uv run python scripts/06_inversion_sweep.py --iter --stage calibrate
    uv run python scripts/06_inversion_sweep.py --iter --stage sweep
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from pressure.causal import random_directions, shuffled_label_direction, top_pc_direction
from pressure.config import CFG
from pressure.data import extraction_corpus, split_extract_select
from pressure.inversion import (
    JUDGED_HARMFUL,
    JUDGED_HARMLESS,
    REFUSED,
    UNPARSED,
    add_direction,
    build_batch,
    build_prompt,
    generate_batch,
    judge_inversion,
    judge_inversion_strict,
    load_inversion_prompts,
    save_json,
)
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, load_model, n_layers

EXCLUDE = ("cyberbullying",)
DIRS_CACHE = "inversion_dirs.pt"


def collect(model, tok, prompts, key):
    return torch.stack(
        [residuals_at_named(model, tok, chat_prompt(tok, p), p)[key] for p in prompts]
    )


def build_directions(model, tok, n_fit: int):
    """Raw per-layer diff-of-means, plus nulls, at both token positions.

    Raw, not unit: Zhao's coefficient is calibrated against the unnormalised scale.
    Position 'task_last' is their t_inst (mode_dir='hf'); 'context_last' is t_post-inst
    (mode_dir='refuse'). The Arditi arm reuses the position that won Gate B.
    """
    path = CFG.results_dir / DIRS_CACHE
    if path.exists():
        return torch.load(path, weights_only=False)

    harmful, harmless = extraction_corpus()
    h_fit, _ = split_extract_select(harmful)
    b_fit, _ = split_extract_select(harmless)
    H, B = h_fit.prompts[:n_fit], b_fit.prompts[:n_fit]

    print(f"extracting raw directions over {len(H)}+{len(B)} prompts", flush=True)
    Th, Tb = collect(model, tok, H, "task_last"), collect(model, tok, B, "task_last")
    Ch, Cb = collect(model, tok, H, "context_last"), collect(model, tok, B, "context_last")

    sel = json.loads((CFG.results_dir / "arditi_selection.json").read_text())
    star = sel["selected"]
    offsets = tuple(sel["positions"])
    from pressure.hooks import residuals_at

    def mean_at(prompts, off_idx):
        tot = None
        for p in prompts:
            a = residuals_at(model, tok, chat_prompt(tok, p), offsets)  # (layers,pos,hid)
            tot = a if tot is None else tot + a
        return (tot / len(prompts))[:, off_idx, :]

    oi = offsets.index(star["position"])
    r_arditi = mean_at(H, oi) - mean_at(B, oi)

    blob = {
        "r_harm": Th.mean(0) - Tb.mean(0),
        "r_ref": Ch.mean(0) - Cb.mean(0),
        "r_arditi": r_arditi,
        "shuffled": shuffled_label_direction(Ch, Cb),
        "top_pc": top_pc_direction(torch.cat([Ch, Cb])),
        "arditi_layer": star["layer"],
        "arditi_position": star["position"],
        "n_fit": len(H),
    }
    # Random nulls matched to r_ref's per-layer norm, so "beats random" is not vacuous.
    norms = blob["r_ref"].norm(dim=-1, keepdim=True)
    rnd = torch.stack(random_directions(blob["r_ref"].shape[-1], n=5, seed=CFG.seed))
    blob["random"] = rnd.unsqueeze(1) * norms.unsqueeze(0)  # (5, layers, hidden)
    torch.save(blob, path)
    return blob


def run_cell(model, tok, prompts, question, vec, layer, coeff, context_only, max_new,
             batch, decode_steps):
    """One (direction, layer, sign) cell over a set of prompts. Returns judgement rows."""
    rows = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        instrs = [build_prompt(p, question) for p in chunk]
        rendered = [chat_prompt(tok, s) for s in instrs]
        enc, mask = build_batch(tok, rendered, instrs, context_only=context_only)
        if vec is None:
            texts = generate_batch(model, tok, enc, max_new)
        else:
            with add_direction(model, vec[layer], layer, coeff, mask, decode_steps):
                texts = generate_batch(model, tok, enc, max_new)
        for p, t in zip(chunk, texts):
            rows.append({"prompt": p, "reply": t,
                         "zhao": judge_inversion(t), "strict": judge_inversion_strict(t)})
    return rows


def summarise(rows, target):
    """Fraction judged `target`, plus the diagnostic buckets."""
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "zhao_target": sum(r["zhao"] == target for r in rows) / n,
        "strict_target": sum(r["strict"] == target for r in rows) / n,
        "unparsed": sum(r["zhao"] == UNPARSED for r in rows) / n,
        "refused": sum(r["strict"] == REFUSED for r in rows) / n,
        "disagree": sum(r["zhao"] != r["strict"] for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--stage", choices=("calibrate", "sweep"), default="sweep")
    ap.add_argument("--n", type=int, default=50, help="held-out items per class")
    ap.add_argument("--n-fit", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--decode-steps", type=int, default=1)
    ap.add_argument("--template", type=int, default=None, help="default: pre-flight winner")
    ap.add_argument("--coeff", type=float, default=None, help="default: calibrated value")
    ap.add_argument("--coeffs", default="1,2,4", help="calibrate stage only")
    ap.add_argument("--cal-layers", default="8,12,16,20", help="calibrate stage only")
    args = ap.parse_args()

    model, tok = load_model(eval_model=not args.iter)
    L = n_layers(model)
    templates = load_inversion_prompts()

    pre = json.loads((CFG.results_dir / "inversion_preflight.json").read_text())
    t_idx = args.template if args.template is not None else pre["best"]["idx"]
    question = templates[t_idx]
    print(f"template {t_idx}: {question}", flush=True)

    D = build_directions(model, tok, args.n_fit)
    harmful, harmless = extraction_corpus()
    h_fit, h_sel = split_extract_select(harmful)
    b_fit, b_sel = split_extract_select(harmless)

    # ---- Stage 1: calibrate the coefficient on the SELECTION split ------------
    if args.stage == "calibrate":
        H = [p for p in h_sel.prompts if not any(x in p.lower() for x in EXCLUDE)][:12]
        B = list(b_sel.prompts)[:12]
        layers = [int(x) for x in args.cal_layers.split(",")]
        out = []
        print(f"\ncalibrating on selection split, layers {layers}", flush=True)
        print(f"{'coeff':>6} {'layer':>6} {'harmless->Certainly':>20} {'unparsed':>9}", flush=True)
        for c in [float(x) for x in args.coeffs.split(",")]:
            for lay in layers:
                rows = run_cell(model, tok, B, question, D["r_harm"], lay, c, True,
                                args.max_new, args.batch, args.decode_steps)
                s = summarise(rows, JUDGED_HARMFUL)
                out.append({"coeff": c, "layer": lay, **s})
                print(f"{c:>6.1f} {lay:>6} {s['zhao_target']:>20.2f} {s['unparsed']:>9.2f}",
                      flush=True)
        best = max(out, key=lambda r: (r["zhao_target"], -r["unparsed"]))
        print(f"\nbest coeff {best['coeff']} (layer {best['layer']}, "
              f"flip {best['zhao_target']:.2f})", flush=True)
        save_json(CFG.results_dir / "inversion_calibration.json",
                  {"template_idx": t_idx, "rows": out, "best": best})
        return

    # ---- Stages 2-3: the layer sweep on held-out prompts ----------------------
    cal_path = CFG.results_dir / "inversion_calibration.json"
    coeff = args.coeff if args.coeff is not None else json.loads(
        cal_path.read_text())["best"]["coeff"]
    print(f"coefficient {coeff} (frozen)", flush=True)

    H = [p for p in h_fit.prompts if not any(x in p.lower() for x in EXCLUDE)][: args.n]
    B = list(b_fit.prompts)[: args.n]
    print(f"held-out: {len(H)} harmful, {len(B)} harmless\n", flush=True)

    # (name, vector, sign, context_only) per Zhao's Fig 5 + our Arditi arm.
    arms_harmless = [
        ("r_harm+", D["r_harm"], +1, True),
        ("r_ref+", D["r_ref"], +1, False),
        ("r_ref-", D["r_ref"], -1, False),
        ("r_arditi+", D["r_arditi"], +1, False),
    ]
    arms_harmful = [
        ("r_harm-", D["r_harm"], -1, True),
        ("r_ref+", D["r_ref"], +1, False),
        ("r_ref-", D["r_ref"], -1, False),
        ("r_arditi-", D["r_arditi"], -1, False),
    ]

    results = {"template_idx": t_idx, "question": question, "coeff": coeff,
               "n_layers": L, "decode_steps": args.decode_steps,
               "max_new_tokens": args.max_new,
               "arditi_layer": int(D["arditi_layer"]),
               "arditi_position": int(D["arditi_position"]),
               "panels": {}}
    transcripts = []

    for panel, prompts, arms, target in (
        ("harmless", B, arms_harmless, JUDGED_HARMFUL),   # success = flip No -> Certainly
        ("harmful", H, arms_harmful, JUDGED_HARMLESS),    # success = flip Certainly -> No
    ):
        base = run_cell(model, tok, prompts, question, None, 0, 0.0, False,
                        args.max_new, args.batch, args.decode_steps)
        bs = summarise(base, target)
        results["panels"][panel] = {"baseline": bs, "arms": {}}
        print(f"[{panel}] baseline flip-rate {bs['zhao_target']:.2f} "
              f"unparsed {bs['unparsed']:.2f}", flush=True)

        for name, vec, sign, ctx in arms:
            series = []
            t0 = time.time()
            for lay in range(L):
                rows = run_cell(model, tok, prompts, question, sign * vec, lay, coeff,
                                ctx, args.max_new, args.batch, args.decode_steps)
                s = summarise(rows, target)
                series.append({"layer": lay, **s})
                for r in rows:
                    transcripts.append({"panel": panel, "arm": name, "layer": lay, **r})
            results["panels"][panel]["arms"][name] = series
            peak = max(series, key=lambda r: r["zhao_target"])
            print(f"  {name:>11}  peak flip {peak['zhao_target']:.2f} @L{peak['layer']:<2} "
                  f"(strict {peak['strict_target']:.2f}, refused {peak['refused']:.2f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        save_json(CFG.results_dir / "inversion_sweep.json", results)

    save_json(CFG.results_dir / "transcripts_inversion.json", transcripts)
    print(f"\nwrote inversion_sweep.json and {len(transcripts)} transcripts", flush=True)


if __name__ == "__main__":
    main()
