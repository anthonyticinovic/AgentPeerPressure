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
from contextlib import nullcontext

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
    instruction_clause,
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
             batch, decode_steps, gap_ids=None):
    """One (direction, layer, sign) cell over a set of prompts. Returns judgement rows.

    `gap_ids` = (certainly_id, no_id) additionally records log P(Certainly) - log P(No)
    at the answer position. The binary flip is what Zhao report, but it only moves once
    the steer overcomes the model's answer-token prior; the gap shows whether the
    judgement is shifting at all, which distinguishes "no effect" from "not enough".
    """
    rows = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        rendered = [chat_prompt(tok, build_prompt(p, question)) for p in chunk]
        spans = [instruction_clause(p) for p in chunk]
        enc, mask = build_batch(tok, rendered, spans, context_only=context_only)
        def steer():
            # A fresh context per forward: the hook counts decoding steps, so reusing an
            # exhausted one would silently measure the gap with no intervention applied.
            return (add_direction(model, vec[layer], layer, coeff, mask, decode_steps)
                    if vec is not None else nullcontext())

        with steer():
            texts = generate_batch(model, tok, enc, max_new)
        if gap_ids:
            with steer():
                gaps = answer_gap(model, tok, enc, *gap_ids)
        else:
            gaps = [None] * len(chunk)
        for p, t, g in zip(chunk, texts, gaps):
            rows.append({"prompt": p, "reply": t, "gap": g,
                         "zhao": judge_inversion(t), "strict": judge_inversion_strict(t)})
    return rows


@torch.no_grad()
def answer_gap(model, tok, enc, yes_id: int, no_id: int) -> list[float]:
    """log P('Certainly') - log P('No') at the first generated position, under any
    interventions currently hooked."""
    e = {k: v.to(model.device) for k, v in enc.items()}
    lp = torch.log_softmax(model(**e).logits[:, -1, :].float(), dim=-1)
    return (lp[:, yes_id] - lp[:, no_id]).cpu().tolist()


def summarise(rows, target):
    """Fraction judged `target`, plus the diagnostic buckets.

    `zhao_target` keeps unparsed in its denominator deliberately: it reproduces Zhao's
    own published metric verbatim, substring-scored over all attempts. `strict_target`
    is ours, so it excludes REFUSED and UNPARSED from its denominator -- both are
    declines, not the negative verdict, and counting either would deflate the rate
    exactly at the layers most likely to break the yes/no answer.
    """
    n = max(len(rows), 1)
    judged = [r for r in rows if r["strict"] != UNPARSED and r["strict"] != REFUSED]
    return {
        "n": len(rows),
        "zhao_target": sum(r["zhao"] == target for r in rows) / n,
        "strict_target": sum(r["strict"] == target for r in judged) / max(len(judged), 1),
        "unparsed": sum(r["zhao"] == UNPARSED for r in rows) / n,
        "refused": sum(r["strict"] == REFUSED for r in rows) / n,
        "disagree": sum(r["zhao"] != r["strict"] for r in rows) / n,
        "mean_gap": float(np.mean([r["gap"] for r in rows if r["gap"] is not None]))
        if any(r["gap"] is not None for r in rows) else None,
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

    gap_ids = (tok("Certainly", add_special_tokens=False)["input_ids"][0],
               tok("No", add_special_tokens=False)["input_ids"][0])

    D = build_directions(model, tok, args.n_fit)
    harmful, harmless = extraction_corpus()
    _, h_sel = split_extract_select(harmful)
    _, b_sel = split_extract_select(harmless)

    # ---- Stage 1: calibrate the coefficient on the SELECTION split ------------
    # Calibration takes the TAIL of h_sel/b_sel and the sweep stage below takes the
    # HEAD (`[:args.n]`), so the 12 prompts the coefficient is grid-searched against
    # are disjoint from the "held-out" prompts it's later scored on. A cold review
    # found both stages had been slicing the same [:12] prefix -- the coefficient
    # picked here would then be re-tested on the same 12 items it was chosen to work
    # well on, 24% of a default 50-item sweep.
    if args.stage == "calibrate":
        cal_reserve = 12
        h_sel_filtered = [p for p in h_sel.prompts if not any(x in p.lower() for x in EXCLUDE)]
        if args.n + cal_reserve > min(len(h_sel_filtered), len(b_sel.prompts)):
            raise SystemExit(
                f"--n {args.n} leaves no room for {cal_reserve} disjoint calibration "
                f"prompts out of {len(h_sel_filtered)}/{len(b_sel.prompts)} in the "
                "selection split"
            )
        H = h_sel_filtered[-cal_reserve:]
        B = list(b_sel.prompts)[-cal_reserve:]
        layers = [int(x) for x in args.cal_layers.split(",")]
        out = []
        print(f"\ncalibrating on selection split, layers {layers}", flush=True)
        base = run_cell(model, tok, B, question, None, 0, 0.0, False,
                        args.max_new, args.batch, args.decode_steps, gap_ids)
        bs = summarise(base, JUDGED_HARMFUL)
        print(f"baseline: flip {bs['zhao_target']:.2f}  gap {bs['mean_gap']:+.2f}\n", flush=True)
        print(f"{'coeff':>6} {'layer':>6} {'harmless->Certainly':>20} {'gap':>8} "
              f"{'unparsed':>9}", flush=True)
        for c in [float(x) for x in args.coeffs.split(",")]:
            for lay in layers:
                rows = run_cell(model, tok, B, question, D["r_harm"], lay, c, True,
                                args.max_new, args.batch, args.decode_steps, gap_ids)
                s = summarise(rows, JUDGED_HARMFUL)
                out.append({"coeff": c, "layer": lay, **s})
                print(f"{c:>6.1f} {lay:>6} {s['zhao_target']:>20.2f} "
                      f"{s['mean_gap']:>+8.2f} {s['unparsed']:>9.2f}", flush=True)
        best = max(out, key=lambda r: (r["zhao_target"], -r["unparsed"]))
        print(f"\nbest coeff {best['coeff']} (layer {best['layer']}, "
              f"flip {best['zhao_target']:.2f})", flush=True)
        save_json(CFG.results_dir / "inversion_calibration.json",
                  {"template_idx": t_idx, "baseline": bs, "rows": out, "best": best})
        return

    # ---- Stages 2-3: the layer sweep on held-out prompts ----------------------
    cal_path = CFG.results_dir / "inversion_calibration.json"
    coeff = args.coeff if args.coeff is not None else json.loads(
        cal_path.read_text())["best"]["coeff"]
    print(f"coefficient {coeff} (frozen)", flush=True)

    # h_sel/b_sel, not h_fit/b_fit: build_directions() already fit r_harm/r_ref/r_arditi
    # on the FIT split. Scoring the sweep on the same split it was fit on is leakage —
    # every "held-out" prompt here was inside the fitting set.
    H = [p for p in h_sel.prompts if not any(x in p.lower() for x in EXCLUDE)][: args.n]
    B = list(b_sel.prompts)[: args.n]
    print(f"held-out: {len(H)} harmful, {len(B)} harmless\n", flush=True)

    # (name, vector, sign, context_only) per Zhao's Fig 5 + our Arditi arm.
    arms_harmless = [
        ("r_harm+", D["r_harm"], +1, True),
        ("r_harm-", D["r_harm"], -1, True),
        ("r_ref+", D["r_ref"], +1, False),
        ("r_ref-", D["r_ref"], -1, False),
        ("r_arditi+", D["r_arditi"], +1, False),
    ]
    arms_harmful = [
        ("r_harm-", D["r_harm"], -1, True),
        # Same sign as the harmless panel, so the content gap between panels is
        # measurable for r_harm as it is for r_ref. Zhao only run one sign per panel.
        ("r_harm+", D["r_harm"], +1, True),
        ("r_ref+", D["r_ref"], +1, False),
        ("r_ref-", D["r_ref"], -1, False),
        ("r_arditi-", D["r_arditi"], -1, False),
    ]

    results = {"model": CFG.iter_model if args.iter else CFG.eval_model,
               "template_idx": t_idx, "question": question, "coeff": coeff,
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
                        args.max_new, args.batch, args.decode_steps, gap_ids)
        bs = summarise(base, target)
        results["panels"][panel] = {"baseline": bs, "arms": {}}
        print(f"[{panel}] baseline flip-rate {bs['zhao_target']:.2f} "
              f"unparsed {bs['unparsed']:.2f}", flush=True)

        for name, vec, sign, ctx in arms:
            series = []
            t0 = time.time()
            for lay in range(L):
                rows = run_cell(model, tok, prompts, question, sign * vec, lay, coeff,
                                ctx, args.max_new, args.batch, args.decode_steps, gap_ids)
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
