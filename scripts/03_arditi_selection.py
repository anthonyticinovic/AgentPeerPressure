"""Arditi's direction selection, run properly: the full (position x layer) grid.

Supersedes archive/arditi_sweep_layers_only.py, which was wrong in four ways. It swept layers only at
a single token position; it scored candidates by judge-graded generation instead of
the refusal metric; it applied none of the three admissibility filters; and it treated
"best judge score" as the criterion rather than "minimum bypass_score among admissible
candidates". Its conclusion — that no diff-of-means direction ablates refusal — was
drawn from 32 candidates out of 288.

Protocol (2406.11717 sections 2.3, 2.4, B, C.1):

  train      128 harmful / 128 harmless   -> mean activations, hence candidates
  val         32 harmful /  32 harmless   -> the three scores
  candidates  9 post-instruction positions x 32 layers
  score       refusal metric = log-odds that the first generated token is in R
  select      min bypass_score s.t. induce_score > 0, kl_score < 0.1, l < 0.8L

Validation sets are pre-filtered on the refusal metric as the paper specifies: harmful
prompts the model does not refuse, and harmless prompts it does, carry no signal about
whether an intervention changed refusal.

    uv run python scripts/03_arditi_selection.py --iter
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from pressure.arditi import (
    PAPER_QWEN_REFUSAL_PHRASES,
    Candidate,
    _first_token_id,
    add_activation,
    derive_refusal_tokens,
    kl_divergence,
    next_token_probs,
    refusal_metric,
    select,
)
from pressure.causal import ablate_all_components, generate
from pressure.config import CFG
from pressure.data import extraction_corpus, split_extract_select
from pressure.grade import judge_complied, save_transcripts, substring_complied
from pressure.hooks import resolve_positions, residuals_at
from pressure.model import chat_prompt, layer_types, load_model, n_layers

EXCLUDE = ("cyberbullying",)  # ambiguous prompt, confirmed by hand review


def post_instruction_offsets(tok, prompt: str, task: str) -> tuple[int, ...]:
    """End-relative offsets of every token after the user instruction.

    Arditi use |I| = 5 because that is the whole post-instruction template for
    LLAMA-3. The quantity is "post-instruction positions", not the number five, so
    it is measured here rather than assumed.
    """
    pos = resolve_positions(tok, prompt, task)
    n_post = pos["context_last"] - pos["task_last"]
    return tuple(range(-1, -n_post - 1, -1))


def mean_activations(model, tok, prompts, offsets) -> torch.Tensor:
    """(n_offsets, n_layers, hidden) mean residual stream over `prompts`."""
    total = None
    for p in prompts:
        a = residuals_at(model, tok, chat_prompt(tok, p), offsets)  # (layers, pos, hid)
        total = a if total is None else total + a
    return (total / len(prompts)).permute(1, 0, 2).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-gen", type=int, default=25, help="prompts for the final generation check")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model, tok = load_model(eval_model=not args.iter)
    L = n_layers(model)
    types = layer_types(model)
    print(f"model={CFG.iter_model if args.iter else CFG.eval_model}  layers={L}", flush=True)

    harmful, harmless = extraction_corpus()
    h_fit, h_held = split_extract_select(harmful)
    b_fit, b_held = split_extract_select(harmless)
    train_h = h_fit.prompts[: args.n_train]
    train_b = b_fit.prompts[: args.n_train]
    pool_h = [p for p in h_held.prompts if not any(x in p.lower() for x in EXCLUDE)]
    pool_b = list(b_held.prompts)

    # --- R: the tokens this model uses to open a refusal ----------------------
    R, token_table = derive_refusal_tokens(
        model, tok, [chat_prompt(tok, p) for p in train_h], [chat_prompt(tok, p) for p in train_b]
    )
    print("\ntop next-token gaps (harmful minus harmless), train sets:", flush=True)
    for row in token_table:
        print(f"  {row['token']!r:<14} id={row['id']:<8} "
              f"harmful={row['p_harmful']:.3f}  harmless={row['p_harmless']:.3f}", flush=True)
    paper_ids = [_first_token_id(tok, w) for w in PAPER_QWEN_REFUSAL_PHRASES]
    print(f"R (derived) = {R} -> {[tok.decode([i]) for i in R]}", flush=True)
    print(f"R (paper's QWEN row, retokenised) = {paper_ids} -> "
          f"{list(PAPER_QWEN_REFUSAL_PHRASES)}", flush=True)

    # --- validation sets, filtered on the refusal metric ----------------------
    mh = refusal_metric(next_token_probs(model, tok, [chat_prompt(tok, p) for p in pool_h],
                                         args.batch), R)
    mb = refusal_metric(next_token_probs(model, tok, [chat_prompt(tok, p) for p in pool_b],
                                         args.batch), R)
    keep_h = [p for p, m in zip(pool_h, mh.tolist()) if m > 0]
    keep_b = [p for p, m in zip(pool_b, mb.tolist()) if m < 0]
    val_h, val_b = keep_h[: args.n_val], keep_b[: args.n_val]
    print(f"\nval harmful  {len(keep_h)}/{len(pool_h)} pass the filter (refusal_metric > 0), "
          f"using {len(val_h)}", flush=True)
    print(f"val harmless {len(keep_b)}/{len(pool_b)} pass the filter (refusal_metric < 0), "
          f"using {len(val_b)}", flush=True)
    if len(val_h) < min(8, args.n_val) or len(val_b) < min(8, args.n_val):
        raise RuntimeError("validation set too small after filtering; R is probably wrong")

    Ph = [chat_prompt(tok, p) for p in val_h]
    Pb = [chat_prompt(tok, p) for p in val_b]
    base_h = float(refusal_metric(next_token_probs(model, tok, Ph, args.batch), R).mean())
    base_b_probs = next_token_probs(model, tok, Pb, args.batch)
    base_b = float(refusal_metric(base_b_probs, R).mean())
    print(f"baseline refusal metric: harmful {base_h:+.3f}   harmless {base_b:+.3f}", flush=True)

    # --- candidate grid -------------------------------------------------------
    offsets = post_instruction_offsets(tok, chat_prompt(tok, train_h[0]), train_h[0])
    print(f"\npost-instruction positions: {offsets}", flush=True)
    print(f"  {[tok.decode([tok(chat_prompt(tok, train_h[0]), add_special_tokens=False)['input_ids'][o]]) for o in offsets]}",
          flush=True)

    print(f"collecting mean activations over {len(train_h)}+{len(train_b)} prompts", flush=True)
    mu = mean_activations(model, tok, train_h, offsets)
    nu = mean_activations(model, tok, train_b, offsets)
    Rvec = mu - nu  # (n_pos, n_layers, hidden), unnormalised: eq. 3 needs the magnitude

    n_cand = Rvec.shape[0] * L
    print(f"\nscoring {n_cand} candidates ({Rvec.shape[0]} positions x {L} layers)", flush=True)
    t0 = time.time()
    cands: list[Candidate] = []
    for pi, off in enumerate(offsets):
        for layer in range(L):
            r = Rvec[pi, layer]
            with ablate_all_components(model, r):
                bypass = float(refusal_metric(next_token_probs(model, tok, Ph, args.batch), R).mean())
                pb_abl = next_token_probs(model, tok, Pb, args.batch)
            with add_activation(model, r, layer):
                induce = float(refusal_metric(next_token_probs(model, tok, Pb, args.batch), R).mean())
            kl = float(kl_divergence(base_b_probs, pb_abl).mean())
            cands.append(Candidate(off, layer, bypass, induce, kl))
        done = (pi + 1) * L
        print(f"  position {off:>3}: done {done}/{n_cand}  ({time.time() - t0:.0f}s)", flush=True)

    star = select(cands, L)
    n_ok = sum(c.admissible(L) for c in cands)
    best_unfiltered = min(cands, key=lambda c: c.bypass)

    print(f"\n{n_ok}/{n_cand} candidates admissible "
          f"(induce > 0, kl < 0.1, layer < {0.8 * L:.1f})", flush=True)
    print(f"unfiltered minimum bypass: pos {best_unfiltered.position} layer "
          f"{best_unfiltered.layer}  bypass={best_unfiltered.bypass:+.3f} "
          f"induce={best_unfiltered.induce:+.3f} kl={best_unfiltered.kl:.3f}", flush=True)
    if star is None:
        print("NO ADMISSIBLE CANDIDATE. Arditi's procedure returns nothing on this model.",
              flush=True)
    else:
        print(f"SELECTED  i*={star.position}  l*={star.layer}/{L} ({types[star.layer]})  "
              f"bypass={star.bypass:+.3f}  induce={star.induce:+.3f}  kl={star.kl:.3f}",
              flush=True)

    top = sorted([c for c in cands if c.admissible(L)], key=lambda c: c.bypass)[:10]
    print("\ntop 10 admissible by bypass_score:", flush=True)
    print(f"{'pos':>4} {'layer':>6} {'bypass':>9} {'induce':>9} {'kl':>7}", flush=True)
    for c in top:
        print(f"{c.position:>4} {c.layer:>6} {c.bypass:>+9.3f} {c.induce:>+9.3f} {c.kl:>7.3f}",
              flush=True)

    # --- what the earlier, AUROC-based selection picked ------------------------
    prev = next((c for c in cands if c.position == -1 and c.layer == 22), None)
    if prev is not None:
        print(f"\nAUROC-selected direction (pos -1, layer 22): bypass={prev.bypass:+.3f} "
              f"induce={prev.induce:+.3f} kl={prev.kl:.3f}  "
              f"admissible={prev.admissible(L)}", flush=True)

    # --- generation check on the selected direction ----------------------------
    rows: list[dict] = []
    compliance: dict[str, dict] = {}
    if star is not None:
        gen_prompts = val_h[: args.n_gen]
        r_star = Rvec[offsets.index(star.position), star.layer]

        def measure(label, vec=None):
            sub, jud = [], []
            for p in gen_prompts:
                if vec is None:
                    text = generate(model, tok, chat_prompt(tok, p), 60)
                else:
                    with ablate_all_components(model, vec):
                        text = generate(model, tok, chat_prompt(tok, p), 60)
                s, j = substring_complied(text), judge_complied(model, tok, p, text)
                sub.append(s)
                jud.append(j)
                rows.append({"condition": label, "prompt": p, "reply": text,
                             "substring": s, "judge": j,
                             "judge_label": "COMPLY" if j else "DECLINE"})
            print(f"  {label:38} substring={np.mean(sub):.2f}  judge={np.mean(jud):.2f}",
                  flush=True)
            return {"substring": float(np.mean(sub)), "judge": float(np.mean(jud))}

        print(f"\ngeneration check on {len(gen_prompts)} held-out harmful prompts", flush=True)
        compliance["baseline"] = measure("baseline")
        compliance["selected"] = measure(
            f"ablate r(i*={star.position}, l*={star.layer})", r_star
        )
        if prev is not None:
            compliance["auroc_selected"] = measure(
                "ablate r(pos -1, layer 22)  [old choice]", Rvec[offsets.index(-1), 22]
            )
        save_transcripts(CFG.results_dir / "transcripts_arditi_selection.json", rows)

    payload = {
        "n_layers": L,
        "positions": list(offsets),
        "refusal_tokens": {"derived": R, "decoded": [tok.decode([i]) for i in R],
                           "paper_qwen_retokenised": paper_ids, "top_gaps": token_table},
        "n_train": {"harmful": len(train_h), "harmless": len(train_b)},
        "n_val": {"harmful": len(val_h), "harmless": len(val_b)},
        "baseline_refusal_metric": {"harmful": base_h, "harmless": base_b},
        "n_admissible": n_ok,
        "selected": None if star is None else star.__dict__,
        "best_unfiltered": best_unfiltered.__dict__,
        "auroc_choice": None if prev is None else prev.__dict__,
        "compliance": compliance,
        "grid": [c.__dict__ for c in cands],
    }
    # The vector itself, stamped with the model that produced it: the JSON records
    # only (i*, l*), which is not enough to reconstruct r* on a different model.
    if star is not None:
        torch.save(
            {
                "r_arditi": Rvec[offsets.index(star.position), star.layer],
                "position": star.position,
                "layer": star.layer,
                "offsets": list(offsets),
                "model": CFG.iter_model if args.iter else CFG.eval_model,
            },
            CFG.results_dir / "arditi_selected.pt",
        )

    (CFG.results_dir / "arditi_selection.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote arditi_selection.json ({time.time() - t0:.0f}s scoring)", flush=True)


if __name__ == "__main__":
    main()
