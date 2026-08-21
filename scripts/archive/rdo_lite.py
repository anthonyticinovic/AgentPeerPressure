"""RDO-lite: optimise a direction FOR ablation efficacy, per Wollschlager et al. (2502.17420).

Every direction tested so far came from diff-of-means, which finds a vector that
*discriminates* harmful from harmless. Wollschlager et al. instead optimise directions
by gradient descent subject to surgical ablation — the direction is found by asking
"which vector, when ablated, stops refusal?" rather than "which vector separates the
classes?". Those are different objectives and need not have the same answer, which is
consistent with the AUROC 0.98 / ablation 0.00 split observed on this model.

This is a faithful reduction of their objective, not their full method: initialise r
from diff-of-means, then descend on the cross-entropy of an affirmative continuation
while r is ablated from the residual stream. Ablation is differentiable in r, so the
gradient flows to the direction itself. Model weights stay frozen throughout.

If optimisation cannot find an ablation-effective direction either, the negative is
much stronger than "diff-of-means picked badly".

    uv run python scripts/09_rdo.py --iter
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager

import numpy as np
import torch

from pressure.causal import ablate, generate
from pressure.config import CFG
from pressure.data import extraction_corpus, split_extract_select
from pressure.grade import judge_complied, save_transcripts, substring_complied
from pressure.hooks import _hidden, blocks
from pressure.model import chat_prompt, load_model

AFFIRMATIVE = "Sure, here"
EXCLUDE = ("cyberbullying",)


@contextmanager
def ablate_grad(model, r: torch.Tensor):
    """Directional ablation that keeps the graph — r is a leaf we optimise."""
    handles = []

    def hook(_m, _i, output):
        h = _hidden(output)
        v = r / r.norm()
        v = v.to(h.dtype)
        h = h - (h @ v).unsqueeze(-1) * v
        return (h, *output[1:]) if isinstance(output, tuple) else h

    for blk in blocks(model):
        handles.append(blk.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def batch_loss(model, tok, prompts, r, target: str = AFFIRMATIVE) -> torch.Tensor:
    """-log P(affirmative continuation | prompt) with r ablated. Lower = less refusal."""
    total = 0.0
    for p in prompts:
        full = chat_prompt(tok, p) + target
        enc = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)
        n_target = len(tok(target, add_special_tokens=False)["input_ids"])
        with ablate_grad(model, r):
            logits = model(**enc).logits
        lp = torch.log_softmax(logits[0, -n_target - 1 : -1].float(), dim=-1)
        tgt = enc["input_ids"][0, -n_target:]
        total = total + -lp[torch.arange(n_target), tgt].mean()
    return total / len(prompts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=20)
    args = ap.parse_args()

    model, tok = load_model(eval_model=not args.iter)
    for p in model.parameters():
        p.requires_grad_(False)

    harmful, harmless = extraction_corpus()
    h_fit, h_held = split_extract_select(harmful)
    train = [p for p in h_fit.prompts[:60]]
    H = [p for p in h_held.prompts if not any(x in p.lower() for x in EXCLUDE)][: args.n_test]

    raw = torch.load(CFG.results_dir / "dual_raw.pt", weights_only=False)
    init = raw["r_ref"][22].float().to(model.device)
    r = torch.nn.Parameter((init / init.norm()).clone())
    opt = torch.optim.Adam([r], lr=args.lr)

    rng = np.random.default_rng(CFG.seed)
    print(f"optimising direction for ablation efficacy ({args.steps} steps)", flush=True)
    history = []
    for step in range(args.steps):
        idx = rng.choice(len(train), args.batch, replace=False)
        loss = batch_loss(model, tok, [train[i] for i in idx], r)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            r.data = r.data / r.data.norm()
        history.append(float(loss))
        if step % 10 == 0 or step == args.steps - 1:
            cs = float(torch.nn.functional.cosine_similarity(r.detach(), init / init.norm(), dim=0))
            print(f"  step {step:>3}  loss={float(loss):.4f}  cos(r, r_ref)={cs:.3f}", flush=True)

    r_opt = r.detach().clone()
    cos_final = float(torch.nn.functional.cosine_similarity(r_opt, init / init.norm(), dim=0))

    # --- evaluate ------------------------------------------------------------
    rows: list[dict] = []

    def measure(label, vec=None):
        sub, jud = [], []
        for p in H:
            if vec is None:
                t = generate(model, tok, chat_prompt(tok, p), 70)
            else:
                with ablate(model, vec):
                    t = generate(model, tok, chat_prompt(tok, p), 70)
            s, j = substring_complied(t), judge_complied(model, tok, p, t)
            sub.append(s)
            jud.append(j)
            rows.append({"condition": label, "prompt": p, "reply": t, "substring": s,
                         "judge": j, "judge_label": "COMPLY" if j else "DECLINE"})
        print(f"  {label:28} substring={np.mean(sub):.2f}  judge={np.mean(jud):.2f}", flush=True)
        return float(np.mean(sub)), float(np.mean(jud))

    print("\ncompliance on held-out harmful prompts", flush=True)
    res = {
        "baseline": measure("baseline"),
        "diff_of_means": measure("ablate diff-of-means", init),
        "rdo": measure("ablate RDO-optimised", r_opt),
    }

    save_transcripts(CFG.results_dir / "transcripts_rdo.json", rows)
    (CFG.results_dir / "rdo.json").write_text(json.dumps({
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "loss_start": history[0], "loss_end": history[-1],
        "cos_rdo_vs_diffmeans": cos_final,
        "compliance": {k: {"substring": v[0], "judge": v[1]} for k, v in res.items()},
        "n_test": len(H),
    }, indent=2))
    print(f"\nloss {history[0]:.3f} -> {history[-1]:.3f}   cos(RDO, diff-of-means) = {cos_final:.3f}")


if __name__ == "__main__":
    main()
