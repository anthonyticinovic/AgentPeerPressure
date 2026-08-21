"""Arditi's selection criterion: pick the direction by CAUSAL effect, not by AUROC.

Arditi et al. generate a candidate direction at every (layer, position), then select
the one whose ablation most reduces refusal on a validation set. They note explicitly
that the best-separating direction is not the best-ablating one.

Every earlier result in this project tested exactly one candidate — the AUROC winner —
and concluded from it that ablation does not work. This sweeps all 32 candidates.

    uv run python scripts/08_arditi_sweep.py --iter
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from pressure.causal import ablate, generate
from pressure.config import CFG
from pressure.data import extraction_corpus, split_extract_select
from pressure.grade import judge_complied, substring_complied
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, load_model

EXCLUDE = ("cyberbullying",)   # ambiguous prompt, confirmed by hand review

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--n-fit", type=int, default=200)
    a = ap.parse_args()
    model, tok = load_model(eval_model=not a.iter)
    h, b = extraction_corpus(); hf, hh = split_extract_select(h); bf, _ = split_extract_select(b)
    H = [p for p in hh.prompts if not any(x in p.lower() for x in EXCLUDE)][: a.n]

    def collect(ps, key="context_last"):
        return torch.stack([residuals_at_named(model, tok, chat_prompt(tok, p), p)[key] for p in ps])
    r = collect(hf.prompts[: a.n_fit]).mean(0) - collect(bf.prompts[: a.n_fit]).mean(0)

    def rate(vec=None):
        sub, jud = [], []
        for p in H:
            if vec is None: t = generate(model, tok, chat_prompt(tok, p), 60)
            else:
                with ablate(model, vec): t = generate(model, tok, chat_prompt(tok, p), 60)
            sub.append(substring_complied(t)); jud.append(judge_complied(model, tok, p, t))
        return float(np.mean(sub)), float(np.mean(jud))

    base = rate()
    print(f"baseline compliance: substring={base[0]:.2f} judge={base[1]:.2f}\n", flush=True)
    print(f"{'layer':>5} {'substring':>10} {'judge':>7}", flush=True)
    rows = []
    for L in range(r.shape[0]):
        s, j = rate(r[L])
        rows.append({"layer": L, "substring": s, "judge": j})
        flag = "  <<<" if j >= 0.5 else ("  <<" if j >= 0.3 else "")
        print(f"{L:>5} {s:>10.2f} {j:>7.2f}{flag}", flush=True)
    best = max(rows, key=lambda x: x["judge"])
    print(f"\nbest by CAUSAL effect: layer {best['layer']}  judge-compliance {best['judge']:.2f}")
    print(f"AUROC-selected layer was 22, judge-compliance "
          f"{[x for x in rows if x['layer']==22][0]['judge']:.2f}")
    (CFG.results_dir/"arditi_sweep.json").write_text(json.dumps(
        {"baseline": {"substring": base[0], "judge": base[1]}, "rows": rows,
         "best_causal_layer": best["layer"], "n_prompts": len(H)}, indent=2))

if __name__ == "__main__":
    main()
