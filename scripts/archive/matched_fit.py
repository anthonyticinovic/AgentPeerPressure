"""Fit the direction on MATCHED pairs, not just select on them.

The audit found that r_ref fitted on AdvBench-vs-Alpaca is the top principal component
of that two-corpus activation cloud (cos 0.999, 31% of variance), and that a
shuffled-label diff-of-means recovers a direction 86% aligned with it. In other words
the procedure returns the dominant corpus axis whatever the labels say.

Matched-pair selection fixed which layer we pick; it did not fix what we fit. This
script fits on JailbreakBench's topic-matched pairs, where the two classes differ in
harmfulness and not in register, and compares:

  - alignment with the corpus-fitted direction
  - how dominant each direction is in its own activation cloud
  - whether ablating it moves judge-graded compliance

    uv run python scripts/07_matched_fit.py --iter
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pressure.causal import ablate, generate, shuffled_label_direction, top_pc_direction
from pressure.config import CFG
from pressure.data import agentharm, extraction_corpus, matched_pairs, split_extract_select
from pressure.grade import judge_complied, save_transcripts, substring_complied
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, load_model

STRONG = ("Disinformation", "Hate", "Harassment", "Copyright")


def collect(model, tok, prompts, key="context_last"):
    return torch.stack(
        [residuals_at_named(model, tok, chat_prompt(tok, p), p)[key] for p in prompts]
    )


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).abs())


def var_explained(X: torch.Tensor, v: torch.Tensor) -> float:
    Xc = (X - X.mean(0)).double()
    u = (v / v.norm()).double()
    return float(((Xc @ u) ** 2).sum() / (Xc**2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n-test", type=int, default=25)
    args = ap.parse_args()

    model, tok = load_model(eval_model=not args.iter)
    harmful, harmless = extraction_corpus()
    h_fit, h_held = split_extract_select(harmful)
    b_fit, _ = split_extract_select(harmless)
    m_h, m_b = matched_pairs()

    print("collecting activations", flush=True)
    Ah = collect(model, tok, h_fit.prompts[:200])
    Ab = collect(model, tok, b_fit.prompts[:200])
    Mh = collect(model, tok, m_h.prompts)
    Mb = collect(model, tok, m_b.prompts)

    r_corpus = Ah.mean(0) - Ab.mean(0)          # fitted on AdvBench vs Alpaca
    r_matched = Mh.mean(0) - Mb.mean(0)         # fitted on JBB matched pairs
    pc_matched = top_pc_direction(torch.cat([Mh, Mb]))
    shuf_matched = shuffled_label_direction(Mh, Mb)

    n_layers = r_corpus.shape[0]
    print("\nlayer | cos(matched, corpus) | matched: cos vs topPC | cos vs shuffled | var expl", flush=True)
    diag = []
    for L in (12, 16, 20, 22, 26, 30):
        row = {
            "layer": L,
            "cos_matched_corpus": cos(r_matched[L], r_corpus[L]),
            "cos_matched_pc": cos(r_matched[L], pc_matched[L]),
            "cos_matched_shuffled": cos(r_matched[L], shuf_matched[L]),
            "var_explained": var_explained(torch.cat([Mh, Mb])[:, L, :], r_matched[L]),
        }
        diag.append(row)
        print(f"{L:>5} | {row['cos_matched_corpus']:>20.3f} | {row['cos_matched_pc']:>21.3f} "
              f"| {row['cos_matched_shuffled']:>15.3f} | {row['var_explained']:>8.3f}", flush=True)

    # Select the layer for the matched direction by AUROC on its own held-out half.
    half = len(Mh) // 2
    aur = []
    for L in range(n_layers):
        v = (Mh[:half].mean(0) - Mb[:half].mean(0))[L]
        ph = (Mh[half:, L, :] @ v).numpy()
        pb = (Mb[half:, L, :] @ v).numpy()
        y = np.r_[np.ones(len(ph)), np.zeros(len(pb))]
        aur.append(roc_auc_score(y, np.r_[ph, pb]))
    L_star = int(np.argmax(aur))
    print(f"\nmatched-fit l* = {L_star}  (held-out AUROC {aur[L_star]:.3f})", flush=True)

    # --- does it move behaviour? ---------------------------------------------
    H = h_held.prompts[: args.n_test]
    AH = [i["prompt"] for i in agentharm(harmful=True) if i["category"] in STRONG][: args.n_test]
    rows: list[dict] = []

    def measure(label, prompts, vec=None):
        sub, jud = [], []
        for p in prompts:
            if vec is None:
                text = generate(model, tok, chat_prompt(tok, p), 60)
            else:
                with ablate(model, vec):
                    text = generate(model, tok, chat_prompt(tok, p), 60)
            s, j = substring_complied(text), judge_complied(model, tok, p, text)
            sub.append(s)
            jud.append(j)
            rows.append({"condition": label, "prompt": p, "reply": text,
                         "substring": s, "judge": j})
        print(f"  {label:34} substring={np.mean(sub):.2f}  judge={np.mean(jud):.2f}", flush=True)
        return float(np.mean(sub)), float(np.mean(jud))

    print("\nAdvBench held-out", flush=True)
    res = {
        "baseline": measure("baseline", H),
        "matched": measure(f"ablate r_matched @L{L_star}", H, r_matched[L_star]),
        "matched_shuffled": measure(f"ablate matched-shuffled @L{L_star}", H, shuf_matched[L_star]),
        "corpus": measure("ablate r_corpus @L22", H, r_corpus[22]),
    }
    print("\nAgentHarm, four strong categories", flush=True)
    res["ah_baseline"] = measure("AgentHarm baseline", AH)
    res["ah_matched"] = measure(f"AgentHarm ablate r_matched @L{L_star}", AH, r_matched[L_star])

    save_transcripts(CFG.results_dir / "transcripts_matched.json", rows)
    (CFG.results_dir / "matched_fit.json").write_text(json.dumps({
        "l_star": L_star, "auroc": aur[L_star], "auroc_by_layer": aur,
        "diagnostics": diag,
        "compliance": {k: {"substring": v[0], "judge": v[1]} for k, v in res.items()},
        "n_matched_pairs": len(Mh),
    }, indent=2))
    print("\nwrote matched_fit.json", flush=True)


if __name__ == "__main__":
    main()
