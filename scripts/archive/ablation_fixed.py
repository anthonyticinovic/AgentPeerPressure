"""Ablation at the SELECTED layers, with variance-matched controls and judge grading.

Supersedes the ablation rows in 05_audit_rerun.py, which used `lay[-1] // 2` — an
arbitrary middle layer — instead of the selected l*. Controls:

  shuffled-label : diff-of-means with permuted labels. Same construction, same data,
                   meaningless labels. This is the real null for a diff-of-means.
  top PC         : highest-variance direction in the same activations. Bounds how much
                   of any effect is "removed a lot of variance" vs "removed the right thing".
  random         : isotropic. Reported only to show it cannot move anything.

    uv run python scripts/06_ablation_fixed.py --iter
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pressure.causal import (
    ablate,
    generate,
    random_directions,
    shuffled_label_direction,
    top_pc_direction,
)
from pressure.config import CFG
from pressure.data import agentharm, extraction_corpus, split_extract_select
from pressure.grade import judge_complied, save_transcripts, substring_complied
from pressure.hooks import residuals_at_named
from pressure.model import chat_prompt, load_model

L_REF, L_HARM = 22, 15
STRONG_CATEGORIES = ("Disinformation", "Hate", "Harassment", "Copyright")


def collect(model, tok, prompts, key):
    return torch.stack(
        [residuals_at_named(model, tok, chat_prompt(tok, p), p)[key] for p in prompts]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n-fit", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=25)
    args = ap.parse_args()

    harmful, harmless = extraction_corpus()
    h_fit, h_held = split_extract_select(harmful)
    b_fit, _ = split_extract_select(harmless)
    H = h_held.prompts[: args.n_test]
    AH = [
        i["prompt"] for i in agentharm(harmful=True) if i["category"] in STRONG_CATEGORIES
    ][: args.n_test]

    model, tok = load_model(eval_model=not args.iter)
    print(f"model={CFG.iter_model if args.iter else CFG.eval_model}", flush=True)

    Ah = collect(model, tok, h_fit.prompts[: args.n_fit], "context_last")
    Ab = collect(model, tok, b_fit.prompts[: args.n_fit], "context_last")
    Th = collect(model, tok, h_fit.prompts[: args.n_fit], "task_last")
    Tb = collect(model, tok, b_fit.prompts[: args.n_fit], "task_last")

    r_ref = Ah.mean(0) - Ab.mean(0)
    r_harm = Th.mean(0) - Tb.mean(0)
    shuf = shuffled_label_direction(Ah, Ab)
    pc = top_pc_direction(torch.cat([Ah, Ab]))
    rnd = random_directions(r_ref.shape[-1], n=1)[0]

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
            rows.append(
                {"condition": label, "prompt": p, "reply": text, "substring": s, "judge": j}
            )
        print(f"  {label:32} substring={np.mean(sub):.2f}  judge={np.mean(jud):.2f}", flush=True)
        return float(np.mean(sub)), float(np.mean(jud))

    print(f"\nAdvBench held-out, ablation at selected layers (r_ref L{L_REF}, r_harm L{L_HARM})",
          flush=True)
    res = {
        "baseline": measure("baseline", H),
        "r_ref": measure(f"ablate r_ref @L{L_REF}", H, r_ref[L_REF]),
        "shuffled": measure(f"ablate shuffled-label @L{L_REF}", H, shuf[L_REF]),
        "top_pc": measure(f"ablate top PC @L{L_REF}", H, pc[L_REF]),
        "random": measure(f"ablate random @L{L_REF}", H, rnd),
        "r_harm": measure(f"ablate r_harm @L{L_HARM}", H, r_harm[L_HARM]),
    }
    print("\nAgentHarm (four strong categories), chat format", flush=True)
    res["agentharm_baseline"] = measure("AgentHarm baseline", AH)
    res["agentharm_ablate_r_ref"] = measure(f"AgentHarm ablate r_ref @L{L_REF}", AH, r_ref[L_REF])

    save_transcripts(CFG.results_dir / "transcripts_ablation.json", rows)
    (CFG.results_dir / "audit_ablation_fixed.json").write_text(
        json.dumps(
            {
                "layers": {"r_ref": L_REF, "r_harm": L_HARM},
                "n_test": args.n_test,
                "compliance": {k: {"substring": v[0], "judge": v[1]} for k, v in res.items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(rows)} transcripts for hand audit", flush=True)


if __name__ == "__main__":
    main()
