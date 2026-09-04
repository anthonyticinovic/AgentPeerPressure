"""Diagnostic for a failed inversion pre-flight: is the judgement absent, or just unread?

The pre-flight found the 4B answers 'No' to every prompt, harmful or harmless. Greedy
decoding only reports the argmax, so it cannot distinguish two very different situations:

  (a) the model has no harmfulness judgement here    -> logit gap flat across classes
  (b) it has one, but a strong 'No' prior dominates  -> gap separates, argmax does not

(b) is recoverable and (a) is not, so this must be settled before reporting a stop.
Also varies the two things most likely to be responsible: the template, and thinking mode
(disabled project-wide, but this is a classification task and the template pre-closes the
reasoning block).

    uv run python scripts/diagnostics/diag_inversion.py --iter
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from pressure.config import CFG
from pressure.data import extraction_corpus, split_extract_select
from pressure.inversion import build_prompt, load_inversion_prompts, save_json
from pressure.model import load_model

EXCLUDE = ("cyberbullying",)


def render(tok, text: str, thinking: bool) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


@torch.no_grad()
def first_token_gap(model, tok, prompts: list[str], yes_id: int, no_id: int) -> list[float]:
    """log P('Certainly') - log P('No') at the first generated position."""
    out = []
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), 8):
        enc = tok(prompts[i : i + 8], return_tensors="pt", add_special_tokens=False,
                  padding=True, padding_side="left")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        out.extend((lp[:, yes_id] - lp[:, no_id]).cpu().tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--templates", type=int, default=6)
    ap.add_argument("--thinking", default="both", choices=("both", "off", "on"))
    args = ap.parse_args()
    args.thinking_modes = {"both": (False, True), "off": (False,), "on": (True,)}[args.thinking]

    model, tok = load_model(eval_model=not args.iter)
    templates = load_inversion_prompts()

    harmful, harmless = extraction_corpus()
    _, h_sel = split_extract_select(harmful)
    _, b_sel = split_extract_select(harmless)
    H = [p for p in h_sel.prompts if not any(x in p.lower() for x in EXCLUDE)][: args.n]
    B = list(b_sel.prompts)[: args.n]

    yes_id = tok("Certainly", add_special_tokens=False)["input_ids"][0]
    no_id = tok("No", add_special_tokens=False)["input_ids"][0]
    print(f"'Certainly' id={yes_id}  'No' id={no_id}", flush=True)

    rows = []
    print(f"\n{'think':>6} {'tmpl':>5} {'gap harmful':>12} {'gap harmless':>13} "
          f"{'separation':>11} {'AUROC':>7} {'ok':>5}", flush=True)
    for thinking in args.thinking_modes:
        for idx in list(range(args.templates)):
            q = templates[idx]
            ph = [render(tok, build_prompt(p, q), thinking) for p in H]
            pb = [render(tok, build_prompt(p, q), thinking) for p in B]
            gh = first_token_gap(model, tok, ph, yes_id, no_id)
            gb = first_token_gap(model, tok, pb, yes_id, no_id)
            sep = float(np.mean(gh) - np.mean(gb))
            # AUROC of the gap as a harmful/harmless classifier.
            from sklearn.metrics import roc_auc_score
            y = np.r_[np.ones(len(gh)), np.zeros(len(gb))]
            auroc = float(roc_auc_score(y, np.r_[gh, gb]))
            # A template is only usable if greedy decoding lands on the right answer for
            # BOTH classes. High AUROC with a uniform 'No' argmax means the judgement is
            # present but unreadable by generation, which is what the pre-flight saw.
            behaves = bool(np.mean(gh) > 0 and np.mean(gb) < 0)
            rows.append({"thinking": thinking, "template_idx": idx, "question": q,
                         "gap_harmful": float(np.mean(gh)),
                         "gap_harmless": float(np.mean(gb)),
                         "separation": sep, "auroc": auroc, "behaves": behaves})
            print(f"{str(thinking):>6} {idx:>5} {np.mean(gh):>12.2f} {np.mean(gb):>13.2f} "
                  f"{sep:>11.2f} {auroc:>7.3f} {'YES' if behaves else '':>5}", flush=True)

    usable = [r for r in rows if r["behaves"]]
    best = max(usable or rows, key=lambda r: (r["behaves"], r["separation"]))
    print(f"\n{len(usable)}/{len(rows)} configurations answer correctly under greedy decoding",
          flush=True)
    print(f"best: thinking={best['thinking']} template {best['template_idx']} "
          f"AUROC={best['auroc']:.3f} separation={best['separation']:.2f} "
          f"behaves={best['behaves']}", flush=True)
    print(f"  {best['question']}", flush=True)

    save_json(CFG.results_dir / "diag_inversion.json",
              {"n_per_class": args.n, "rows": rows, "best": best})
    print("\nwrote diag_inversion.json", flush=True)


if __name__ == "__main__":
    main()
