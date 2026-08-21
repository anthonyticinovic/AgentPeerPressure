"""Arditi et al. (2406.11717) direction selection, implemented from the paper.

Three things this module gets right that earlier scripts in this project did not:

1. **The candidate set is a grid, not a line.** Section 2.3: candidates are
   `r_i^(l) = mu_i^(l) - nu_i^(l)` for every *post-instruction* token position i and
   every layer l. Table 5 shows i* is not always -1 (LLAMA-3 8B and YI 6B both select
   -5), so sweeping layers at a fixed position searches a slice of their space.
   Qwen3.5 with thinking disabled has nine post-instruction tokens
   ("<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"), so the grid is
   9 x 32 = 288 candidates.

2. **Selection uses the refusal metric, not generation.** Appendix B: refusals begin
   with a small set of characteristic tokens R, so `log P(R) - log P(not R)` at the
   first generated position is a per-candidate score costing one forward pass. That is
   what makes 288 candidates affordable. Grading generations with a judge does not.

3. **Three filters, not one.** Appendix C.1: minimise `bypass_score` subject to
   `induce_score > 0`, `kl_score < 0.1`, and `l < 0.8L`. The KL filter rejects
   directions that wreck the model on harmless prompts; the layer filter rejects
   directions near the unembedding, which suppress refusal by blocking the *token*
   "I" rather than by removing the *feature*. Both filters reject directions that
   score well on bypass, which is exactly why they exist.

Magnitude convention follows the paper: ablation uses the unit vector (eq. 4),
activation addition uses the raw difference-in-means vector (eq. 3), because its norm
carries the harmful/harmless distance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .hooks import _hidden, blocks

# Arditi's QWEN CHAT row of Table 4, resolved against this tokeniser rather than
# copied: their token ids come from an older Qwen vocabulary ("As" is 2121 there,
# 2053 here). Used only to cross-check the empirically derived set.
PAPER_QWEN_REFUSAL_PHRASES = ("I", "As")


def _first_token_id(tok, text: str) -> int:
    return tok(text, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def next_token_probs(model, tok, prompts: list[str], batch: int = 16) -> torch.Tensor:
    """Mean-free (n_prompts, vocab) next-token probabilities at the last position.

    Left padding, so index -1 is the true final token of every row regardless of
    length. Called under whatever intervention hooks are active.
    """
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        enc = tok(
            chunk,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            padding_side="left",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.softmax(logits, dim=-1).cpu())
        del logits
    return torch.cat(out)


def refusal_metric(probs: torch.Tensor, refusal_ids: list[int]) -> torch.Tensor:
    """log P(first token in R) - log P(first token not in R), per row (eq. 9).

    Log-odds rather than probability: refusal probabilities saturate near 0 and 1, and
    a mean over saturated probabilities cannot distinguish "just refused" from
    "refused overwhelmingly", which is the distinction the selection depends on.
    """
    p_ref = probs[:, refusal_ids].sum(-1).clamp(1e-9, 1 - 1e-9)
    return torch.log(p_ref) - torch.log1p(-p_ref)


def derive_refusal_tokens(
    model,
    tok,
    harmful: list[str],
    harmless: list[str],
    n: int = 2,
    min_prob: float = 0.02,
) -> tuple[list[int], list[dict]]:
    """Find R empirically: tokens that dominate the model's own refusal openings.

    Appendix B defines R as "the tokens most likely to begin the model's refusals",
    and gives a different set per model family. Hard-coding their Qwen ids would
    import an assumption about a two-generation-old tokeniser and a non-thinking
    chat template; deriving it takes two forward passes over the training sets.

    Ranks by harmful mean probability minus harmless mean probability, so a token
    like "**" that opens both refusals and compliant answers cannot enter R.
    """
    ph = next_token_probs(model, tok, harmful).mean(0)
    pb = next_token_probs(model, tok, harmless).mean(0)
    gap = ph - pb
    order = torch.argsort(gap, descending=True)

    table = [
        {
            "token": tok.decode([int(t)]),
            "id": int(t),
            "p_harmful": float(ph[t]),
            "p_harmless": float(pb[t]),
            "gap": float(gap[t]),
        }
        for t in order[:10]
    ]
    chosen = [row["id"] for row in table if row["p_harmful"] >= min_prob][:n]
    if not chosen:
        chosen = [table[0]["id"]]
    return chosen, table


@contextmanager
def add_activation(model, r: torch.Tensor, layer: int):
    """Eq. 3: x^(l) <- x^(l) + r^(l), at layer l only, across all token positions.

    `r` is the raw difference-in-means vector, not normalised. Its norm is the
    harmful/harmless mean distance at that layer and position, which is what sets
    the intervention strength; unit-normalising would apply a constant strength to
    layers whose natural scale differs by an order of magnitude.
    """
    v = r.to(model.device, dtype=next(model.parameters()).dtype)

    def hook(_m, _i, output):
        h = _hidden(output) + v
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = blocks(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, over the vocabulary."""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(-1)


@dataclass(frozen=True)
class Candidate:
    position: int  # end-relative offset, -1 is the last prompt token
    layer: int
    bypass: float
    induce: float
    kl: float

    def admissible(self, n_layers: int) -> bool:
        """Appendix C.1's three filters."""
        return self.induce > 0 and self.kl < 0.1 and self.layer < 0.8 * n_layers


def select(candidates: list[Candidate], n_layers: int) -> Candidate | None:
    """Minimum bypass_score among admissible candidates (Appendix C.1)."""
    ok = [c for c in candidates if c.admissible(n_layers)]
    return min(ok, key=lambda c: c.bypass) if ok else None
