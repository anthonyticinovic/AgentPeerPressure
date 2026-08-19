"""Residual-stream capture via raw PyTorch forward hooks on the decoder blocks.

Post-block residual: the hook fires on the output of `model.model.layers[i]`, so
`store[i]` is unambiguously "the residual stream after block i". In transformers 5.x
a Qwen3.5 decoder layer returns a plain tensor; v4 returned a tuple. Both are handled
so the same code runs on either.

Positions are selected *inside* the hook rather than after it. A full capture is
n_layers x seq_len x hidden floats — over 1GB for a long agentic context — and only
a handful of positions are ever used.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .config import CFG


def blocks(model):
    """The decoder block list. Single place that knows the module path."""
    return model.model.layers


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture_residual(model, store: dict[int, torch.Tensor], indices: list[int] | None = None):
    """Populate `store[layer]` with the post-block residual at `indices`.

    `store[layer]` is (n_indices, hidden) when indices are given, else (seq, hidden).
    Batch dimension is assumed to be 1 — every call site runs one prompt at a time.
    """
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            h = _hidden(output)[0]
            sel = h if indices is None else h[indices, :]
            store[idx] = sel.detach().to(torch.float32).cpu()

        return hook

    for i, block in enumerate(blocks(model)):
        handles.append(block.register_forward_hook(make_hook(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def resolve_positions(tok, prompt: str, task_text: str | None = None) -> dict[str, int]:
    """Map the plan's two named token positions to absolute indices.

    `context_last` is the final token of the full rendered context.
    `task_last`    is the final token of the task instruction block — the last token
                   whose characters fall inside `task_text`. With thinking disabled the
                   prompt ends with template tokens, so these two differ by ~6.

    Offset mapping is exact; do not substitute "tokenise the prefix and take its
    length", which is wrong wherever tokenisation is not prefix-stable.
    """
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    out = {"context_last": len(enc["input_ids"]) - 1}

    if task_text is not None:
        start = prompt.rfind(task_text)
        if start < 0:
            raise ValueError("task_text does not occur in prompt")
        end = start + len(task_text)
        inside = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not inside:
            raise ValueError("task_text maps to no tokens")
        out["task_last"] = inside[-1]

    return out


def offsets_to_indices(seq_len: int, offsets: tuple[int, ...]) -> list[int]:
    """Convert negative end-relative offsets to absolute indices, dropping any that
    fall off the front of a short sequence."""
    return [seq_len + o for o in offsets if seq_len + o >= 0]


@torch.no_grad()
def residuals_at(model, tok, prompt: str, offsets: tuple[int, ...] | None = None) -> torch.Tensor:
    """Return (n_layers, n_offsets, hidden) for end-relative token offsets."""
    offsets = offsets if offsets is not None else CFG.sweep_offsets
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    seq_len = enc["input_ids"].shape[1]
    indices = offsets_to_indices(seq_len, offsets)

    store: dict[int, torch.Tensor] = {}
    with capture_residual(model, store, indices):
        model(**enc)
    return torch.stack([store[i] for i in sorted(store)])


@torch.no_grad()
def residuals_at_named(model, tok, prompt: str, task_text: str) -> dict[str, torch.Tensor]:
    """Return {'task_last': (n_layers, hidden), 'context_last': (n_layers, hidden)}.

    This is the evaluation-time capture. Both positions are always computed —
    reporting only the better one is forbidden (plan §5.7).
    """
    pos = resolve_positions(tok, prompt, task_text)
    names = sorted(pos, key=lambda k: pos[k])
    indices = [pos[n] for n in names]

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    store: dict[int, torch.Tensor] = {}
    with capture_residual(model, store, indices):
        model(**enc)
    stacked = torch.stack([store[i] for i in sorted(store)])  # (n_layers, n_named, hidden)
    return {name: stacked[:, j, :] for j, name in enumerate(names)}
