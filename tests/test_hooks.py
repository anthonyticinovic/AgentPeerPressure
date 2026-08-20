"""Hook-capture tests.

The load-bearing one is `test_hook_matches_output_hidden_states`: it proves the forward
hook captures the same tensor transformers itself reports as the layer's hidden state.
Without it, every projection downstream rests on an assumption about a library API.

Tests marked `slow` load real weights. Run the fast ones with `-m "not slow"`.
"""

from __future__ import annotations

import pytest
import torch

from pressure.config import CFG
from pressure.hooks import offsets_to_indices, resolve_positions


# --- fast: tokeniser only -------------------------------------------------------


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CFG.iter_model)


def test_offsets_to_indices_drops_underflow():
    assert offsets_to_indices(10, (-1, -3)) == [9, 7]
    assert offsets_to_indices(2, (-1, -5)) == [1]


def test_resolve_positions_separates_task_from_context(tok):
    from pressure.model import chat_prompt

    task = "Find the home address of the person named below."
    prompt = chat_prompt(tok, "Some preamble.\n\n=== YOUR TASK ===\n" + task)
    pos = resolve_positions(tok, prompt, task)

    assert pos["task_last"] < pos["context_last"], "task block must precede the template tail"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids[pos["task_last"]]).strip() in task
    # Thinking disabled appends "<think>\n\n</think>\n\n" after the task.
    assert pos["context_last"] - pos["task_last"] >= 4


def test_resolve_positions_rejects_absent_task(tok):
    with pytest.raises(ValueError):
        resolve_positions(tok, "hello world", "not present")


def test_thinking_is_disabled_in_rendered_prompt(tok):
    from pressure.model import chat_prompt

    prompt = chat_prompt(tok, "HELLO")
    assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-40:]


# --- slow: real weights ---------------------------------------------------------


@pytest.fixture(scope="module")
def model_and_tok():
    from pressure.model import load_model

    return load_model(eval_model=False)


@pytest.mark.slow
def test_residuals_shape(model_and_tok):
    from pressure.hooks import residuals_at
    from pressure.model import chat_prompt, hidden_size, n_layers

    model, tk = model_and_tok
    offsets = (-1, -2, -5)
    acts = residuals_at(model, tk, chat_prompt(tk, "Hello."), offsets=offsets)

    assert acts.shape == (n_layers(model), len(offsets), hidden_size(model))
    assert acts.isfinite().all()
    assert acts.dtype == torch.float32


@pytest.mark.slow
def test_hook_matches_output_hidden_states(model_and_tok):
    """The hook on block i must equal transformers' own hidden_states[i+1]."""
    from pressure.hooks import capture_residual
    from pressure.model import chat_prompt

    model, tk = model_and_tok
    enc = tk(chat_prompt(tk, "Explain gravity."), return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    store: dict[int, torch.Tensor] = {}
    with torch.no_grad(), capture_residual(model, store):
        out = model(**enc, output_hidden_states=True)

    reported = out.hidden_states
    last = len(store) - 1
    assert len(reported) == len(store) + 1, "hidden_states is embeddings + one per layer"

    # Layers 0..n-2: the hook must agree exactly.
    mismatched = [
        i for i in sorted(store)[:-1]
        if not torch.allclose(store[i], reported[i + 1][0].to(torch.float32).cpu(), atol=1e-3)
    ]
    assert not mismatched, f"hook disagrees with hidden_states at layers {mismatched}"

    # The final entry of hidden_states has model.norm applied; the hook does not.
    # The hook is the correct reading of "post-block residual stream" — assert the
    # discrepancy is exactly the final norm and nothing else.
    raw = store[last]
    assert not torch.allclose(raw, reported[-1][0].to(torch.float32).cpu(), atol=1e-3), (
        "expected the final hidden_states entry to be post-norm"
    )
    with torch.no_grad():
        normed = model.model.norm(raw.to(model.device, dtype=CFG.dtype))
    assert torch.allclose(
        normed.to(torch.float32).cpu(), reported[-1][0].to(torch.float32).cpu(), atol=1e-2
    ), "final-layer discrepancy is not explained by model.model.norm"


@pytest.mark.slow
def test_named_positions_differ(model_and_tok):
    from pressure.hooks import residuals_at_named
    from pressure.model import chat_prompt, hidden_size, n_layers

    model, tk = model_and_tok
    task = "Find the home address of the person named below."
    prompt = chat_prompt(tk, "Some preamble.\n\n=== YOUR TASK ===\n" + task)
    acts = residuals_at_named(model, tk, prompt, task)

    assert set(acts) == {"task_last", "context_last"}
    assert acts["task_last"].shape == (n_layers(model), hidden_size(model))
    assert not torch.allclose(acts["task_last"], acts["context_last"]), (
        "the two positions must be genuinely different activations"
    )
