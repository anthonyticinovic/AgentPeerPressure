"""Model handle. Loaded once per process — see CLAUDE.md.

transformers 5.x: `AutoConfig` for Qwen3.5 reports the multimodal wrapper
`Qwen3_5ForConditionalGeneration`, but `AutoModelForCausalLM` yields the text-only
`Qwen3_5ForCausalLM` whose decoder blocks live at `model.model.layers`.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CFG

# Approximate bf16 weight footprint, for the local-memory guard.
_APPROX_GB = {"Qwen/Qwen3.5-9B": 18.0, "Qwen/Qwen3.5-4B": 8.0, "Qwen/Qwen3.5-2B": 4.0}
_MPS_BUDGET_GB = 14.0


def repo_for(eval_model: bool) -> str:
    return CFG.eval_model if eval_model else CFG.iter_model


def load_model(eval_model: bool = True, device: str | None = None):
    """Return (model, tokenizer). Refuses loads that cannot fit the local device."""
    repo = repo_for(eval_model)
    device = device or CFG.device

    if device == "mps" and _APPROX_GB.get(repo, 0.0) > _MPS_BUDGET_GB:
        raise RuntimeError(
            f"{repo} needs ~{_APPROX_GB[repo]}GB bf16, over the ~{_MPS_BUDGET_GB}GB MPS "
            "budget on a 24GB machine. Run it on CUDA. Do not quantise to fit — "
            "quantisation perturbs the residual stream and invalidates the frozen artefact."
        )

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=CFG.dtype)
    model.to(device)
    model.eval()
    return model, tok


def chat_prompt(tok, user_text: str, system: str | None = None) -> str:
    """Render a single-turn chat prompt with the generation prefix attached.

    Thinking is disabled per CFG.enable_thinking, which renders a pre-closed
    "<think>\\n\\n</think>\\n\\n" block rather than leaving "<think>" open.
    """
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=CFG.enable_thinking,
    )


def n_layers(model) -> int:
    return len(model.model.layers)


def hidden_size(model) -> int:
    """Read from the embedding, not the config. Qwen3.5 is a heterogeneous model and
    several config attributes raise AmbiguousGlobalPerLayerAttributeError."""
    return model.model.embed_tokens.embedding_dim


def layer_types(model) -> list[str]:
    """Per-layer attention type. Qwen3.5 interleaves 'lllF' — every fourth layer is
    full attention, the rest linear. The sweep reports l*'s type alongside its index,
    because a direction living on a full-attention layer is a different claim from one
    living on a linear-attention layer."""
    return list(model.config.get_text_config().layer_types)
