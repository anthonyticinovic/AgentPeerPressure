"""Device portability. One place that knows how MPS and CUDA differ.

The same code runs on the local M4 Pro (MPS) and on A100/H100 (CUDA). Everything that
must branch on backend branches here, so no experiment script ever contains a device
check. Verified on MPS 2026-08-20; the CUDA paths are exercised on first Spartan run.
"""

from __future__ import annotations

import importlib.util

import torch


def resolve_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    """bf16 where it is native, fp32 on CPU.

    A100 (sm80) and H100 (sm90) have native bf16. MPS on M-series supports it at the
    same ~0.2% relative error as CUDA. CPU bf16 is emulated and slow, so fp32 there.
    """
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def attn_implementation(device: str) -> str:
    """FlashAttention-2 only exists on CUDA and only if installed; sdpa elsewhere."""
    if device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def available_memory_gb(device: str) -> float:
    """Usable device memory. MPS reports a working-set recommendation well below the
    machine's unified memory — on a 24 GB M4 Pro it is ~17.8 GB."""
    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        return free / 2**30
    if device == "mps":
        return torch.mps.recommended_max_memory() / 2**30
    import psutil  # noqa: PLC0415 — CPU path only, keeps psutil off the hot path

    return psutil.virtual_memory().available / 2**30


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def synchronize(device: str) -> None:
    """Needed before timing anything — both backends are asynchronous."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe() -> dict[str, object]:
    """Recorded into every results file, so a number can always be traced to the
    machine that produced it."""
    d = resolve_device()
    out: dict[str, object] = {
        "device": d,
        "dtype": str(resolve_dtype(d)).replace("torch.", ""),
        "attn": attn_implementation(d),
        "available_gb": round(available_memory_gb(d), 1),
        "torch": torch.__version__,
    }
    if d == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        out["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    return out
