"""Central configuration. Every tunable lives here, nothing is hard-coded downstream.

Model ids were resolved against the live Hub on 2026-08-20 (Task 0.3). The plan named
"Qwen 3.6 9B/4B"; the 3.6 line ships no dense model at those sizes, so assumption A3's
own fallback applies and the 3.5 line is used. Both models have 32 layers, so l* is
directly comparable between them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Config:
    # --- Models -------------------------------------------------------------
    # eval_model is the reported model. iter_model is the local 4B: pipeline checks
    # only, never a reported number, except the single scale row in the control table.
    eval_model: str = os.getenv("PRESSURE_EVAL_MODEL", "Qwen/Qwen3.5-9B")
    iter_model: str = os.getenv("PRESSURE_ITER_MODEL", "Qwen/Qwen3.5-4B")
    dtype: torch.dtype = torch.bfloat16
    device: str = field(default_factory=_device)

    # Qwen3.5 enables thinking by default. Disabled here: Arditi and Zhao both built
    # their constructs on non-thinking models, p* stays well-defined, and rollout cost
    # stays inside the 20h budget. Recorded as a design decision, not an accident.
    enable_thinking: bool = False

    # --- Extraction ---------------------------------------------------------
    n_extract_pairs: int = 400
    n_select_pairs: int = 100
    seed: int = 0

    # Token offsets from the end of the rendered prompt, swept in Task 1.3.
    # With thinking disabled the prompt ends "<think>\n\n</think>\n\n", so offsets
    # -1..-6 are template tokens and the sweep must reach past them to hit content.
    sweep_offsets: tuple[int, ...] = (-1, -2, -3, -4, -5, -6, -8, -10, -12)

    # --- Frozen artefact ----------------------------------------------------
    frozen_dir: Path = ROOT / "artifacts" / "frozen"

    # --- Evaluation ---------------------------------------------------------
    conditions: tuple[str, ...] = ("C0", "C1", "C1b", "C2", "C3")
    positions: tuple[str, ...] = ("task_last", "context_last")
    tau_fpr: float = 0.05
    n_random_controls: int = 5

    # --- Paths --------------------------------------------------------------
    boards_dir: Path = ROOT / "boards"
    results_dir: Path = ROOT / "results"

    def __post_init__(self) -> None:
        if not self.eval_model or not self.iter_model:
            raise ValueError("eval_model and iter_model must be non-empty HF repo ids")


CFG = Config()
