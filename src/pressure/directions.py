"""Direction extraction, the layer/position sweep, and the frozen artefact.

Memory note: a naive implementation caches every prompt's activations, which is
n_prompts x n_layers x n_offsets x hidden floats — ~3 GB for 1000 prompts at 4B, and
2.4x that at 9B. Diff-of-means needs only running sums, and the sweep needs only
scalar projections, so nothing here ever holds more than one prompt's activations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import CFG
from .hooks import residuals_at
from .model import chat_prompt


def _usable(tok, prompt: str, offsets: tuple[int, ...]) -> bool:
    """A prompt must be long enough for the deepest offset, else its activation
    tensor has a different shape and would silently misalign the stack."""
    return len(tok(prompt, add_special_tokens=False)["input_ids"]) >= -min(offsets)


class MeanAccumulator:
    """Running mean over (n_layers, n_offsets, hidden) without retaining inputs."""

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.count = 0

    def add(self, acts: torch.Tensor) -> None:
        self.total = acts.clone() if self.total is None else self.total + acts
        self.count += 1

    @property
    def mean(self) -> torch.Tensor:
        if self.total is None:
            raise ValueError("no activations accumulated")
        return self.total / self.count


def accumulate_mean(model, tok, prompts, offsets=None, progress=None) -> torch.Tensor:
    """Mean residual over prompts, shape (n_layers, n_offsets, hidden)."""
    offsets = offsets or CFG.sweep_offsets
    acc = MeanAccumulator()
    for i, p in enumerate(prompts):
        rendered = chat_prompt(tok, p)
        if not _usable(tok, rendered, offsets):
            continue
        acc.add(residuals_at(model, tok, rendered, offsets))
        if progress and i % progress == 0:
            print(f"  {i}/{len(prompts)}", flush=True)
    return acc.mean


def diff_of_means(mean_harmful: torch.Tensor, mean_harmless: torch.Tensor) -> torch.Tensor:
    """Unit-norm direction per (layer, offset)."""
    d = mean_harmful - mean_harmless
    return d / d.norm(dim=-1, keepdim=True)


def project_prompts(model, tok, prompts, direction, offsets=None) -> np.ndarray:
    """Scalar projection per prompt, shape (n_prompts, n_layers, n_offsets)."""
    offsets = offsets or CFG.sweep_offsets
    out = []
    for p in prompts:
        rendered = chat_prompt(tok, p)
        if not _usable(tok, rendered, offsets):
            continue
        acts = residuals_at(model, tok, rendered, offsets)
        out.append((acts * direction).sum(-1).numpy())
    return np.stack(out)


def sweep_auroc(proj_harmful: np.ndarray, proj_harmless: np.ndarray) -> np.ndarray:
    """Held-out AUROC per (layer, offset), shape (n_layers, n_offsets)."""
    n_layers, n_offsets = proj_harmful.shape[1:]
    y = np.r_[np.ones(len(proj_harmful)), np.zeros(len(proj_harmless))]
    out = np.zeros((n_layers, n_offsets))
    for l in range(n_layers):
        for o in range(n_offsets):
            out[l, o] = roc_auc_score(y, np.r_[proj_harmful[:, l, o], proj_harmless[:, l, o]])
    return out


def select_best(auroc: np.ndarray) -> tuple[int, int, float]:
    """Return (layer, offset_index, auroc) at the sweep maximum."""
    l, o = np.unravel_index(auroc.argmax(), auroc.shape)
    return int(l), int(o), float(auroc[l, o])


def calibrate_tau(proj_harmless: np.ndarray, fpr: float | None = None) -> float:
    """Threshold at the given false-positive rate on benign prompts.

    Frozen with the directions and never re-tuned — re-tuning per condition would
    make the 2x2 analysis meaningless.
    """
    fpr = CFG.tau_fpr if fpr is None else fpr
    return float(np.quantile(proj_harmless, 1.0 - fpr))


# --- frozen artefact ------------------------------------------------------------


def freeze(path: Path, **arrays) -> str:
    """Write the artefact plus a manifest, and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (path.parent / "manifest.json").write_text(
        json.dumps({"sha256": digest, "commit": commit, "keys": sorted(arrays)}, indent=2)
    )
    return digest


def load_frozen(path: Path):
    """Load the artefact, asserting its hash still matches the manifest."""
    manifest = json.loads((path.parent / "manifest.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise AssertionError(
            f"frozen artefact mutated: {digest} != {manifest['sha256']}. "
            "Directions must never be recomputed on evaluation data."
        )
    return np.load(path), manifest


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
