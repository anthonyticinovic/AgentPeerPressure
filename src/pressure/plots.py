"""Figures. Always written to PNG as well as returned — see CLAUDE.md."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CFG  # noqa: E402

FULL = "full_attention"


def save(fig, name: str):
    out = CFG.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def sweep_figures(auroc, offsets, types, l_star, o_star, prefix: str = "r_ref"):
    """Heatmap over (layer, offset) plus a per-layer profile at the selected offset.

    Full-attention layers are marked. Qwen3.5 interleaves lllF, so a sawtooth in the
    profile is the architecture rather than noise, and the reader must be able to see
    which layers are which.
    """
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    im = ax0.imshow(auroc, aspect="auto", origin="lower", vmin=0.5, vmax=1.0, cmap="viridis")
    ax0.set_xticks(range(len(offsets)), [str(o) for o in offsets])
    ax0.set_xlabel("token offset from end of prompt")
    ax0.set_ylabel("layer")
    ax0.set_title("Held-out AUROC: harmful vs harmless")
    ax0.scatter([o_star], [l_star], marker="o", s=140, facecolors="none", edgecolors="white", lw=2)
    ax0.annotate(
        f"l*={l_star}, offset={offsets[o_star]}\nAUROC={auroc[l_star, o_star]:.3f}",
        (o_star, l_star), textcoords="offset points", xytext=(8, 8), color="white", fontsize=9,
    )
    for i, t in enumerate(types):
        if t == FULL:
            ax0.axhline(i, color="white", lw=0.4, alpha=0.35)
    fig.colorbar(im, ax=ax0, label="AUROC")

    profile = auroc[:, o_star]
    layers = range(len(profile))
    ax1.plot(profile, layers, color="#444", lw=1.2, zorder=1)
    full_idx = [i for i, t in enumerate(types) if t == FULL]
    lin_idx = [i for i, t in enumerate(types) if t != FULL]
    ax1.scatter(profile[lin_idx], lin_idx, s=26, color="#7aa6c2", label="linear attention", zorder=2)
    ax1.scatter(profile[full_idx], full_idx, s=44, color="#c2452d", label="full attention", zorder=3)
    ax1.axvline(0.5, color="grey", ls=":", lw=1)
    ax1.axhline(l_star, color="#c2452d", ls="--", lw=1, alpha=0.5)
    ax1.set_xlabel(f"AUROC at offset {offsets[o_star]}")
    ax1.set_ylabel("layer")
    ax1.set_xlim(0.4, 1.02)
    ax1.set_title("Per-layer profile by attention type")
    ax1.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    return save(fig, f"{prefix}_sweep")


