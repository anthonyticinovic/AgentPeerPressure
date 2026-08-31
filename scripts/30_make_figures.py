"""Static figures for the write-up (docs/writeup.md), sized for pasting into a
Google Doc - PNG, not interactive HTML.

Reads only already-computed results/*.json; no model calls, no new compute.
Every number plotted here is cross-checked against the write-up's own text
before being drawn (see docs/STATE.md for the verification) - this script
must not become a second, silently-divergent source of truth for the same
figures. If a headline number in docs/writeup.md changes, re-run this and
diff the numbers printed to stdout against the prose before re-pasting.

Usage (from repo root, project venv):
    .venv/bin/python3 scripts/30_make_figures.py

Writes PNGs to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = Path("figures")

# --- shared style ------------------------------------------------------------
# One palette across all four figures: blue = intact refusal / pushed toward
# benign, red = ablated refusal / pushed toward harmful. The mapping is the
# same semantic axis (danger increases) in every figure, so a reader who
# learns it once in Figure 1 does not have to re-learn it in Figure 3.

INTACT = "#2E5C8A"   # steel blue
ABLATED = "#B9483D"  # brick red
INK = "#262322"      # near-black, not pure black
GRID = "#E4DFD3"     # warm light grey
ACCENT = "#C98A2C"   # warm amber, used only for annotation

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")


def _save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _load(name: str):
    return json.loads((RESULTS / name).read_text())


# --- Figure 1: primary interaction ------------------------------------------


def fig1_interaction() -> None:
    d = _load("gate_a_full_base_judged.json")["rows"]
    ab = _load("gate_a_full_abl_judged.json")["rows"]

    def rate(rows, cond):
        vals = [bool(r["full_compliance_judged"]) for r in rows if r["condition"] == cond]
        return 100 * sum(vals) / len(vals)

    intact = [rate(d, "C2"), rate(d, "C3")]
    ablated = [rate(ab, "C2"), rate(ab, "C3")]
    print(f"fig1  intact C2={intact[0]:.1f} C3={intact[1]:.1f}  "
          f"ablated C2={ablated[0]:.1f} C3={ablated[1]:.1f}")

    fig, ax = plt.subplots(figsize=(6.2, 5))
    x = [0, 1]
    ax.plot(x, intact, "o-", color=INTACT, lw=2.5, ms=9, label="refusal intact")
    ax.plot(x, ablated, "o-", color=ABLATED, lw=2.5, ms=9, label="refusal ablated")

    for xi, y in zip(x, intact):
        ax.annotate(f"{y:.1f}%", (xi, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", color=INTACT, fontsize=10, fontweight="bold")
    for xi, y in zip(x, ablated):
        ax.annotate(f"{y:.1f}%", (xi, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=ABLATED, fontsize=10, fontweight="bold")

    ax.set_xticks(x, ["C2\n(peers endorsed)", "C3\n(peers refused)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Full compliance rate (%)")
    ax.set_ylim(0, 52)
    _clean_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("The peer-pressure gap only opens once refusal is ablated",
                  fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, -0.19,
            "interaction (primary, pre-registered): +5.77pp, p = 0.0408  ·  n = 208 items × 9 conditions, Qwen3.5-9B",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "01_interaction")


# --- Figure 2: full condition panel -----------------------------------------


def fig2_conditions() -> None:
    d = _load("gate_a_full_base_judged.json")["rows"]
    ab = _load("gate_a_full_abl_judged.json")["rows"]
    conds = ["C0", "C1", "C1b", "C2", "C3", "C4", "C5", "C6", "C7"]

    def rate(rows, cond):
        vals = [bool(r["full_compliance_judged"]) for r in rows if r["condition"] == cond]
        return 100 * sum(vals) / len(vals)

    intact = [rate(d, c) for c in conds]
    ablated = [rate(ab, c) for c in conds]
    print("fig2  intact ", {c: round(v, 1) for c, v in zip(conds, intact)})
    print("fig2  ablated", {c: round(v, 1) for c, v in zip(conds, ablated)})

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(conds))
    w = 0.36
    ax.bar([i - w / 2 for i in x], intact, width=w, color=INTACT, label="refusal intact")
    ax.bar([i + w / 2 for i in x], ablated, width=w, color=ABLATED, label="refusal ablated")

    for i, c in enumerate(conds):
        if c in ("C2", "C3"):
            ax.axvspan(i - 0.45, i + 0.45, color=ACCENT, alpha=0.12, zorder=0)

    ax.set_xticks(list(x), conds)
    ax.set_ylabel("Full compliance rate (%)")
    ax.set_ylim(0, 55)
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Compliance across all nine conditions", fontsize=13, fontweight="bold", pad=26)
    ax.text(0.5, 1.04,
            "shaded = C2/C3, the pre-registered peer-pressure contrast  ·  AgentHarm, 208 items, Qwen3.5-9B",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "02_conditions")


# --- Figure 3: causal-steering layer sweep ----------------------------------


def fig3_steering() -> None:
    d = _load("inversion_analysis.json")
    arms = d["arms"]
    baseline = d["baseline_p_harmful"]

    panels = [
        ("r_harm", "harmless/r_harm+", "harmless/r_harm-"),
        ("r_ref", "harmless/r_ref+", "harmless/r_ref-"),
        ("r_arditi", "harmless/r_arditi+", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    for ax, (title, plus_key, minus_key) in zip(axes, panels):
        plus = arms[plus_key]["series"]
        layers = [r["layer"] for r in plus]
        p_plus = [r["p_harmful"] for r in plus]
        ax.plot(layers, p_plus, "-", color=ABLATED, lw=2.2, label="pushed toward harmful")
        peak = max(plus, key=lambda r: r["p_harmful"])
        ax.plot(peak["layer"], peak["p_harmful"], "o", color=ABLATED, ms=7, zorder=5)
        ax.annotate(f"{peak['p_harmful']:.2f} @ L{peak['layer']}",
                    (peak["layer"], peak["p_harmful"]), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color=ABLATED, fontweight="bold")

        if minus_key:
            minus = arms[minus_key]["series"]
            p_minus = [r["p_harmful"] for r in minus]
            ax.plot(layers, p_minus, "-", color=INTACT, lw=2.2, label="pushed toward benign")
            mpeak = max(minus, key=lambda r: r["p_harmful"])
            if mpeak["p_harmful"] > 0.05:
                ax.plot(mpeak["layer"], mpeak["p_harmful"], "o", color=INTACT, ms=7, zorder=5)
                ax.annotate(f"{mpeak['p_harmful']:.2f} @ L{mpeak['layer']}",
                            (mpeak["layer"], mpeak["p_harmful"]), textcoords="offset points",
                            xytext=(6, -14), fontsize=9, color=INTACT, fontweight="bold")

        ax.axhline(baseline["harmless"], color=INK, lw=0.8, ls=":", alpha=0.5)
        ax.axhline(baseline["harmful"], color=INK, lw=0.8, ls=":", alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", family="monospace")
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 1.05)
        _clean_axes(ax)

    axes[0].set_ylabel("judged-harmful rate")
    axes[0].text(16, baseline["harmful"] - 0.05, "harmful baseline", fontsize=8,
                 color=INK, alpha=0.55, ha="center", va="top")
    axes[0].text(16, baseline["harmless"] + 0.05, "benign baseline", fontsize=8,
                 color=INK, alpha=0.55, ha="center", va="bottom")
    axes[0].legend(loc="center right", frameon=False, fontsize=8.5)

    fig.suptitle("Only r_harm's steering effect tracks the harmful/benign label",
                  fontsize=14, fontweight="bold", y=1.09)
    fig.text(0.5, 1.015,
              "judged-harmful rate under steering, by layer  ·  Qwen3.5-9B, n=50 held-out/class, single-turn completions",
              ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "03_steering")


# --- Figure 4: power curve ---------------------------------------------------


def fig4_power() -> None:
    d = _load("interaction_power.json")
    power = d["power_by_interaction_pp"]
    pairs = sorted((float(k), v) for k, v in power.items() if v is not None)
    xs, ys = zip(*pairs)
    print("fig4  power:", dict(zip(xs, ys)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(13.5, 32, color=GRID, alpha=0.6, zorder=0)
    ax.text(22.5, 0.5, "not assessable\n(beyond this run's\ndiscordance ceiling)",
            ha="center", va="center", fontsize=9, color=INK, alpha=0.65)

    ax.plot(xs, ys, "o-", color=INK, lw=2.2, ms=7)
    ax.axhline(0.8, color=INTACT, lw=1, ls="--", alpha=0.6)
    ax.text(0.3, 0.815, "conventional \"well-powered\" benchmark (80%)", fontsize=8.5,
            color=INTACT, alpha=0.85)

    ax.axvline(5.77, color=ACCENT, lw=1.6, ls="--")
    ax.annotate("observed effect\n+5.77pp, p = 0.0408\n(26% power)", (5.77, 0.26),
                textcoords="offset points", xytext=(12, 30), fontsize=9.5,
                color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))

    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("true interaction effect (percentage points)")
    ax.set_ylabel("power")
    _clean_axes(ax)
    ax.set_title("Only 26% power to detect the observed effect", fontsize=13,
                  fontweight="bold", pad=14)
    ax.text(0.5, -0.16,
            "resampling-based power curve, scripts/21_interaction_power.py  ·  n = 208 items, C2−C3 interaction",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "04_power")


def main() -> None:
    fig1_interaction()
    fig2_conditions()
    fig3_steering()
    fig4_power()


if __name__ == "__main__":
    main()
