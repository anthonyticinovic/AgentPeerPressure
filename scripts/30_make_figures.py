"""Static figures for the write-up (docs/writeup.md), sized for pasting into a
Google Doc - PNG, not interactive HTML.

Third pass. First pass produced four figures; second pass cut/merged three of
them; this pass scrapped two more of the survivors on the user's own review,
keeping only the conditions dot plot. The bar: a figure earns its place only
if it shows shape, rank, or trend a table genuinely hides - not a prettier
restatement of numbers already in one. See docs/STATE.md for the full history
(including the steering-sweep and power-curve figures cut here, if either is
ever worth resurrecting - both are in git history, not rebuilt from scratch).

Reads only already-computed results/*.json; no model calls. Every number
plotted here is cross-checked against the write-up's own text before being
drawn - this script must not become a second, silently-divergent source of
truth for the same figures. If a headline number in docs/writeup.md changes,
re-run this and diff the numbers printed to stdout against the prose before
re-pasting.

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
# One palette across every figure: blue = intact refusal / pushed toward
# benign, red = ablated refusal / pushed toward harmful - the same semantic
# axis (danger increases) everywhere, so a reader who learns it once does
# not have to re-learn it in the next figure.

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
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")


def _save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _load(name: str):
    return json.loads((RESULTS / name).read_text())


def _condition_rates() -> tuple[list[str], list[float], list[float]]:
    d = _load("gate_a_full_base_judged.json")["rows"]
    ab = _load("gate_a_full_abl_judged.json")["rows"]
    conds = ["C0", "C1", "C1b", "C2", "C3", "C4", "C5", "C6", "C7"]

    def rate(rows, cond):
        vals = [bool(r["full_compliance_judged"]) for r in rows if r["condition"] == cond]
        return 100 * sum(vals) / len(vals)

    intact = [rate(d, c) for c in conds]
    ablated = [rate(ab, c) for c in conds]
    return conds, intact, ablated


# --- conditions + primary contrast, one chart --------------------------------
# A sorted horizontal dot plot: rank across all nine conditions is visible
# from the sort, and C2/C3 are pulled out by weight and colour so the primary
# contrast reads directly off the same chart. The one figure that survived
# two rounds of review.


def fig_conditions() -> None:
    conds, intact, ablated = _condition_rates()
    print("conditions  intact ", {c: round(v, 1) for c, v in zip(conds, intact)})
    print("conditions  ablated", {c: round(v, 1) for c, v in zip(conds, ablated)})

    order = sorted(range(len(conds)), key=lambda i: ablated[i])
    conds_s = [conds[i] for i in order]
    intact_s = [intact[i] for i in order]
    ablated_s = [ablated[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ys = range(len(conds_s))
    for y, a, b in zip(ys, intact_s, ablated_s):
        ax.plot([a, b], [y, y], "-", color=GRID, lw=6, solid_capstyle="round", zorder=1)
    ax.scatter(intact_s, ys, s=90, color=INTACT, zorder=3, label="refusal intact")
    ax.scatter(ablated_s, ys, s=90, color=ABLATED, zorder=3, label="refusal ablated")

    labels = []
    for c in conds_s:
        if c in ("C2", "C3"):
            labels.append(f"$\\bf{{{c}}}$")
        else:
            labels.append(c)
    ax.set_yticks(list(ys), labels)
    for y, c in zip(ys, conds_s):
        if c in ("C2", "C3"):
            ax.get_yticklabels()[y].set_color(ACCENT)

    for y, c, a, b in zip(ys, conds_s, intact_s, ablated_s):
        if c in ("C2", "C3"):
            ax.annotate(f"{a:.1f}%", (a, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, color=INTACT, fontweight="bold")
            ax.annotate(f"{b:.1f}%", (b, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, color=ABLATED, fontweight="bold")

    ax.set_xlim(0, 55)
    ax.set_ylim(-0.8, len(conds_s) - 0.2)
    ax.set_xlabel("Full compliance rate (%)", labelpad=10)
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Where the primary contrast sits among all nine conditions",
                  fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, -0.19,
            "sorted by ablated-arm rate  ·  C2/C3 (amber) = pre-registered primary contrast, interaction +5.77pp, p = 0.0408\n"
            "AgentHarm, 208 items, Qwen3.5-9B",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "conditions")


def main() -> None:
    fig_conditions()


if __name__ == "__main__":
    main()
