"""Static figures for the write-up (docs/writeup.md), sized for pasting into a
Google Doc - PNG, not interactive HTML.

Second pass. First pass produced four figures; two were cut and one was
merged into another after review found them "charts for the sake of it" -
prettier restatements of numbers already in a table. The bar going forward:
a figure earns its place only if it shows shape, rank, or trend a table
genuinely hides. See docs/STATE.md for the full review.

Reads only already-computed results/*.json; no model calls. The one
exception is interaction_power_fine.json, a finer-grained resampling grid
than the original results/interaction_power.json - real new analysis (CPU
only, ~8 minutes, scripts/21_interaction_power.py), not a replot of an
existing number. Every number plotted here is cross-checked against the
write-up's own text before being drawn - this script must not become a
second, silently-divergent source of truth for the same figures. If a
headline number in docs/writeup.md changes, re-run this and diff the
numbers printed to stdout against the prose before re-pasting.

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


# --- Figure 1: causal-steering layer sweep -----------------------------------
# Kept from the first pass, trimmed from 3 panels to 2. The write-up's own
# Result 1 text already discloses r_arditi is literally the same vector as
# r_ref at this scale (site collision at i*=-1 == context_last) - a third
# panel showing identical data was padding, not evidence. That fact is now a
# footnote instead of a redundant plot.


def fig1_steering() -> None:
    d = _load("inversion_analysis.json")
    arms = d["arms"]
    baseline = d["baseline_p_harmful"]

    panels = [
        ("r_harm", "harmless/r_harm+", "harmless/r_harm-"),
        ("r_ref", "harmless/r_ref+", "harmless/r_ref-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    for ax, (title, plus_key, minus_key) in zip(axes, panels):
        plus = arms[plus_key]["series"]
        layers = [r["layer"] for r in plus]
        p_plus = [r["p_harmful"] for r in plus]
        ax.plot(layers, p_plus, "-", color=ABLATED, lw=2.2, label="pushed toward harmful")
        peak = max(plus, key=lambda r: r["p_harmful"])
        ax.plot(peak["layer"], peak["p_harmful"], "o", color=ABLATED, ms=7, zorder=5)
        ax.annotate(f"{peak['p_harmful']:.2f} @ L{peak['layer']}",
                    (peak["layer"], peak["p_harmful"]), textcoords="offset points",
                    xytext=(6, 8), fontsize=9.5, color=ABLATED, fontweight="bold")

        minus = arms[minus_key]["series"]
        p_minus = [r["p_harmful"] for r in minus]
        ax.plot(layers, p_minus, "-", color=INTACT, lw=2.2, label="pushed toward benign")
        mpeak = max(minus, key=lambda r: r["p_harmful"])
        if mpeak["p_harmful"] > 0.05:
            ax.plot(mpeak["layer"], mpeak["p_harmful"], "o", color=INTACT, ms=7, zorder=5)
            ax.annotate(f"{mpeak['p_harmful']:.2f} @ L{mpeak['layer']}",
                        (mpeak["layer"], mpeak["p_harmful"]), textcoords="offset points",
                        xytext=(6, -16), fontsize=9.5, color=INTACT, fontweight="bold")

        ax.axhline(baseline["harmless"], color=INK, lw=0.8, ls=":", alpha=0.5)
        ax.axhline(baseline["harmful"], color=INK, lw=0.8, ls=":", alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold", family="monospace")
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 1.05)
        ax.spines["left"].set_visible(True)
        ax.grid(axis="y")

    axes[0].set_ylabel("judged-harmful rate")
    axes[0].text(21, baseline["harmful"] - 0.05, "harmful baseline", fontsize=8,
                 color=INK, alpha=0.55, ha="center", va="top")
    axes[0].text(21, baseline["harmless"] + 0.05, "benign baseline", fontsize=8,
                 color=INK, alpha=0.55, ha="center", va="bottom")
    axes[0].legend(loc="center right", frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.72)
    fig.suptitle("Only r_harm's steering effect tracks the harmful/benign label",
                  fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.855,
              "judged-harmful rate under steering, by layer  ·  Qwen3.5-9B, n=50 held-out/class, single-turn\n"
              "r_arditi omitted: at this scale it is the same vector as r_ref (site collision, i*=-1 == context_last)",
              ha="center", fontsize=9, color=INK, alpha=0.7)
    _save(fig, "01_steering")


# --- Figure 2: conditions + primary contrast, one chart -----------------------
# Replaces the first pass's Figure 1 (interaction line plot: 4 numbers) and
# Figure 2 (paired vertical bars: 18 shapes + legend). A sorted horizontal
# dot plot does both jobs in one read: rank across all nine conditions is
# visible from the sort, and C2/C3 are pulled out by weight and colour so
# the primary contrast reads directly off the same chart.


def fig2_conditions() -> None:
    conds, intact, ablated = _condition_rates()
    print("fig2  intact ", {c: round(v, 1) for c, v in zip(conds, intact)})
    print("fig2  ablated", {c: round(v, 1) for c, v in zip(conds, ablated)})

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
    _save(fig, "02_conditions")


# --- Figure 3: power curve, now a real curve ----------------------------------
# The first pass drew a "curve" through 3 points (0/5/10pp). That is 3 real
# values joined by straight lines pretending to be a smooth function - worse
# than the table it replaced, since the table did not imply false smoothness.
# This re-runs scripts/21_interaction_power.py at 1pp resolution (real
# resampling, ~8 min CPU, see results/interaction_power_fine.json) so the
# curve is actual data, including the cliff where the method runs out of
# discordant pairs to resample, not an assumed threshold.


def fig3_power() -> None:
    d = _load("interaction_power_fine.json")
    power = d["power_by_interaction_pp"]
    points = sorted(((float(k), v) for k, v in power.items()), key=lambda p: p[0])
    all_x = [x for x, _ in points]
    real = [(x, v) for x, v in points if v is not None]
    xs, ys = zip(*real)
    none_x = [x for x, v in points if v is None]
    cliff = min(none_x) if none_x else max(all_x)
    print("fig3  power (fine grid):", dict(zip(xs, ys)), " cliff at", cliff)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if cliff <= max(all_x):
        ax.axvspan(cliff, max(all_x) + 1, color=GRID, alpha=0.6, zorder=0)
        ax.text((cliff + max(all_x) + 1) / 2, 0.5,
                "not assessable\n(out of real\ndiscordant pairs\nto resample)",
                ha="center", va="center", fontsize=9, color=INK, alpha=0.65)

    ax.plot(xs, ys, "-", color=INK, lw=2, zorder=2)
    ax.scatter(xs, ys, s=22, color=INK, zorder=3)
    ax.axhline(0.8, color=INTACT, lw=1, ls="--", alpha=0.6)
    ax.text(0.3, 0.815, "conventional \"well-powered\" benchmark (80%)", fontsize=8.5,
            color=INTACT, alpha=0.85)

    ax.axvline(5.77, color=ACCENT, lw=1.6, ls="--")
    ax.annotate("observed effect\n+5.77pp, p = 0.0408", (5.77, 0.26),
                textcoords="offset points", xytext=(14, 55), fontsize=9.5,
                color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))

    ax.set_xlim(0, max(all_x) + 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("true interaction effect (percentage points)")
    ax.set_ylabel("power")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.set_title("Power rises quickly, but the observed effect sits low on the curve",
                  fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, -0.16,
            "resampling-based power curve at 1pp resolution, scripts/21_interaction_power.py  ·  n = 208 items, C2−C3 interaction",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK, alpha=0.75)
    _save(fig, "03_power")


def main() -> None:
    fig1_steering()
    fig2_conditions()
    fig3_power()


if __name__ == "__main__":
    main()
