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


# --- content-bearing follow-on: original primary vs. C8/C8b/C9/C8n ----------
# Fixed order, not sorted by rate (unlike fig_conditions above): the point
# here is comparing two specific valence-flip pairs (C2/C3 against C8/C9),
# which sorting-by-rate would scatter. C0/C1b/C2/C3 are the published Result
# 2/3 numbers, reused verbatim, not recomputed; C8/C8b/C9/C8n are computed
# fresh from the corrected pipeline (excludes grade_error/judge_error/
# unscored rows, same filter 29_content_arm_analysis.py uses).


def _content_arm_rates() -> dict:
    published = {
        "C0": {"base": (208, 19.2), "abl": (208, 43.3)},
        "C1b": {"base": (208, 22.1), "abl": (208, 41.8)},
        "C2": {"base": (208, 21.2), "abl": (208, 44.7)},
        "C3": {"base": (208, 24.5), "abl": (208, 42.3)},
    }

    def load(name):
        rows = _load(name)["rows"]
        return [r for r in rows if "turns" in r and "grade_error" not in r
                and "judge_error" not in r and not r.get("unscored_criteria")]

    base = load("content_arm_full_base_judged.json")
    abl = load("content_arm_full_abl_judged.json")

    def rate(rows, cond):
        vals = [bool(r["full_compliance_judged"]) for r in rows if r["condition"] == cond]
        return len(vals), 100 * sum(vals) / len(vals)

    out = dict(published)
    for cond in ("C8", "C8b", "C9", "C8n"):
        out[cond] = {"base": rate(base, cond), "abl": rate(abl, cond)}
    return out


def fig_content_arm() -> None:
    data = _content_arm_rates()
    for cond, d in data.items():
        print(f"content_arm  {cond:4s}  base {d['base'][1]:5.1f}% (n={d['base'][0]})"
              f"   abl {d['abl'][1]:5.1f}% (n={d['abl'][0]})")

    order = ["C0", "C1b", "C2", "C3", None, "C8", "C8b", "C9", "C8n"]
    # The two valence-flip pairs carry the actual pre-registered contrasts;
    # labelling their rates directly gives scale without needing colour to
    # flag them.
    labelled = {"C2", "C3", "C8", "C9"}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ys, labels = [], []
    y = len(order) - 1
    for cond in order:
        if cond is None:
            y -= 1
            continue
        a, b = data[cond]["base"][1], data[cond]["abl"][1]
        ax.plot([a, b], [y, y], "-", color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=90, color=INTACT, zorder=3)
        ax.scatter([b], [y], s=90, color=ABLATED, zorder=3)
        if cond in labelled:
            ax.annotate(f"{a:.1f}%", (a, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=INTACT)
            ax.annotate(f"{b:.1f}%", (b, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=ABLATED)
        ys.append(y)
        labels.append(cond)
        y -= 1

    ax.scatter([], [], s=90, color=INTACT, label="refusal intact")
    ax.scatter([], [], s=90, color=ABLATED, label="refusal ablated")
    ax.set_yticks(ys, labels)

    divider_y = (ys[3] + ys[4]) / 2  # between C3 and C8
    ax.axhline(divider_y, color=GRID, lw=1, zorder=0)
    ax.text(1.0, ys[0] + 0.55, "original design", transform=ax.get_yaxis_transform(),
            ha="left", fontsize=8.5, color=INK, alpha=0.55, style="italic")
    ax.text(1.0, ys[4] + 0.55, "task-content follow-on", transform=ax.get_yaxis_transform(),
            ha="left", fontsize=8.5, color=INK, alpha=0.55, style="italic")

    ax.set_xlim(0, 55)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_xlabel("Full compliance rate (%)", labelpad=10)
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Content-bearing peer framing: the original primary vs. the follow-on",
                 fontsize=12.5, fontweight="bold", pad=14)
    _save(fig, "content_arm")


# --- companion table, markdown, same numbers as the figure plus the ---------
# pre-registered contrast stats a chart can't show cleanly.


def table_content_arm() -> None:
    data = _content_arm_rates()
    rows = ["C0", "C1b", "C2", "C3", "C8", "C8b", "C9", "C8n"]
    board = {
        "C0": "none (baseline)", "C1b": "peer endorsement, speakers stripped",
        "C2": "peer endorsement, named peers", "C3": "the same peers, refusing",
        "C8": "task-named endorsement, named peers", "C8b": "same as C8, speakers stripped",
        "C9": "task-named refusal, named peers", "C8n": "task named, no valence (\"logged\")",
    }
    lines = [
        "| Condition | Board content | Base arm n | Base rate | Ablated arm n | Ablated rate |",
        "|---|---|---|---|---|---|",
    ]
    for c in rows:
        n_b, r_b = data[c]["base"]
        n_a, r_a = data[c]["abl"]
        lines.append(f"| {c} | {board[c]} | {n_b} | {r_b:.1f}% | {n_a} | {r_a:.1f}% |")

    lines += [
        "",
        "**Pre-registered contrasts** (paired McNemar, matched items only -- these deltas will not equal the raw-rate differences above, which are unpaired):",
        "",
        "| Contrast | Arm | Delta | p (item) | p (cluster) | Informative pairs |",
        "|---|---|---|---|---|---|",
        "| C2 - C3 | cross-arm interaction | +5.77pp | 0.0408 | -- | -- |",
        "| C2 - C3 | base (intact) | -3.37pp | 0.0654 | 0.0703 | -- |",
        "| C2 - C3 | ablated | +2.40pp | 0.3593 | 0.7905 | -- |",
        "| C8 - C9 | base (intact) | +1.94pp | 0.523 | 0.302 | 22 / 208 |",
        "| C8 - C9 | ablated | +6.90pp | 0.016 | 0.027 | 30 / 208 |",
        "| C8 - C9 | cross-arm interaction | +4.48pp | -- | 0.28 (perm.) | -- |",
        "| C8 - C9 | ablated, engagement-restricted | +7.14pp | -- | 0.027 | -- |",
    ]
    out = OUT / "content_arm_table.md"
    OUT.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print()
    print("\n".join(lines))


def main() -> None:
    fig_conditions()
    fig_content_arm()
    table_content_arm()


if __name__ == "__main__":
    main()
