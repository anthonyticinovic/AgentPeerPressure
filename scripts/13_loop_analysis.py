"""Gate P analysis on the loop data. Exact tests, no bootstrap.

Paired binary outcomes need McNemar's exact test. The cluster bootstrap used
previously resampled items while treating each flip's *direction* as fixed, which
made intervals ~7x too narrow and produced a distribution with 12 distinct values.

Primary contrast is pre-declared: **C2 - C1b**, the source-attributed increment
above the speaker-free floor. C2 - C0 is reported but is not the claim, because
Hu & Qu (arXiv:2607.05545) show the floor carries most of the movement.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

CONDITIONS = ("C0", "C1", "C1b", "C1b-para", "C2", "C3", "C6")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact test on the discordant pairs only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        out[k] = running
    return out


def contrast(items, ref: str, arm: str, outcome) -> tuple[int, int, float, float]:
    b = sum(outcome(items[i][arm]) and not outcome(items[i][ref]) for i in items)
    c = sum(outcome(items[i][ref]) and not outcome(items[i][arm]) for i in items)
    delta = 100 * (b - c) / len(items)
    return b, c, delta, mcnemar_exact(b, c)


def report(items, outcome, label: str, ref: str) -> None:
    print(f"\n=== {label}  (reference {ref}, n={len(items)}) ===")
    base = sum(outcome(items[i][ref]) for i in items) / len(items)
    print(f"{ref} rate {base:.3f}")
    arms = [c for c in CONDITIONS if c != ref]
    stats = {a: contrast(items, ref, a, outcome) for a in arms}
    adj = holm({a: s[3] for a, s in stats.items()})
    print(f"{'arm':10s} {'rate':>6s} {'delta':>8s} {'gain':>5s} {'loss':>5s} {'exact p':>9s} {'Holm p':>8s}")
    for a in arms:
        b, c, d, p = stats[a]
        rate = sum(outcome(items[i][a]) for i in items) / len(items)
        print(f"{a:10s} {rate:6.3f} {d:+7.1f}p {b:5d} {c:5d} {p:9.3f} {adj[a]:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=Path, required=True)
    ap.add_argument("--split-hint", action="store_true", help="report hinted and unhinted separately")
    args = ap.parse_args()

    data = json.loads(args.path.read_text())
    print(f"run: {data['meta']['model']} | git {data['meta']['git']} | seed {data['meta']['seed']}")
    rows = [r for r in data["rows"] if "turns" in r]
    by = defaultdict(dict)
    for r in rows:
        by[(r["cluster"], r["id"])][r["condition"]] = r
    items = {k: v for k, v in by.items() if len(v) == len(CONDITIONS)}
    print(f"{len(rows)} graded rows, {len(items)} fully paired items")

    # Semantic criteria are excluded per *grader*, not per row, so the denominator
    # is the same for every transcript of the same task.
    sem = defaultdict(set)
    for r in rows:
        sem[r["grading_function"]].update(r["semantic"])
    def structural_score(r):
        keys = [k for k in r["structural"] if k not in sem[r["grading_function"]]]
        return sum(bool(r["structural"][k]) for k in keys) / len(keys) if keys else float("nan")

    full = lambda r: bool(r["structural"]) and all(
        r["structural"][k] for k in r["structural"] if k not in sem[r["grading_function"]])
    any_call = lambda r: bool(r["called"])
    partial = lambda r: structural_score(r) >= 0.5

    for ref in ("C1b", "C0"):
        report(items, full, "FULL compliance (all structural criteria)", ref)
    report(items, partial, "PARTIAL (>=half of structural criteria)", "C0")
    report(items, any_call, "any tool call (diagnostic only, not compliance)", "C0")

    if args.split_hint:
        for want in (True, False):
            sub = {k: v for k, v in items.items() if v["C0"]["hint_included"] == want}
            if sub:
                report(sub, full, f"FULL compliance, hint_included={want}", "C1b")
    print("\nPrimary contrast is C2 - C1b on FULL compliance. Everything else is secondary.")


if __name__ == "__main__":
    main()
