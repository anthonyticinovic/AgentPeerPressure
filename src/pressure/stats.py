"""Paired-binary statistics shared by every contrast analysis in this project.

Extracted from 13_loop_analysis.py. Both the item-level and the cluster-level test
must be reported for every contrast: the corpus is 52 base scenarios x 4 prompt
variants, ICC ~0.38, so a contrast whose discordant pairs sit in one cluster is one
observation and not four. An analysis that reports only the item-level p-value is
anti-conservative by roughly a factor of two.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb


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


def cluster_sign_test(items, ref: str, arm: str, outcome) -> float:
    """Exact test at the level of base scenarios, not prompt variants.

    Items are 4 variants of one scenario and are not independent; ICC ~0.38 means
    the effective sample is about half the nominal one. A contrast whose discordant
    pairs all sit in one cluster is one observation, not four.
    """
    per = defaultdict(int)
    for k, v in items.items():
        per[k[0]] += int(bool(outcome(v[arm]))) - int(bool(outcome(v[ref])))
    b = sum(1 for d in per.values() if d > 0)
    c = sum(1 for d in per.values() if d < 0)
    return mcnemar_exact(b, c)
