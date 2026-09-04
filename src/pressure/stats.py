"""Paired-binary statistics shared by every contrast analysis in this project.

Extracted from 13_loop_analysis.py. Both the item-level and the cluster-level test
must be reported for every contrast: the corpus is 52 base scenarios x 4 prompt
variants, ICC ~0.38, so a contrast whose discordant pairs sit in one cluster is one
observation and not four. An analysis that reports only the item-level p-value is
anti-conservative by roughly a factor of two.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb, sqrt
from typing import Callable


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate k/n. Default z is the two-sided 95%
    quantile. Unlike the normal (Wald) interval, this stays inside [0, 1] and is not
    degenerate at k=0 or k=n -- both routine here, where a cell can be a handful of
    rows out of a few hundred.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def binom_exact_p(k: int, n: int, p0: float) -> float:
    """Two-sided exact binomial test: P(observing a count at least as extreme as k)
    under Binomial(n, p0). Mirrors `mcnemar_exact`'s style -- stdlib `math.comb`, no
    scipy. Sums the probability of every outcome no more likely than the observed one,
    the standard two-sided definition (not a doubled one-sided tail, which
    over/under-covers when p0 != 0.5).
    """
    if n == 0:
        return 1.0
    probs = [comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    p_obs = probs[k]
    # Relative, not absolute, tolerance. Found by adversarial review, 2026-09-04:
    # an absolute `eps=1e-12` degenerates into "sum everything below 1e-12" once
    # p_obs itself is far smaller than that (e.g. ~1e-24 on n=208), so every
    # sufficiently extreme k returned the same floored constant instead of its
    # own p-value. A relative tolerance scales with p_obs at any magnitude.
    eps = 1e-9
    return min(1.0, sum(p for p in probs if p <= p_obs * (1 + eps)))


def cluster_bootstrap_ci(rows: list[dict], cluster_key: Callable[[dict], object],
                          indicator: Callable[[dict], bool], n_boot: int = 10000,
                          seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a rate, resampling whole clusters rather than rows.

    Same clustering unit as `cluster_sign_test` -- `cluster_key(row)` groups by the
    base scenario, not the row. ICC ~0.38 there means row-level resampling (what a
    plain `wilson_ci` on pooled counts assumes) treats several correlated prompt
    variants as independent observations and understates the interval. Each bootstrap
    draw resamples clusters with replacement and pools every row inside the drawn
    clusters, so a cluster's own row count is preserved exactly as observed -- only
    which clusters appear, and how many times, is randomised. `indicator` counts the
    numerator (e.g. "complied and flagged"); every row in `rows` counts toward the
    denominator.
    """
    counts: dict = defaultdict(lambda: [0, 0])  # cluster -> [numerator, n]
    for r in rows:
        c = counts[cluster_key(r)]
        c[1] += 1
        if indicator(r):
            c[0] += 1
    clusters = list(counts.values())
    n_clusters = len(clusters)
    if n_clusters == 0:
        return (0.0, 1.0)

    rng = random.Random(seed)
    rates = []
    for _ in range(n_boot):
        k_sum = n_sum = 0
        for _ in range(n_clusters):
            k, n = clusters[rng.randrange(n_clusters)]
            k_sum += k
            n_sum += n
        if n_sum:
            rates.append(k_sum / n_sum)
    if not rates:
        return (0.0, 1.0)
    rates.sort()
    lo_i = int((alpha / 2) * (len(rates) - 1))
    hi_i = int((1 - alpha / 2) * (len(rates) - 1))
    return (rates[lo_i], rates[hi_i])


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
