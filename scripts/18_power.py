"""What effect size does this design actually exclude?

The write-up previously quoted a 15pp bound from a power table that deflated the
effective sample using the ICC of the compliance *level*. That is the wrong
quantity: for a within-item paired contrast what matters is whether the paired
*differences* cluster, and they do not. Everything here is computed from the run,
so no bound is ever a constant in a template again.

    uv run python scripts/18_power.py --in results/peer_loop_9b_judged.json \
        --out results/power_9b.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n * 2)


def icc(groups: dict[str, list[float]]) -> float:
    """One-way random-effects ICC. Reported for the level and for the difference,
    because quoting the first as if it governed the second is the original error."""
    flat = [x for v in groups.values() for x in v]
    grand, k = st.mean(flat), st.mean([len(v) for v in groups.values()])
    between = sum(len(v) * (st.mean(v) - grand) ** 2 for v in groups.values()) / (len(groups) - 1)
    within = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in groups.values()) / (len(flat) - len(groups))
    return (between - within) / (between + (k - 1) * within)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", default="C1b")
    ap.add_argument("--arm", default="C2")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    data = json.loads(args.path.read_text())
    by: dict = defaultdict(dict)
    for r in data["rows"]:
        if "turns" in r:
            by[(r["cluster"], r["id"])][r["condition"]] = r
    conds = data["meta"]["conditions"]
    items = {k: v for k, v in by.items() if len(v) == len(conds)}
    ks = list(items)
    full = lambda r: bool(r["full_compliance_judged"])

    diffs = [int(full(items[k][args.arm])) - int(full(items[k][args.ref])) for k in ks]
    n = len(diffs)
    mean = sum(diffs) / n
    half = 1.96 * st.stdev(diffs) / n**0.5

    by_cluster: dict = defaultdict(list)
    for k, dv in zip(ks, diffs):
        by_cluster[k[0]].append(dv)
    clusters = list(by_cluster)

    rng = random.Random(0)
    boot = []
    for _ in range(args.reps):
        flat = [x for _ in clusters for x in by_cluster[rng.choice(clusters)]]
        boot.append(sum(flat) / len(flat))
    boot.sort()
    lo, hi = boot[int(0.025 * args.reps)], boot[int(0.975 * args.reps)]

    levels: dict = defaultdict(list)
    for k in ks:
        levels[k[0]].append(float(full(items[k][args.ref])))

    # Power: resample clusters under an injected effect, applied to items that are
    # behaviourally live (the model acts in at least one arm). An effect cannot move
    # an item that never acts, so injecting it corpus-wide would overstate power.
    live = [k for k in ks if any(bool(items[k][c]["called"]) for c in conds)]
    base = {k: bool(full(items[k][args.ref])) for k in ks}
    curve = {}
    for eff in (2, 3, 5, 7.5, 10, 15):
        hits = 0
        for _ in range(args.sims):
            b = c = 0
            for k in ks:
                ref_v = base[k]
                arm_v = ref_v
                if k in live and not ref_v and rng.random() < eff / 100 * len(ks) / len(live):
                    arm_v = True
                b += arm_v and not ref_v
                c += ref_v and not arm_v
            hits += mcnemar_exact(b, c) < 0.05
        curve[eff] = hits / args.sims

    payload = {
        "model": data["meta"]["model"],
        "ref": args.ref, "arm": args.arm,
        "n_items": n, "n_clusters": len(clusters), "n_live_items": len(live),
        "delta_pp": 100 * mean,
        "ci_item_pp": [100 * (mean - half), 100 * (mean + half)],
        "ci_cluster_boot_pp": [100 * lo, 100 * hi],
        "icc_level": icc(levels),
        "icc_difference": icc({c: [float(x) for x in v] for c, v in by_cluster.items()}),
        "power_by_effect_pp": curve,
        "bootstrap_reps": args.reps, "power_sims": args.sims,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"delta {payload['delta_pp']:+.2f}pp")
    print(f"  item 95% CI      [{payload['ci_item_pp'][0]:+.2f}, {payload['ci_item_pp'][1]:+.2f}] pp")
    print(f"  cluster boot CI  [{payload['ci_cluster_boot_pp'][0]:+.2f}, {payload['ci_cluster_boot_pp'][1]:+.2f}] pp")
    print(f"  ICC level {payload['icc_level']:+.3f}   ICC difference {payload['icc_difference']:+.3f}")
    print("  power:", {k: round(v, 2) for k, v in curve.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
