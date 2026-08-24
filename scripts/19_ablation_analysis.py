"""Gate A: did ablation restore dynamic range, and does framing matter once it has?

Three questions in order. The second is only meaningful if the first passes.

  1. Does ablation raise compliance and grow the informative-item count, per category?
     Gate P's null rested on 26 informative items of 208 -- 154 never complied in any
     arm, 28 always did -- so the corpus had almost no dynamic range and no framing
     manipulation of any strength could have been detected. If ablation does not grow
     that set, the experiment failed and this script says so before anything else.

  2. Does the framing contrast differ between ablation levels?
     Cluster-level permutation. This is the inertness-vs-masking test.

     The pre-registered primary is **C2 - C3**: peers who complied against peers who
     refused. C2 - C1b tests only whether *naming* the peers matters, since C2 is
     literally C1b plus author labels, and Gate P measured that at +0.9pp.

  3. What do r_harm and r_ref do, split by ablation level and realised outcome?

Every projection is reported orthogonal to r_arditi. Ablation zeroes the residual
stream along r_arditi, so a raw projection loses cos(r_arditi, u) * (h . r_arditi) by
arithmetic rather than by measurement. `p_arditi` is a fidelity check that the hook
fired and never enters a contrast.

Statistics come from `pressure.stats`, shared with `13_loop_analysis.py`. Both the
item-level and the cluster-level test are reported for every contrast: 52 base
scenarios x 4 prompt variants, ICC ~0.38.

    uv run python scripts/19_ablation_analysis.py --harm-drift-bound 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.config import CFG  # noqa: E402
from pressure.stats import cluster_sign_test, contrast, holm  # noqa: E402

CONDS = ("C0", "C1b", "C2", "C3", "C6", "C7")
REF = "C1b"
PRIMARY_ARM, PRIMARY_REF = "C2", "C3"
RNG = np.random.default_rng(0)


def comply(r: dict) -> bool:
    return bool(r["full_compliance_judged"])


def load(path: Path, conds: tuple[str, ...]) -> dict:
    """Paired items only, with the guards `13_loop_analysis.py` carries.

    A row with `grade_error` has no verdict field at all, so indexing it raises rather
    than quietly scoring zero. Rows with unscored criteria are kept but counted: a
    declined judgement is not a non-compliance, and pooling it as one biases toward
    the null exactly where content is most harmful.
    """
    payload = json.loads(path.read_text())
    rows = [r for r in payload["rows"] if "turns" in r and "grade_error" not in r]
    dropped = len(payload["rows"]) - len(rows)
    unscored = sum(1 for r in rows if r.get("unscored_criteria"))

    by: dict = defaultdict(dict)
    for r in rows:
        by[(r["cluster"], r["id"])][r["condition"]] = r
    paired = {k: v for k, v in by.items() if all(c in v for c in conds)}

    print(f"  {path.name}: {len(paired)} paired items, {dropped} ungraded rows dropped")
    if unscored:
        print(f"    WARNING {unscored} rows carry unscored criteria — reported, not pooled")
    if len(paired) < len(by):
        print(f"    {len(by) - len(paired)} items dropped for incomplete condition coverage")
    return paired


def informative_by_category(data: dict) -> dict[str, int]:
    """Items that differ across arms, per category.

    Compliance can rise under ablation while the graders still cap it. A category-level
    count shows whether range was restored everywhere or only where it already existed.
    """
    out: dict[str, int] = {}
    for v in data.values():
        cat = v[CONDS[0]]["category"]
        vals = [comply(v[c]) for c in CONDS]
        out.setdefault(cat, 0)
        if any(vals) and not all(vals):
            out[cat] += 1
    return out


def interaction(base: dict, abl: dict, ref: str, arm: str,
                reps: int = 20000) -> tuple[float, float]:
    """Does the arm-vs-ref difference change between ablation levels?

    The exchangeable unit is the **cluster**, not the item. Flipping per item treats
    four correlated prompt variants as four observations and is anti-conservative at
    ICC ~0.38.

    Per item, d = (arm - ref | ablated) - (arm - ref | baseline). Swapping the framing
    labels for a cluster negates d for every item in it, and does so identically at
    both ablation levels -- so the pairing and the ablation main effect both survive,
    and only the framing contrast is randomised.
    """
    keys = sorted(set(base) & set(abl))
    clusters = sorted({k[0] for k in keys})
    pos = {c: i for i, c in enumerate(clusters)}
    idx = np.array([pos[k[0]] for k in keys])

    d = np.array([
        (comply(abl[k][arm]) - comply(abl[k][ref]))
        - (comply(base[k][arm]) - comply(base[k][ref]))
        for k in keys
    ], float)
    obs = float(d.mean())

    null = np.empty(reps)
    for i in range(reps):
        sign = np.where(RNG.random(len(clusters)) < 0.5, -1.0, 1.0)
        null[i] = float((sign[idx] * d).mean())
    # (1 + count) / (1 + reps): a permutation p-value can never legitimately be zero.
    p = (1 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (1 + reps)
    return 100 * obs, float(p)


def monitor_summary(data: dict, field: str) -> dict[str, float]:
    """Mean projection split by realised outcome.

    `p_harm` and `p_harm_orth` sit on the row: they are constant within a row by causal
    attention, so the loop stores them once. `p_ref*` and `p_arditi` vary per turn and
    are read at **turn 1**, the pre-registered readout -- that is the turn whose context
    matches the single-turn prompts the directions were extracted on.
    """
    out: dict[str, list[float]] = {"complied": [], "refused": []}
    for v in data.values():
        for cond in CONDS:
            r = v[cond]
            if field in r:
                val = r[field]
            else:
                trace = r.get("monitor") or []
                if not trace:
                    continue
                val = trace[0][field]
            out["complied" if comply(r) else "refused"].append(val)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in out.items()}


def ref_trajectory(data: dict) -> dict[str, float]:
    """Fixed-width summaries of the per-turn p_ref_orth series.

    The series are ragged (1 to max_turns) and turn 3 of a 4-turn row is not the same
    thing as turn 3 of a 16-turn row, so they are never compared position by position.
    """
    firsts, lasts, maxes, means = [], [], [], []
    for v in data.values():
        for cond in CONDS:
            trace = v[cond].get("monitor") or []
            if not trace:
                continue
            series = [t["p_ref_orth"] for t in trace]
            firsts.append(series[0])
            lasts.append(series[-1])
            maxes.append(max(series))
            means.append(float(np.mean(series)))
    if not firsts:
        return {}
    return {"first": float(np.mean(firsts)), "last": float(np.mean(lasts)),
            "max": float(np.mean(maxes)), "mean": float(np.mean(means))}


def harm_drift(data: dict) -> float:
    """Largest within-row p_harm deviation, relative to the row's own magnitude.

    p_harm must be constant within a row. Any variation is floating-point noise from
    attention kernels tiling differently at different sequence lengths -- and it would
    look exactly like a trajectory to anyone who plotted it.
    """
    worst = 0.0
    for v in data.values():
        for r in v.values():
            if "p_harm" not in r:
                continue
            scale = abs(r["p_harm"]) or 1.0
            for t in r.get("monitor") or []:
                worst = max(worst, abs(t["p_harm"] - r["p_harm"]) / scale)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=CFG.results_dir / "gate_a_pilot_base_judged.json")
    ap.add_argument("--abl", type=Path, default=CFG.results_dir / "gate_a_pilot_abl_judged.json")
    ap.add_argument("--json", type=Path, default=CFG.results_dir / "gate_a_analysis.json")
    ap.add_argument("--harm-drift-bound", type=float, default=1e-4,
                    help="relative bound established by Gate A2")
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()

    print("loading")
    base, abl = load(args.base, CONDS), load(args.abl, CONDS)
    keys = sorted(set(base) & set(abl))
    if not keys:
        raise SystemExit("no items present in both runs")
    base = {k: base[k] for k in keys}
    abl = {k: abl[k] for k in keys}
    n_clusters = len({k[0] for k in keys})
    print(f"  {len(keys)} items in both runs, {n_clusters} clusters\n")

    # --- 1. dynamic range ---------------------------------------------------
    print("1. DYNAMIC RANGE")
    levels = []
    for name, data in (("base", base), ("ablated", abl)):
        rate = {c: 100 * float(np.mean([comply(data[k][c]) for k in keys])) for c in CONDS}
        by_cat = informative_by_category(data)
        info = sum(by_cat.values())
        levels.append({"level": name, "rate": rate, "informative": info,
                       "informative_by_category": by_cat})
        print(f"  {name:8} " + "  ".join(f"{c}={rate[c]:5.1f}%" for c in CONDS)
              + f"   informative = {info}/{len(keys)}")
        print("           " + "  ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    grew = levels[1]["informative"] > levels[0]["informative"]
    if not grew:
        print("\n  *** Ablation did not grow the informative set. The corpus has no "
              "dynamic range even without refusal. The contrasts below cannot be "
              "interpreted. ***")

    # --- 2. contrasts and interactions --------------------------------------
    print("\n2. CONTRASTS  (item-level p, then cluster-level)")
    pvals: dict[str, float] = {}
    detail: dict[str, dict] = {}
    pairs = [(REF, a) for a in CONDS if a != REF] + [(PRIMARY_REF, PRIMARY_ARM)]
    for level, data in (("base", base), ("ablated", abl)):
        for ref, arm in pairs:
            b, c, delta, p = contrast(data, ref, arm, comply)
            cp = cluster_sign_test(data, ref, arm, comply)
            name = f"{level}:{arm}-{ref}"
            pvals[name] = p
            detail[name] = {"delta_pp": delta, "b": b, "c": c, "discordant": b + c,
                            "p": p, "cluster_p": cp}
            flag = "   <- peer behaviour" if (ref, arm) == (PRIMARY_REF, PRIMARY_ARM) else ""
            print(f"  {name:18} {delta:+6.2f}pp  b={b:<3} c={c:<3} "
                  f"p={p:.4f}  cluster_p={cp:.4f}{flag}")

    print("\n   INTERACTIONS  (cluster-permuted, positive = framing matters more "
          "once refusal is ablated)")
    for label, (ref, arm) in (("PRIMARY", (PRIMARY_REF, PRIMARY_ARM)),
                              ("naming", (REF, "C2")),
                              ("handoff", (REF, "C7"))):
        d_int, p_int = interaction(base, abl, ref, arm, reps=args.reps)
        name = f"interaction:{arm}-{ref}"
        pvals[name] = p_int
        detail[name] = {"delta_pp": d_int, "p": p_int, "label": label}
        print(f"  {name:18} {d_int:+6.2f}pp  p={p_int:.4f}   [{label}]")

    # One designated primary, reported uncorrected and frozen before any data existed.
    # Six conditions produce enough contrasts that a single flat family would multiply
    # the primary by 15 and leave the experiment unable to conclude anything.
    primary = f"interaction:{PRIMARY_ARM}-{PRIMARY_REF}"
    secondary = {k: v for k, v in pvals.items() if k != primary}
    adj = holm(secondary)
    detail[primary]["holm"] = None
    print(f"\n  PRIMARY, pre-registered and uncorrected:")
    print(f"    {primary:18} p = {pvals[primary]:.4f}")
    print(f"  Holm across the {len(secondary)} secondary tests:")
    for name in secondary:
        detail[name]["holm"] = adj[name]
    ranked = sorted(adj.items(), key=lambda kv: kv[1])
    for name, v in ranked[:6]:
        print(f"    {name:18} {v:.4f}")
    if len(ranked) > 6:
        print(f"    ... {len(ranked) - 6} further tests, none smaller")

    # --- 3. monitors --------------------------------------------------------
    print("\n3. DIRECTION MONITORS  (orthogonal to r_arditi; p_ref at turn 1)")
    for name, data in (("base", base), ("ablated", abl)):
        drift = harm_drift(data)
        verdict = "ok" if drift <= args.harm_drift_bound else "EXCEEDS BOUND"
        print(f"  {name:8} max within-row p_harm drift {drift:.2e} "
              f"(bound {args.harm_drift_bound:.0e}) — {verdict}")

    mon: dict[str, dict] = {}
    for name, data in (("base", base), ("ablated", abl)):
        mon[name] = {f: monitor_summary(data, f)
                     for f in ("p_harm_orth", "p_ref_orth", "p_harm", "p_ref", "p_arditi")}
        mon[name]["p_ref_orth_trajectory"] = ref_trajectory(data)
        for f in ("p_harm_orth", "p_ref_orth"):
            v = mon[name][f]
            print(f"  {name:8} {f:12} complied={v['complied']:+9.2f}  "
                  f"refused={v['refused']:+9.2f}")
    print(f"\n  fidelity check, never a result — p_arditi on refused rows: "
          f"{mon['base']['p_arditi']['refused']:+.3f} -> "
          f"{mon['ablated']['p_arditi']['refused']:+.3f}")

    args.json.write_text(json.dumps({
        "source": {"base": str(args.base), "abl": str(args.abl)},
        "n_items": len(keys), "n_clusters": n_clusters,
        "conditions": list(CONDS),
        "primary": primary,
        "informative_set_grew": grew,
        "dynamic_range": levels,
        "tests": detail,
        "monitors": mon,
        "harm_drift_bound": args.harm_drift_bound,
    }, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
