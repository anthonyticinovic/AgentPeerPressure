"""Power for the actual interaction test — not a proxy for it.

`docs/writeup.md`'s Result 3 quotes a power table for the primary pre-registered
test (does the C2-C3 contrast change between ablation levels) that turned out not to
be reproducible from any committed script (adversarial review, 2026-08-28) — the
second time this project has shipped an unreproducible power table under this exact
description (the first was retracted, `docs/STATE.md:545`). `scripts/18_power.py`
computes power for a single arm-vs-ref contrast via exact McNemar; the interaction is
a *difference of two differences*, tested by `interaction()` in
`19_ablation_analysis.py` via a cluster-level sign-flip permutation, a different
statistic with materially larger variance. This script simulates power for THAT
test, specifically.

Method: take the real, observed per-item |d| magnitudes (d = the interaction test's
own paired difference-of-differences) and resample by drawing independent per-cluster
random SIGNS, exactly the operation `interaction()`'s own permutation null already
performs. This is "conditional power": fix the discordance this experiment actually
produced, simulate only over the randomness the test itself treats as noise. Its
calibration needs no argument beyond "this is literally the test's own null reference
distribution" — verified below anyway, because two prior parametric attempts in this
script's history both looked reasonable on paper and were not: v1 fixed C3 at its
real value and stochastically generated C2 from aggregate transition rates, which is
asymmetric between C2/C3 and gave power 0.258 at a true effect of 0. v2 (same idea as
here) calibrated close to nominal but wasn't validated precisely enough. v3 modelled
each item's C2/C3 pair as an "exchangeable" Bernoulli draw from a shared per-item rate
— exchangeable in principle, but empirically landed at power ~0.02 under the true
null (a >6-SD miss, confirmed with a 3000-sim direct check, not sampling noise) for
reasons not fully pinned down — plausibly the same discreteness/sparsity that makes
exact permutation tests with ~52 clustered units conservative in the first place,
compounding with the parametric model in some way this investigation didn't resolve.
Rather than trust an unresolved discrepancy, this version reverts to the
provably-correct resampling method and accepts its one real limitation honestly: the
injectable effect is capped by how much real discordance exists to redistribute. Past
that ceiling, this script reports "not assessable by this method" instead of a
number — a censored table is more honest than a wrong one.

    uv run python scripts/21_interaction_power.py --out results/interaction_power.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ABLATION = _load_module("ablation_analysis_19", Path(__file__).resolve().parent / "19_ablation_analysis.py")


def perm_p_batch(d: np.ndarray, idx: np.ndarray, n_clusters: int, reps: int, rng: np.random.Generator) -> float:
    """Vectorised reimplementation of `interaction()`'s permutation core.

    Identical math to `19_ablation_analysis.py::interaction()` — cluster-level sign
    flip, same two-sided count, same (1+count)/(1+reps) correction — but batches all
    `reps` draws into one array op instead of a Python `for` loop, because the sweep
    below calls this thousands of times where the source calls it once. Validated
    against the production function directly in `main()`, every run.
    """
    obs = float(d.mean())
    signs = np.where(rng.random((reps, n_clusters)) < 0.5, -1.0, 1.0)
    null = (signs[:, idx] * d).mean(axis=1)
    return (1 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (1 + reps)


def real_d_and_liveness(base: dict, abl: dict, ref: str, arm: str, keys: list) -> tuple[np.ndarray, np.ndarray]:
    """Real observed d per item, and whether the model engaged in any of the four
    rows involved (an effect cannot move an item that never acts)."""
    comply = ABLATION.comply
    base_ref = np.array([comply(base[k][ref]) for k in keys], dtype=float)
    base_arm = np.array([comply(base[k][arm]) for k in keys], dtype=float)
    abl_ref = np.array([comply(abl[k][ref]) for k in keys], dtype=float)
    abl_arm = np.array([comply(abl[k][arm]) for k in keys], dtype=float)
    d = (abl_arm - abl_ref) - (base_arm - base_ref)
    live = np.array([
        bool(base[k][ref].get("called")) or bool(base[k][arm].get("called")) or
        bool(abl[k][ref].get("called")) or bool(abl[k][arm].get("called"))
        for k in keys
    ])
    return d, live


def cluster_bias(target_pp: float, abs_d: np.ndarray, idx: np.ndarray, n_clusters: int,
                  live: np.ndarray, n_items: int) -> tuple[np.ndarray, float]:
    """Per-cluster P(sign=+1): 0.5 (unbiased) for clusters with no live item, else
    0.5 + b/2 where b is solved so the expected mean of the signed resample equals
    `target_pp`/100 — using only the |d| magnitude this experiment actually produced.
    Returns (per-cluster probability, achievable_ceiling_pp) — the ceiling is what
    b=1 (every live cluster forced positive) would give; `target_pp` is silently
    capped there and the caller must check for it rather than trust the number."""
    live_cluster = np.zeros(n_clusters, dtype=bool)
    live_cluster[idx[live]] = True
    s_live = abs_d[live_cluster[idx]].sum()
    ceiling_pp = 100.0 * s_live / n_items
    b = 0.0 if s_live == 0 else (target_pp / 100.0) * n_items / s_live
    b = max(-1.0, min(1.0, b))
    p = np.full(n_clusters, 0.5)
    p[live_cluster] = 0.5 + b / 2.0
    return p, ceiling_pp


def simulate_d(abs_d: np.ndarray, idx: np.ndarray, cluster_p: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """One resample: independent per-cluster sign draw at `cluster_p`, applied to the
    real |d| magnitudes. At cluster_p == 0.5 everywhere this is exactly a draw from
    `interaction()`'s own null reference distribution."""
    n_clusters = len(cluster_p)
    signs = np.where(rng.random(n_clusters) < cluster_p, 1.0, -1.0)
    return signs[idx] * abs_d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("results/gate_a_full_base_judged.json"))
    ap.add_argument("--abl", type=Path, default=Path("results/gate_a_full_abl_judged.json"))
    ap.add_argument("--out", type=Path, default=Path("results/interaction_power.json"))
    ap.add_argument("--ref", default=ABLATION.PRIMARY_REF)
    ap.add_argument("--arm", default=ABLATION.PRIMARY_ARM)
    ap.add_argument("--effects", default="0,5,10,15,20,25,30",
                     help="candidate true interaction effect sizes, pp")
    ap.add_argument("--sims", type=int, default=4000, help="simulated datasets per effect size")
    ap.add_argument("--reps", type=int, default=20000, help="permutation draws per simulated dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = ABLATION.load(args.base, ABLATION.CONDS)
    abl = ABLATION.load(args.abl, ABLATION.CONDS)
    keys = sorted(set(base) & set(abl))
    clusters = sorted({k[0] for k in keys})
    pos = {c: i for i, c in enumerate(clusters)}
    idx = np.array([pos[k[0]] for k in keys])
    n_items, n_clusters = len(keys), len(clusters)

    d_real, live = real_d_and_liveness(base, abl, args.ref, args.arm, keys)
    abs_d = np.abs(d_real)
    n_live = int(live.sum())

    # --- validation: fast vectorised permutation vs. the actual production function ---
    obs_mine = float(d_real.mean()) * 100
    obs_real, p_real = ABLATION.interaction(base, abl, args.ref, args.arm, reps=args.reps)
    rng = np.random.default_rng(args.seed)
    p_mine = perm_p_batch(d_real, idx, n_clusters, args.reps, rng)
    print(f"validation: obs mine={obs_mine:+.4f}pp  production={obs_real:+.4f}pp  "
          f"(must match to float precision)")
    print(f"            p    mine={p_mine:.4f}      production={p_real:.4f}  "
          f"(MC agreement expected, not exact)")
    if abs(obs_mine - obs_real) > 1e-6:
        raise SystemExit("validation FAILED: observed effect does not match the production "
                          "interaction() function — do not trust the sweep below.")
    if abs(p_mine - p_real) > 0.03:
        print("  WARNING: p-value agreement is looser than expected MC noise at this rep "
              "count — investigate before trusting the sweep.")

    # --- the actual power sweep ---
    effects = [float(x) for x in args.effects.split(",")]
    curve: dict[str, float | None] = {}
    ceiling_pp = None
    for eff in effects:
        cluster_p, ceiling_pp = cluster_bias(eff, abs_d, idx, n_clusters, live, n_items)
        beyond_ceiling = eff > ceiling_pp
        if beyond_ceiling:
            curve[str(eff)] = None
            print(f"  eff={eff:5.1f}pp  power=  n/a   [beyond this experiment's own "
                  f"discordance ceiling of {ceiling_pp:.1f}pp — not assessable by this method]")
            continue
        rng = np.random.default_rng(args.seed + 1 + int(eff * 10))
        hits = 0
        for _ in range(args.sims):
            d = simulate_d(abs_d, idx, cluster_p, rng)
            p = perm_p_batch(d, idx, n_clusters, args.reps, rng)
            hits += p < 0.05
        curve[str(eff)] = hits / args.sims
        tag = "  <- calibration (should be ~0.05)" if eff == 0 else ""
        print(f"  eff={eff:5.1f}pp  power={curve[str(eff)]:.3f}{tag}")

    calib = curve.get("0.0", curve.get("0"))
    calibrated = calib is not None and 0.02 <= calib <= 0.10
    if calib is not None and not calibrated:
        print(f"  WARNING: calibration check {calib:.3f} is outside [0.02, 0.10] at n_sims="
              f"{args.sims} — the simulation may be biased; do not quote this table as-is.")

    payload = {
        "ref": args.ref, "arm": args.arm,
        "n_items": n_items, "n_clusters": n_clusters, "n_live_items": n_live,
        "validation": {"obs_mine_pp": obs_mine, "obs_production_pp": obs_real,
                        "p_mine": p_mine, "p_production": p_real},
        "calibration_power_at_0pp": calib, "calibration_ok": calibrated,
        "achievable_effect_ceiling_pp": ceiling_pp,
        "power_by_interaction_pp": curve,
        "method": "conditional power: real |d| magnitudes, per-cluster sign resampled "
                  "at 0.5 (null) or biased toward the target effect (live clusters "
                  "only); effects beyond this run's own discordance ceiling are "
                  "reported as null (not assessable), never extrapolated",
        "sims_per_effect": args.sims, "perm_reps_per_sim": args.reps, "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
