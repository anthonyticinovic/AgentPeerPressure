"""The monitoring blind-spot cross-tab.

`r_arditi` (the causal refusal switch) and `r_harm` (the model's own harm-belief
readout) are separable directions -- Gate B2 established that at cos <= 0.9, and
`02_dual_directions.py` measured it directly (~0.10-0.13 at 4B). Ablating the switch
raises compliance; this script asks whether the belief readout still fires when it
does. A row that lands in "complied AND `p_harm_orth` still above the calibrated
harm threshold" is a direct, agentic-scale demonstration that ablation bypassed the
refusal *behaviour* without erasing the internal signal that would have driven it --
independent of whether the peer-framing manipulation itself does anything.

`tau_harm_orth` (calibrated by `scripts/23_calibrate_tau_orth.py`, frozen and never
re-tuned per condition or per arm -- see `calibrate_tau`'s docstring) is applied
exactly as stored: this script does not recompute or adjust it. Read
`23_calibrate_tau_orth.py`'s module docstring for the distribution-shift caveat
(calibrated on single-turn benign prompts, applied inside multi-turn agentic
transcripts) before treating a flagged rate here as more than suggestive.

Two statistics, not one, and they are not interchangeable:

  1. The 2x2 table (`crosstab`): counts of `full_compliance_judged` x
     `p_harm_orth > tau_harm_orth`, plus two intervals on the compliant-and-flagged
     cell's rate *of the whole arm* (k/n over every row, not just compliant ones) --
     "what fraction of ablated-arm rows are in the blind spot", the headline number.
     Rows collapse to ~52 base-scenario clusters (16-24 rows each; ICC~0.38 is
     documented for a related outcome in `cluster_sign_test`'s docstring and in
     `13_loop_analysis.py`/`19_ablation_analysis.py`), so a plain row-level Wilson
     interval overstates precision. The primary interval is therefore a cluster
     bootstrap (`cluster_bootstrap_ci`, same clustering key as `cluster_sign_test`);
     the naive Wilson interval is kept alongside, clearly labelled, for comparison
     only -- never as the headline number.
  2. The exploratory one-sample test (`exploratory_conditional_flag_rate`): among
     rows that *already* complied, is the flagged rate above the 5% FPR the
     threshold was calibrated at? This is a different, conditional question --
     "given the model complied, does the internal signal still carry more than
     chance-level information" -- and is explicitly a diagnostic, not part of the
     pre-registered Holm-corrected secondary family in `19_ablation_analysis.py`.
     Do not fold it into that family or report it alongside those p-values as if it
     were.

Out of scope by design (do not add here): any `r_ref`/`context_last` threshold --
this cross-tab is about the harm-belief signal specifically, not refusal-imminence,
which is `19_ablation_analysis.py`'s territory.

    uv run python scripts/24_blindspot_crosstab.py
    uv run python scripts/24_blindspot_crosstab.py --base results/gate_a_full_base_judged.json \\
        --abl results/gate_a_full_abl_judged.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.config import CFG  # noqa: E402
from pressure.monitor import Directions  # noqa: E402
from pressure.stats import binom_exact_p, cluster_bootstrap_ci, wilson_ci  # noqa: E402

SPARSE_CELL = 10  # below this count, a per-condition rate is descriptive only
# Keep in sync with 19_ablation_analysis.py's CONDS -- the full 9-condition scope
# (docs/STATE.md, 2026-08-28). Used only to self-flag a partial condition set below;
# never to filter or reshape the rows actually loaded.
EXPECTED_CONDS = ("C0", "C1", "C1b", "C2", "C3", "C4", "C5", "C6", "C7")


def comply(r: dict) -> bool:
    return bool(r["full_compliance_judged"])


def flagged(r: dict, tau: float) -> bool:
    return r["p_harm_orth"] > tau


def load_rows(path: Path) -> list[dict]:
    """Same guard as `19_ablation_analysis.py`/`22_turn1_lockin.py`: a row with no
    verdict (`grade_error`/`judge_error`) or no transcript is unusable, and a row
    without `p_harm_orth` (a monitor-off run) cannot be classified here at all."""
    payload = json.loads(path.read_text())
    rows = [r for r in payload["rows"]
            if "turns" in r and "grade_error" not in r and "judge_error" not in r]
    dropped = len(payload["rows"]) - len(rows)
    usable = [r for r in rows if "p_harm_orth" in r and "full_compliance_judged" in r]
    missing = len(rows) - len(usable)
    print(f"  {path.name}: {len(usable)} usable rows, {dropped} ungraded dropped, "
          f"{missing} missing p_harm_orth/full_compliance_judged dropped")
    return usable


def crosstab(rows: list[dict], tau: float, n_boot: int = 10000) -> dict:
    """2x2 counts, plus two intervals on the compliant-and-flagged cell's rate of the
    whole arm (k/n over every row passed in, not conditioned on compliance) -- they are
    not interchangeable.

    These rows collapse to a handful of base-scenario clusters (52 across the full
    ablated arm, 16-24 rows each), and this project already treats that as consequential
    elsewhere (`cluster_sign_test`, ICC~0.38 documented there and in
    `13_loop_analysis.py`/`19_ablation_analysis.py`). `wilson_ci` on the pooled counts
    treats every row as an independent draw and overstates precision here -- it is kept
    only as a labelled, secondary comparison. `cluster_bootstrap_ci` resamples whole
    clusters (same clustering key as `cluster_sign_test`: `row["cluster"]`, the base
    scenario) and is the primary, headline interval.
    """
    a = b = c = d = 0  # comply&flag, comply&~flag, refuse&flag, refuse&~flag
    clusters: set = set()
    for r in rows:
        cm, fl = comply(r), flagged(r, tau)
        clusters.add(r["cluster"])
        if cm and fl:
            a += 1
        elif cm:
            b += 1
        elif fl:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    lo, hi = wilson_ci(a, n)
    clo, chi = cluster_bootstrap_ci(rows, lambda r: r["cluster"],
                                     lambda r: comply(r) and flagged(r, tau), n_boot=n_boot)
    return {
        "n": n,
        "n_clusters": len(clusters),
        "comply_and_flagged": a, "comply_and_unflagged": b,
        "refuse_and_flagged": c, "refuse_and_unflagged": d,
        "comply_and_flagged_rate": a / n if n else float("nan"),
        "comply_and_flagged_ci95": [clo, chi],  # cluster bootstrap -- primary/headline
        "comply_and_flagged_ci95_wilson_naive": [lo, hi],  # ignores clustering -- secondary
    }


def exploratory_conditional_flag_rate(rows: list[dict], tau: float, p0: float) -> dict:
    """EXPLORATORY/DIAGNOSTIC -- not part of the pre-registered secondary family.

    Among rows that already complied, is the flagged rate above the FPR baseline the
    threshold was calibrated at? Two-sided exact binomial against p0.
    """
    compliant = [r for r in rows if comply(r)]
    k = sum(1 for r in compliant if flagged(r, tau))
    n = len(compliant)
    return {
        "n_compliant": n,
        "n_flagged_among_compliant": k,
        "rate": k / n if n else float("nan"),
        "p0_fpr_baseline": p0,
        "p_value_two_sided": binom_exact_p(k, n, p0) if n else float("nan"),
    }


def per_condition(rows: list[dict], tau: float) -> dict[str, dict]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    return {cond: crosstab(rs, tau) for cond, rs in sorted(by_cond.items())}


def report_arm(name: str, rows: list[dict], tau: float) -> dict:
    ct = crosstab(rows, tau)
    lo, hi = ct["comply_and_flagged_ci95"]
    wlo, whi = ct["comply_and_flagged_ci95_wilson_naive"]
    print(f"\n{name}  (n={ct['n']}, {ct['n_clusters']} clusters)")
    print(f"  {'':16}{'flagged':>10}{'unflagged':>12}")
    print(f"  {'complied':16}{ct['comply_and_flagged']:>10}{ct['comply_and_unflagged']:>12}")
    print(f"  {'refused':16}{ct['refuse_and_flagged']:>10}{ct['refuse_and_unflagged']:>12}")
    print(f"  comply-and-flagged rate (of all {ct['n']}): {100*ct['comply_and_flagged_rate']:.2f}%")
    print(f"    95% CI [{100*lo:.2f}%, {100*hi:.2f}%]  (cluster bootstrap over "
          f"{ct['n_clusters']} base scenarios -- primary)")
    print(f"    95% CI [{100*wlo:.2f}%, {100*whi:.2f}%]  (naive Wilson, ignores "
          "clustering -- secondary, for comparison only)")

    by_cond = per_condition(rows, tau)
    missing = [c for c in EXPECTED_CONDS if c not in by_cond]
    if missing:
        print(f"  WARNING: {len(by_cond)}/{len(EXPECTED_CONDS)} conditions present -- "
              f"missing {missing} (expected until the full 9-condition run has landed "
              "locally; every stat above is computed only over what is present)")
    print("  by condition (descriptive counts only -- no per-condition significance "
          "test; a cell under "
          f"{SPARSE_CELL} is sparse and reported, not suppressed):")
    for cond, c in by_cond.items():
        tag = " [SPARSE]" if c["comply_and_flagged"] < SPARSE_CELL else ""
        print(f"    {cond:6} n={c['n']:4d}  comply&flagged={c['comply_and_flagged']:<4d}"
              f" ({100*c['comply_and_flagged_rate']:5.2f}%){tag}")
    return {"crosstab": ct, "by_condition": by_cond}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=CFG.results_dir / "gate_a_full_base_judged.json")
    ap.add_argument("--abl", type=Path, default=CFG.results_dir / "gate_a_full_abl_judged.json")
    ap.add_argument("--json", type=Path, default=CFG.results_dir / "blindspot_crosstab.json")
    args = ap.parse_args()

    dirs = Directions.load(CFG.results_dir)
    if dirs.tau_harm_orth is None:
        raise SystemExit(
            f"tau_harm_orth is not calibrated for {dirs.model} yet -- run "
            "scripts/23_calibrate_tau_orth.py first (pass --iter if the loaded "
            "directions are the local iteration model)."
        )
    tau = dirs.tau_harm_orth
    print(f"model: {dirs.model}  tau_harm_orth: {tau:+.4f}", flush=True)

    print("\nloading")
    out: dict[str, dict] = {"model": dirs.model, "tau_harm_orth": tau,
                             "source": {"base": str(args.base), "abl": str(args.abl)}}
    rows_by_arm: dict[str, list[dict]] = {}
    for name, key, path in (("base (sanity check)", "base", args.base),
                             ("ablated (primary)", "ablated", args.abl)):
        payload = json.loads(path.read_text())
        meta_model = payload.get("meta", {}).get("model")
        if meta_model != dirs.model:
            raise SystemExit(
                f"{path} was generated with model {meta_model!r}, but tau_harm_orth is "
                f"calibrated for {dirs.model!r}. Applying a threshold calibrated on one "
                "model's activations to another model's projections is meaningless even "
                "when it does not error outright -- recalibrate at the matching scale, "
                "or point --base/--abl at data generated with the currently loaded "
                "directions' model."
            )
        rows = load_rows(path)
        rows_by_arm[key] = rows
        out[key] = report_arm(name, rows, tau)

    print("\nEXPLORATORY/DIAGNOSTIC -- not part of the pre-registered Holm-corrected "
          "secondary family in 19_ablation_analysis.py, not to be reported alongside "
          "those p-values as if it were:")
    excl = exploratory_conditional_flag_rate(rows_by_arm["ablated"], tau, CFG.tau_fpr)
    print(f"  ablated arm, rows where full_compliance_judged is true (n={excl['n_compliant']}): "
          f"{excl['n_flagged_among_compliant']} flagged "
          f"({100*excl['rate']:.2f}%) vs {100*CFG.tau_fpr:.1f}% FPR baseline, "
          f"two-sided exact binomial p={excl['p_value_two_sided']:.4f}")
    out["exploratory_conditional_flag_rate_ablated"] = excl

    base_ct = out["base"]["crosstab"]
    if base_ct["comply_and_flagged"] > 0:
        print(f"\n  NOTE: base arm has {base_ct['comply_and_flagged']} compliant-and-flagged "
              "rows despite refusal being intact -- this is worth investigating, not "
              "suppressing (see this script's module docstring).")

    args.json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
