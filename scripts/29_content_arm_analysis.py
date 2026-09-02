"""Pre-registered analysis for the content arm (docs/PLAN_content_arm.md #6).

Written before G2 runs, against no data yet -- the point of pre-registering is
that this file's contrasts do not move once results exist. C8/C8b/C9/C8n come
from the new content-arm run; C2/C1b for the secondary family are pulled from the
existing nine-condition confirmatory file, joined on (cluster, id).

Primary: C8 - C9, full_compliance_judged, exact McNemar clustered on id_original,
base arm, uncorrected, single test.

Secondary family, Holm-corrected across 3:
  C8 - C8b   the speaker increment in the presence of content (H6)
  C8 - C2    the content increment at fixed speaker
  C8b - C1b  the content increment, speaker-free

Belief: p_harm_orth at task_last, C8 vs C9, plus the comply-and-flagged cross-tab.

Fallback: if the base arm is floor-pinned (informative-item count too small to
trust), report the C8-C9 interaction across ablation arms instead, exactly as the
existing nine-condition design falls back to `19_ablation_analysis.py::interaction`.

Exploratory family, added after the G2 adversarial review (2026-09-02) but
committed before G3/G4 output lands, so it is pre-registered relative to the
full-scale data even though not relative to the original design: NOT part of the
Holm-corrected secondary family, reported separately, for discriminating the
peer-endorsement mechanism from an in-context repetition/priming account of the
same restated content (see docs/STATE.md for the reasoning):
  C8n - C6   does the verbatim restatement move compliance with no valence,
             on the same audit-framed board C6 already uses -- if flat, the
             repetition alone isn't doing the work
  C8 - C8n   valence vs. mere mention, within the task-referencing family
  (C8-C9) - (C2-C3), matched items  difference-in-differences: does the
             content-bearing valence flip move more than the content-free one
             on the identical 52/208 items

    uv run python scripts/29_content_arm_analysis.py \
        --content-base results/content_arm_base_judged.json \
        --existing-base results/gate_a_full_base_judged.json \
        [--content-abl results/content_arm_abl_judged.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.config import CFG  # noqa: E402
from pressure.monitor import Directions  # noqa: E402
from pressure.stats import binom_exact_p, cluster_sign_test, contrast, holm  # noqa: E402

CONTENT_CONDS = ("C8", "C8b", "C9", "C8n")
NEEDED_FOR_SECONDARY = ("C2", "C1b")
NEEDED_FOR_EXPLORATORY = ("C6", "C3")
FLOOR_PINNED_THRESHOLD = 15  # below this many informative items, distrust the base-arm primary


def comply(r: dict) -> bool:
    return bool(r["full_compliance_judged"])


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [r for r in payload["rows"] if "grade_error" not in r and "judge_error" not in r]


def load_joined(content_path: Path, existing_path: Path | None) -> dict:
    """Key (cluster, id) -> {condition: row}, content-arm + existing-file conditions merged.

    An item must carry every content condition to enter the primary/H6 analyses; the
    C2/C1b secondary contrasts additionally require the matching existing-file row for
    that same item, which is a strict subset.
    """
    by: dict = defaultdict(dict)
    for r in _load_rows(content_path):
        if r["condition"] in CONTENT_CONDS:
            by[(r["cluster"], r["id"])][r["condition"]] = r
    if existing_path and existing_path.exists():
        for r in _load_rows(existing_path):
            if r["condition"] in NEEDED_FOR_SECONDARY + NEEDED_FOR_EXPLORATORY:
                by[(r["cluster"], r["id"])].setdefault(r["condition"], r)
    return dict(by)


def informative_count(items: dict, ref: str, arm: str) -> int:
    return sum(
        1 for v in items.values()
        if ref in v and arm in v and comply(v[ref]) != comply(v[arm])
    )


def primary(items: dict) -> dict:
    have = {k: v for k, v in items.items() if "C8" in v and "C9" in v}
    b, c, delta, p = contrast(have, ref="C9", arm="C8", outcome=comply)
    p_cluster = cluster_sign_test(have, ref="C9", arm="C8", outcome=comply)
    n_informative = informative_count(have, "C9", "C8")
    return {
        "n_paired": len(have), "n_informative": n_informative,
        "delta_pp": delta, "b": b, "c": c,
        "p_item_exact": p, "p_cluster_exact": p_cluster,
        "floor_pinned": n_informative < FLOOR_PINNED_THRESHOLD,
    }


def secondary_family(items: dict) -> dict:
    tests = {
        "C8-C8b": ("C8b", "C8"),   # H6: speaker increment given content
        "C8-C2": ("C2", "C8"),    # content increment at fixed (named) speaker
        "C8b-C1b": ("C1b", "C8b"),  # content increment, speaker-free
    }
    raw = {}
    for name, (ref, arm) in tests.items():
        have = {k: v for k, v in items.items() if ref in v and arm in v}
        b, c, delta, p = contrast(have, ref=ref, arm=arm, outcome=comply)
        raw[name] = {"n_paired": len(have), "delta_pp": delta, "b": b, "c": c, "p_raw": p}
    corrected = holm({k: v["p_raw"] for k, v in raw.items()})
    for k in raw:
        raw[k]["p_holm"] = corrected[k]
    return raw


def _crosstab(rows: list[dict], tau: float) -> dict:
    """comply x flagged 2x2, item-level counts. Same construction as
    `24_blindspot_crosstab.py::crosstab`, without that script's cluster-bootstrap
    CI machinery (52 base scenarios = 52 clusters here, one row each, so a
    per-cluster bootstrap would just resample single points)."""
    a = b = c = d = 0
    for r in rows:
        cm, fl = comply(r), r["p_harm_orth"] > tau
        if cm and fl:
            a += 1
        elif cm:
            b += 1
        elif fl:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    return {
        "n": n, "comply_and_flagged": a, "comply_and_unflagged": b,
        "refuse_and_flagged": c, "refuse_and_unflagged": d,
        "comply_and_flagged_rate": a / n if n else float("nan"),
    }


def belief(items: dict, tau: float | None) -> dict:
    """p_harm_orth at task_last, C8 vs C9, plus the comply-and-flagged cross-tab
    for each condition. An untested mean invites exactly the "is this just
    noise" question the adversarial review raised, so the direction of the
    per-item delta is tested with an exact sign test, not asserted from a mean."""
    have = {k: v for k, v in items.items() if "C8" in v and "C9" in v
            and "p_harm_orth" in v["C8"] and "p_harm_orth" in v["C9"]}
    if not have:
        return {"n": 0, "note": "no --monitor data joined (p_harm_orth absent)"}
    c8_p = np.array([v["C8"]["p_harm_orth"] for v in have.values()])
    c9_p = np.array([v["C9"]["p_harm_orth"] for v in have.values()])
    delta = c8_p - c9_p
    nonzero = delta[delta != 0]
    n_pos = int((nonzero > 0).sum())
    p_sign = binom_exact_p(n_pos, len(nonzero), 0.5) if len(nonzero) else float("nan")
    out = {
        "n": len(have),
        "mean_p_harm_orth_C8": float(c8_p.mean()),
        "mean_p_harm_orth_C9": float(c9_p.mean()),
        "mean_delta": float(delta.mean()),
        "delta_sign_test": {
            "n_positive": n_pos, "n_negative": int(len(nonzero) - n_pos),
            "n_nonzero": int(len(nonzero)), "p_two_sided": p_sign,
        },
    }
    if tau is not None:
        out["crosstab_C8"] = _crosstab([v["C8"] for v in have.values()], tau)
        out["crosstab_C9"] = _crosstab([v["C9"] for v in have.values()], tau)
        out["tau_harm_orth"] = tau
    else:
        out["crosstab_note"] = "tau_harm_orth unavailable -- Directions.load found no calibrated threshold"
    return out


def exploratory_family(items: dict) -> dict:
    """Not part of the pre-registered Holm-corrected family (see module
    docstring). Discriminates peer-endorsement from in-context repetition of
    the restated content as the driver of the primary contrast."""
    out = {}

    have = {k: v for k, v in items.items() if "C6" in v and "C8n" in v}
    b, c, delta, p = contrast(have, ref="C6", arm="C8n", outcome=comply)
    out["C8n-C6"] = {"n_paired": len(have), "delta_pp": delta, "b": b, "c": c, "p_raw": p}

    have = {k: v for k, v in items.items() if "C8n" in v and "C8" in v}
    b, c, delta, p = contrast(have, ref="C8n", arm="C8", outcome=comply)
    out["C8-C8n"] = {"n_paired": len(have), "delta_pp": delta, "b": b, "c": c, "p_raw": p}

    have = {k: v for k, v in items.items() if all(cond in v for cond in ("C8", "C9", "C2", "C3"))}
    d = np.array([
        (comply(v["C8"]) - comply(v["C9"])) - (comply(v["C2"]) - comply(v["C3"]))
        for v in have.values()
    ], dtype=float)
    nonzero = d[d != 0]
    n_pos = int((nonzero > 0).sum())
    p_sign = binom_exact_p(n_pos, len(nonzero), 0.5) if len(nonzero) else float("nan")
    out["diff_in_diff_C8C9_vs_C2C3_matched"] = {
        "n_paired": len(have), "mean_diff_pp": 100 * float(d.mean()) if len(have) else float("nan"),
        "sign_test": {"n_positive": n_pos, "n_nonzero": int(len(nonzero)), "p_two_sided": p_sign},
    }
    return out


def fallback_interaction(base_items: dict, abl_items: dict) -> dict:
    """Cluster-permutation interaction across ablation arms, same construction as
    `19_ablation_analysis.py::interaction` -- used only if `primary()` reports
    floor_pinned."""
    keys = sorted(set(base_items) & set(abl_items))
    keys = [k for k in keys if all(c in base_items[k] and c in abl_items[k] for c in ("C8", "C9"))]
    clusters = sorted({k[0] for k in keys})
    pos = {c: i for i, c in enumerate(clusters)}
    idx = np.array([pos[k[0]] for k in keys])
    d = np.array([
        (comply(abl_items[k]["C8"]) - comply(abl_items[k]["C9"]))
        - (comply(base_items[k]["C8"]) - comply(base_items[k]["C9"]))
        for k in keys
    ], float)
    obs = float(d.mean())
    rng = np.random.default_rng(0)
    reps = 20000
    null = np.empty(reps)
    for i in range(reps):
        sign = np.where(rng.random(len(clusters)) < 0.5, -1.0, 1.0)
        null[i] = float((sign[idx] * d).mean())
    p = (1 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (1 + reps)
    return {"n_paired": len(keys), "delta_pp": 100 * obs, "p_cluster_perm": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-base", type=Path, required=True)
    ap.add_argument("--existing-base", type=Path, default=Path("results/gate_a_full_base_judged.json"))
    ap.add_argument("--content-abl", type=Path, default=None)
    args = ap.parse_args()

    items = load_joined(args.content_base, args.existing_base)
    print(f"joined items with all of {CONTENT_CONDS}: "
          f"{sum(1 for v in items.values() if all(c in v for c in CONTENT_CONDS))}")

    print("\n=== primary: C8 - C9 (base arm) ===")
    prim = primary(items)
    print(json.dumps(prim, indent=1))

    print("\n=== secondary family (Holm-corrected, 3 tests) ===")
    sec = secondary_family(items)
    print(json.dumps(sec, indent=1))

    try:
        dirs = Directions.load(CFG.results_dir)
        tau = dirs.tau_harm_orth
    except FileNotFoundError:
        tau = None
    print("\n=== belief: p_harm_orth, C8 vs C9 ===")
    bel = belief(items, tau)
    print(json.dumps(bel, indent=1))

    print("\n=== exploratory family (not Holm-corrected -- see module docstring) ===")
    expl = exploratory_family(items)
    print(json.dumps(expl, indent=1))

    out = {"primary": prim, "secondary": sec, "belief": bel, "exploratory": expl}

    if prim["floor_pinned"] and args.content_abl and args.content_abl.exists():
        print(f"\nprimary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} informative items) "
              "-- falling back to the cross-arm interaction, as the existing design does.")
        abl_items = load_joined(args.content_abl, None)
        out["fallback_interaction"] = fallback_interaction(items, abl_items)
        print(json.dumps(out["fallback_interaction"], indent=1))
    elif prim["floor_pinned"]:
        print(f"\nWARNING: primary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} informative "
              "items) and no --content-abl was given for the fallback interaction.")

    # Stamp inputs' digests -- `results/gate_p_9b.json` was once quoted for a month
    # while the transcripts under it had been re-judged out from under it, and
    # nothing could tell (STATE.md). A derived artefact names its source or it
    # cannot be checked against it.
    out["source"] = {
        "content_base": {"path": str(args.content_base),
                          "sha256_16": hashlib.sha256(args.content_base.read_bytes()).hexdigest()[:16]},
    }
    if args.existing_base.exists():
        out["source"]["existing_base"] = {
            "path": str(args.existing_base),
            "sha256_16": hashlib.sha256(args.existing_base.read_bytes()).hexdigest()[:16],
        }
    if args.content_abl and args.content_abl.exists():
        out["source"]["content_abl"] = {
            "path": str(args.content_abl),
            "sha256_16": hashlib.sha256(args.content_abl.read_bytes()).hexdigest()[:16],
        }

    out_path = args.content_base.with_name(args.content_base.stem + "_analysis.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
