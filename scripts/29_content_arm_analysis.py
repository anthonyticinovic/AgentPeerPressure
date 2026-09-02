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

    uv run python scripts/29_content_arm_analysis.py \
        --content-base results/content_arm_base_judged.json \
        --existing-base results/gate_a_full_base_judged.json \
        [--content-abl results/content_arm_abl_judged.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.stats import cluster_sign_test, contrast, holm  # noqa: E402

CONTENT_CONDS = ("C8", "C8b", "C9", "C8n")
NEEDED_FOR_SECONDARY = ("C2", "C1b")
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
            if r["condition"] in NEEDED_FOR_SECONDARY:
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


def belief(items: dict) -> dict:
    have = {k: v for k, v in items.items() if "C8" in v and "C9" in v
            and "p_harm_orth" in v["C8"] and "p_harm_orth" in v["C9"]}
    if not have:
        return {"n": 0, "note": "no --monitor data joined (p_harm_orth absent)"}
    c8_p = np.array([v["C8"]["p_harm_orth"] for v in have.values()])
    c9_p = np.array([v["C9"]["p_harm_orth"] for v in have.values()])
    rows = [v["C8"] for v in have.values()]
    n = len(rows)
    comply_flagged = sum(1 for v in have.values() if comply(v["C8"]))
    return {
        "n": n,
        "mean_p_harm_orth_C8": float(c8_p.mean()),
        "mean_p_harm_orth_C9": float(c9_p.mean()),
        "mean_delta": float((c8_p - c9_p).mean()),
        "comply_C8_n": comply_flagged,
    }


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

    print("\n=== belief: p_harm_orth, C8 vs C9 ===")
    bel = belief(items)
    print(json.dumps(bel, indent=1))

    out = {"primary": prim, "secondary": sec, "belief": bel}

    if prim["floor_pinned"] and args.content_abl and args.content_abl.exists():
        print(f"\nprimary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} informative items) "
              "-- falling back to the cross-arm interaction, as the existing design does.")
        abl_items = load_joined(args.content_abl, None)
        out["fallback_interaction"] = fallback_interaction(items, abl_items)
        print(json.dumps(out["fallback_interaction"], indent=1))
    elif prim["floor_pinned"]:
        print(f"\nWARNING: primary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} informative "
              "items) and no --content-abl was given for the fallback interaction.")

    out_path = args.content_base.with_name(args.content_base.stem + "_analysis.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
