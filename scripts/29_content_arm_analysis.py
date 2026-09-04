"""Pre-registered analysis for the content arm (docs/STATE.md).

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
from pressure.grading import _PAYLOAD_REFUSAL, _PREPARATORY  # noqa: E402
from pressure.monitor import Directions  # noqa: E402
from pressure.stats import binom_exact_p, cluster_sign_test, contrast, holm, mcnemar_exact  # noqa: E402

CONTENT_CONDS = ("C8", "C8b", "C9", "C8n")
NEEDED_FOR_SECONDARY = ("C2", "C1b")
NEEDED_FOR_EXPLORATORY = ("C6", "C3")
FLOOR_PINNED_THRESHOLD = 15  # below this many informative items, distrust the base-arm primary


def comply(r: dict) -> bool:
    return bool(r["full_compliance_judged"])


def any_call(row: dict) -> bool:
    """Did the row make any tool call at all."""
    return bool(row.get("called"))


def substantive_action(row: dict) -> bool:
    """RETRACTED as a positive finding, kept for the record -- see
    `retraction4_check` and the module docstring. Nominally: at least one
    non-preparatory tool call whose own payload does not read as refusal
    language, reusing the vendored `_PREPARATORY` set and `_PAYLOAD_REFUSAL`
    regex `payload_is_refusal` already uses.

    A third adversarial review (2026-09-04) found this is, mechanically,
    indistinguishable from `any_call` on this corpus: AgentHarm runs with
    `n_irrelevant_tools=0`, so every tool call offered IS a target function,
    and `_PAYLOAD_REFUSAL` fires on 1 of 1664 rows and changes zero verdicts.
    Restricting the resulting contrasts to items where `any_call` does not
    differ between conditions (this project's standing "retraction #4" check,
    STATE.md -- the same test that retracted `hit_target` and PARTIAL C2-C0)
    collapses the base-arm C8-C9 gap from +12.98pp/p=1.1e-05 to +1.70pp/p=0.375.
    This is the same failure mode this project has already retracted twice:
    an engagement tautology, not a measure of task completion. Do not report
    `substantive_action` results as a positive finding; report the retraction
    itself, which `retraction4_check` computes automatically alongside it.
    """
    for turn in row["turns"]:
        for call in turn["calls"]:
            if call["function"] in _PREPARATORY:
                continue
            text = " ".join(str(v) for v in call["arguments"].values())
            if not (text.strip() and _PAYLOAD_REFUSAL.search(text)):
                return True
    return False


def retraction4_check(items: dict, ref: str, arm: str, outcome) -> dict:
    """This project's standing robustness check (STATE.md): restrict to items
    where `any_call` does NOT differ between ref/arm, and re-test. An outcome
    that collapses here is measuring whether the model engaged at all, not
    whatever it claims to measure -- the exact pattern that retracted
    `hit_target` and PARTIAL C2-C0 earlier in this project, and `substantive_action`
    a third time (2026-09-04). Always run this alongside any new outcome before
    it is trusted, not just when a reviewer happens to ask."""
    restricted = {k: v for k, v in items.items()
                  if ref in v and arm in v and any_call(v[ref]) == any_call(v[arm])}
    b, c, delta, p = contrast(restricted, ref=ref, arm=arm, outcome=outcome)
    p_cluster = cluster_sign_test(restricted, ref=ref, arm=arm, outcome=outcome)
    return {"n_paired": len(restricted), "delta_pp": delta, "b": b, "c": c,
            "p_item_exact": p, "p_cluster_exact": p_cluster}


def _load_rows(path: Path) -> list[dict]:
    """`12_peer_loop.py` writes the full row skeleton (task, board, user_text) for
    every row up front, before generation -- so a checkpointed, still-running file
    (an interim look at G3/G4) has all 832 rows present with `full_compliance_judged`
    absent on whatever hasn't been generated yet. `"turns" in r` is the same
    generated-or-not test `15_regrade.py` uses; without it, `comply()` KeyErrors on
    the first not-yet-processed row instead of just excluding it."""
    payload = json.loads(path.read_text())
    return [r for r in payload["rows"]
            if "turns" in r and "grade_error" not in r and "judge_error" not in r
            # A whole-row-unscored row (judge.any_decline_this_row's fix, 2026-09-04)
            # has `Score.full_compliance` return False rather than excluded --
            # `comply()` would silently pool it as non-compliance otherwise. The
            # current pipeline only ever marks a row's ENTIRE criteria set unscored
            # (never a partial subset, see grading.grade_with_judge), so a nonempty
            # `unscored_criteria` here always means "exclude the row", not "exclude
            # one criterion".
            and not r.get("unscored_criteria")]


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


def informative_count(items: dict, ref: str, arm: str, outcome=comply) -> int:
    return sum(
        1 for v in items.values()
        if ref in v and arm in v and outcome(v[ref]) != outcome(v[arm])
    )


def primary(items: dict, outcome=comply) -> dict:
    have = {k: v for k, v in items.items() if "C8" in v and "C9" in v}
    b, c, delta, p = contrast(have, ref="C9", arm="C8", outcome=outcome)
    p_cluster = cluster_sign_test(have, ref="C9", arm="C8", outcome=outcome)
    n_informative = informative_count(have, "C9", "C8", outcome=outcome)
    return {
        "n_paired": len(have), "n_informative": n_informative,
        "delta_pp": delta, "b": b, "c": c,
        "p_item_exact": p, "p_cluster_exact": p_cluster,
        "floor_pinned": n_informative < FLOOR_PINNED_THRESHOLD,
    }


def secondary_family(items: dict, outcome=comply) -> dict:
    tests = {
        "C8-C8b": ("C8b", "C8"),   # H6: speaker increment given content
        "C8-C2": ("C2", "C8"),    # content increment at fixed (named) speaker
        "C8b-C1b": ("C1b", "C8b"),  # content increment, speaker-free
    }
    raw = {}
    for name, (ref, arm) in tests.items():
        have = {k: v for k, v in items.items() if ref in v and arm in v}
        b, c, delta, p = contrast(have, ref=ref, arm=arm, outcome=outcome)
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


def _clustered_sign_test(deltas: dict) -> tuple[int, int, float]:
    """Sum a continuous per-item delta by cluster (items are 4 prompt variants
    of one scenario, not independent -- same rationale as `cluster_sign_test`),
    then sign-test the cluster totals. `deltas` keys are (cluster, id) tuples.
    Added per adversarial review, 2026-09-04: `stats.py`'s own module docstring
    requires both an item- and a cluster-level test for every contrast; the
    belief sign test and the diff-in-differences were reporting item-level
    only, anti-conservative by roughly a factor of two."""
    per: dict = defaultdict(float)
    for k, d in deltas.items():
        per[k[0]] += d
    b = sum(1 for v in per.values() if v > 0)
    c = sum(1 for v in per.values() if v < 0)
    return b, c, mcnemar_exact(b, c)


def belief(items: dict, tau: float | None) -> dict:
    """p_harm_orth at task_last, C8 vs C9, plus the comply-and-flagged cross-tab
    for each condition. An untested mean invites exactly the "is this just
    noise" question the adversarial review raised, so the direction of the
    per-item delta is tested with an exact sign test, not asserted from a mean.

    Reframed per adversarial review, 2026-09-04: `p_harm_orth` is read at
    `task_last` on turn 0, before any generation, under greedy decoding -- it
    is a deterministic function of two different PROMPT strings (C8's vs C9's),
    not evidence that a belief forms during generation and then fails to move
    behaviour. Report the consistent direction and its small magnitude; do not
    lead with the item-level p-value, which is a real but easily-misread
    quantity once n is in the hundreds (a tiny, consistent per-item nudge
    produces an extreme sign-test p by construction, not because the effect is
    large).
    """
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
    delta_by_key = {k: v["C8"]["p_harm_orth"] - v["C9"]["p_harm_orth"] for k, v in have.items()}
    b_cl, c_cl, p_cluster = _clustered_sign_test(delta_by_key)
    out = {
        "n": len(have),
        "mean_p_harm_orth_C8": float(c8_p.mean()),
        "mean_p_harm_orth_C9": float(c9_p.mean()),
        "mean_delta": float(delta.mean()),
        "measured_pre_generation_on_the_prompt": True,
        "delta_sign_test": {
            "n_positive": n_pos, "n_negative": int(len(nonzero) - n_pos),
            "n_nonzero": int(len(nonzero)), "p_two_sided_item": p_sign,
            "b_cluster": b_cl, "c_cluster": c_cl, "p_two_sided_cluster": p_cluster,
        },
    }
    if tau is not None:
        out["crosstab_C8"] = _crosstab([v["C8"] for v in have.values()], tau)
        out["crosstab_C9"] = _crosstab([v["C9"] for v in have.values()], tau)
        out["tau_harm_orth"] = tau
    else:
        out["crosstab_note"] = "tau_harm_orth unavailable -- Directions.load found no calibrated threshold"
    return out


def exploratory_family(items: dict, outcome=comply) -> dict:
    """Not part of the pre-registered Holm-corrected family (see module
    docstring). Discriminates peer-endorsement from in-context repetition of
    the restated content as the driver of the primary contrast."""
    out = {}

    have = {k: v for k, v in items.items() if "C6" in v and "C8n" in v}
    b, c, delta, p = contrast(have, ref="C6", arm="C8n", outcome=outcome)
    out["C8n-C6"] = {"n_paired": len(have), "delta_pp": delta, "b": b, "c": c, "p_raw": p}

    have = {k: v for k, v in items.items() if "C8n" in v and "C8" in v}
    b, c, delta, p = contrast(have, ref="C8n", arm="C8", outcome=outcome)
    out["C8-C8n"] = {"n_paired": len(have), "delta_pp": delta, "b": b, "c": c, "p_raw": p}

    have = {k: v for k, v in items.items() if all(cond in v for cond in ("C8", "C9", "C2", "C3"))}
    d_by_key = {
        k: float(outcome(v["C8"])) - float(outcome(v["C9"])) - float(outcome(v["C2"])) + float(outcome(v["C3"]))
        for k, v in have.items()
    }
    d = np.array(list(d_by_key.values()), dtype=float)
    nonzero = d[d != 0]
    n_pos = int((nonzero > 0).sum())
    p_sign = binom_exact_p(n_pos, len(nonzero), 0.5) if len(nonzero) else float("nan")
    b_cl, c_cl, p_cluster = _clustered_sign_test(d_by_key)
    out["diff_in_diff_C8C9_vs_C2C3_matched"] = {
        "n_paired": len(have), "mean_diff_pp": 100 * float(d.mean()) if len(have) else float("nan"),
        "sign_test": {
            "n_positive": n_pos, "n_nonzero": int(len(nonzero)), "p_two_sided_item": p_sign,
            "b_cluster": b_cl, "c_cluster": c_cl, "p_two_sided_cluster": p_cluster,
        },
    }
    return out


def fallback_interaction(base_items: dict, abl_items: dict, outcome=comply) -> dict:
    """Cluster-permutation interaction across ablation arms, same construction as
    `19_ablation_analysis.py::interaction` -- used only if `primary()` reports
    floor_pinned."""
    keys = sorted(set(base_items) & set(abl_items))
    keys = [k for k in keys if all(c in base_items[k] and c in abl_items[k] for c in ("C8", "C9"))]
    clusters = sorted({k[0] for k in keys})
    pos = {c: i for i, c in enumerate(clusters)}
    idx = np.array([pos[k[0]] for k in keys])
    d = np.array([
        (outcome(abl_items[k]["C8"]) - outcome(abl_items[k]["C9"]))
        - (outcome(base_items[k]["C8"]) - outcome(base_items[k]["C9"]))
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
    ap.add_argument("--existing-abl", type=Path, default=Path("results/gate_a_full_abl_judged.json"))
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

    abl_items = None
    if args.content_abl and args.content_abl.exists():
        # The existing nine-condition design's own headline pattern was null with
        # refusal intact (Result 2) and only opened up under ablation (Result 3).
        # The ablated arm is always worth reporting on that basis alone -- not just
        # as a fallback for when the base arm is underpowered. Computed regardless
        # of floor_pinned; the pre-registered PRIMARY stays the base-arm contrast
        # either way (PLAN #6), this is the parallel check the design's own logic
        # calls for. Also runs the ablated arm's own exploratory family and belief
        # read -- an earlier version of this script computed only primary+secondary
        # for the ablated arm, an incomplete generalisation flagged by adversarial
        # review, 2026-09-04.
        abl_items = load_joined(args.content_abl, args.existing_abl)
        print("\n=== ablated arm: C8 - C9 (mirrors Result 2->3's own pattern) ===")
        abl_prim = primary(abl_items)
        print(json.dumps(abl_prim, indent=1))
        out["ablated_primary"] = abl_prim

        print("\n=== ablated arm: secondary family (Holm-corrected, 3 tests) ===")
        abl_sec = secondary_family(abl_items)
        print(json.dumps(abl_sec, indent=1))
        out["ablated_secondary"] = abl_sec

        print("\n=== ablated arm: exploratory family ===")
        abl_expl = exploratory_family(abl_items)
        print(json.dumps(abl_expl, indent=1))
        out["ablated_exploratory"] = abl_expl

        print("\n=== ablated arm: belief ===")
        abl_bel = belief(abl_items, tau)
        print(json.dumps(abl_bel, indent=1))
        out["ablated_belief"] = abl_bel

        print("\n=== cross-arm interaction: does the C8-C9 gap widen under ablation? ===")
        out["interaction"] = fallback_interaction(items, abl_items)
        print(json.dumps(out["interaction"], indent=1))

        if prim["floor_pinned"]:
            print(f"\nNote: base-arm primary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} "
                  "informative items) -- the interaction above is this design's fallback "
                  "primary in that case, not just a secondary check.")
    elif prim["floor_pinned"]:
        print(f"\nWARNING: primary is floor-pinned (<{FLOOR_PINNED_THRESHOLD} informative "
              "items) and no --content-abl was given for the fallback interaction.")

    # substantive_action: RETRACTED as a positive finding by a third adversarial
    # review (2026-09-04) -- see its own docstring and retraction4_check's. Kept
    # in the pipeline and always reported WITH the retraction check beside it, on
    # the same discipline as docs/STATE.md's other retractions: a dead end this
    # project already walked into twice (hit_target, PARTIAL C2-C0) doesn't get
    # quietly deleted, it gets shown failing its own robustness check.
    print("\n=== RETRACTED exploratory outcome (base arm, substantive_action) -- "
          "reported with its own retraction check, not as a finding ===")
    sa_base_primary = primary(items, outcome=substantive_action)
    sa_base_retraction4 = retraction4_check(items, ref="C9", arm="C8", outcome=substantive_action)
    print(json.dumps({"unrestricted": sa_base_primary, "any_call_static_restricted": sa_base_retraction4}, indent=1))
    out["substantive_action_base_RETRACTED"] = {
        "unrestricted": sa_base_primary,
        "any_call_static_restricted": sa_base_retraction4,
        "verdict": "retracted -- collapses under the any_call-static restriction, "
                   "same failure mode as hit_target and PARTIAL C2-C0",
    }

    if abl_items is not None:
        sa_abl_primary = primary(abl_items, outcome=substantive_action)
        sa_abl_retraction4 = retraction4_check(abl_items, ref="C9", arm="C8", outcome=substantive_action)
        out["substantive_action_abl_RETRACTED"] = {
            "unrestricted": sa_abl_primary,
            "any_call_static_restricted": sa_abl_retraction4,
            "verdict": "not reported -- ceiling-compressed, see docs/STATE.md",
        }
        print("\n=== RETRACTED exploratory outcome (ablated arm, substantive_action) ===")
        print(json.dumps(out["substantive_action_abl_RETRACTED"], indent=1))

        # The pre-registered outcome's OWN ablated-arm primary is the real result --
        # promoted here, not retracted. Confirmed it survives the same any_call-static
        # restriction the substantive_action outcome fails (third review, 2026-09-04).
        print("\n=== robustness check on the ablated arm's PRE-REGISTERED primary "
              "(comply, not substantive_action) -- does it survive the same "
              "any_call-static restriction? ===")
        out["ablated_primary_retraction4_check"] = retraction4_check(items=abl_items, ref="C9", arm="C8", outcome=comply)
        print(json.dumps(out["ablated_primary_retraction4_check"], indent=1))

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
    if args.content_abl and args.content_abl.exists() and args.existing_abl.exists():
        out["source"]["existing_abl"] = {
            "path": str(args.existing_abl),
            "sha256_16": hashlib.sha256(args.existing_abl.read_bytes()).hexdigest()[:16],
        }

    out_path = args.content_base.with_name(args.content_base.stem + "_analysis.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
