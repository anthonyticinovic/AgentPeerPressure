"""Gate 3: does the metric manufacture contrasts?

Twice now a condition effect has come from a criterion that has nothing to do with
compliance - first `hit_target` (which was "emitted any tool call"), then
`correct_emails_num` (an email *count*). This script decomposes any condition
difference down to the individual criterion responsible, so an artefact of that
shape is visible before it is reported as a finding.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Criteria that count or route rather than verify what the model did. A contrast
# driven by one of these is an artefact, not an effect.
SUSPECT = ("_num", "_count", "correct_order", "all_target_functions_called")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=Path, required=True)
    ap.add_argument("--ref", default="C0")
    args = ap.parse_args()

    rows = [r for r in json.loads(args.path.read_text())["rows"] if "turns" in r]
    # Read whichever metric the analysis reports. Auditing `structural` on a judged
    # file inspects criteria the report does not use - and the judge's own criteria,
    # the ones most worth auditing, would be invisible.
    judged = any("full_compliance_judged" in r for r in rows)
    crit_key = "structural_judged" if judged else "structural"
    fc_key = "full_compliance_judged" if judged else "full_compliance"
    print(f"metric: {fc_key}")
    if judged:
        primary = None  # every grader verifies its own content once the judge is live
    else:
        split = json.loads((Path("results") / "grader_content_split.json").read_text())
        primary = set(split["primary"])

    items = defaultdict(dict)
    for r in rows:
        items[(r["cluster"], r["id"])][r["condition"]] = r
    conds = sorted({r["condition"] for r in rows})
    paired = {k: v for k, v in items.items() if len(v) == len(conds)}
    print(f"{len(rows)} rows, {len(paired)} paired items, conditions {conds}")
    print(f"graders exercised: {len({r['grading_function'] for r in rows})} / 52")
    print(f"grade errors: {sum('grade_error' in r for r in rows)}  "
          f"tool errors: {sum(any(x.startswith('Error') for t in r['turns'] for x in t['results']) for r in rows)}")

    scopes = [("all graders", None)] if judged else [("PRIMARY (content-verified graders)", primary), ("all graders", None)]
    for label, keys in scopes:
        sub = {k: v for k, v in paired.items()
               if keys is None or v[args.ref]["grading_function"] in keys}
        if not sub:
            continue
        print(f"\n=== {label}: {len(sub)} items ===")
        for c in conds:
            fc = sum(v[c][fc_key] for v in sub.values()) / len(sub)
            print(f"  {c:6s} full_compliance {fc:.3f}")
        # which criterion differs, and how often
        diffs: dict[str, int] = defaultdict(int)
        for v in sub.values():
            for c in conds:
                if c == args.ref:
                    continue
                a, b = v[args.ref][crit_key], v[c][crit_key]
                for k in set(a) | set(b):
                    if a.get(k) != b.get(k):
                        diffs[k] += 1
        if not diffs:
            print("  no criterion differs across conditions on any item")
            continue
        print(f"  {'criterion':34s} {'item-condition disagreements':>28s}")
        for k, n in sorted(diffs.items(), key=lambda kv: -kv[1]):
            flag = "  <-- SUSPECT: counting/routing" if any(s in k for s in SUSPECT) else ""
            print(f"  {k:34s} {n:28d}{flag}")


if __name__ == "__main__":
    main()
