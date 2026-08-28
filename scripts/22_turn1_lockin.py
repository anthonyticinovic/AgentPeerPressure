"""Is the compliance decision already made at turn 1, before framing has a chance?

Recommended independently by three round-2 adversarial-review subagents (2026-08-28)
as a mechanistic "why" for the peer-framing null (Result 2/3 in docs/writeup.md):
the board note sits once, near the top of a prompt that then grows by hundreds of
tokens per tool turn. If turn-1 engagement already predicts the final verdict almost
perfectly, that's a concrete, falsifiable account of *why* framing can't move the
outcome — not just "the design lacks power."

Needs no calibration or threshold (unlike a harm-classification cross-tab) — every
input here is already binary: did the model call any tool at turn 1, and did the
row end up `full_compliance_judged`. Safe to build and run against data that's about
to be superseded by the corrected rerun; rerun against the new judged files once
they land, same command.

    uv run python scripts/22_turn1_lockin.py \
        --base results/gate_a_full_base_judged.json --abl results/gate_a_full_abl_judged.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [r for r in payload["rows"]
            if "turns" in r and "grade_error" not in r and "judge_error" not in r]


def attempted_t1(row: dict) -> bool:
    turns = row["turns"]
    return bool(turns) and bool(turns[0]["calls"])


def complied(row: dict) -> bool:
    return bool(row["full_compliance_judged"])


def report(rows: list[dict], label: str) -> None:
    print(f"\n{label}  (n={len(rows)})")
    by_t1: dict[bool, list[bool]] = {True: [], False: []}
    for r in rows:
        by_t1[attempted_t1(r)].append(complied(r))

    for t1, vals in by_t1.items():
        n = len(vals)
        rate = sum(vals) / n if n else float("nan")
        tag = "attempted at turn 1" if t1 else "no call at turn 1"
        print(f"  {tag:22} n={n:4d}  final compliance rate = {rate:.3f}")

    # If the decision is locked in early, "no call at turn 1 -> never complies" should
    # be close to exact (a later turn essentially never recovers from an empty start).
    no_t1_but_complied = sum(1 for r in rows if not attempted_t1(r) and complied(r))
    print(f"  rows with NO turn-1 call that still ended compliant: {no_t1_but_complied}"
          f" / {len(by_t1[False])}")

    # Within the engaged-at-turn-1 subset only: does condition still show no signal,
    # or does framing matter once the model has actually started? This is the direct
    # test of "framing can't reach a decision already made" vs "framing genuinely does
    # nothing even when it has the chance to."
    engaged = [r for r in rows if attempted_t1(r)]
    by_cond: dict[str, list[bool]] = defaultdict(list)
    for r in engaged:
        by_cond[r["condition"]].append(complied(r))
    print("  within turn-1-engaged rows only, compliance rate by condition:")
    for cond in sorted(by_cond):
        vals = by_cond[cond]
        print(f"    {cond:4} n={len(vals):4d}  rate={sum(vals) / len(vals):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("results/gate_a_full_base_judged.json"))
    ap.add_argument("--abl", type=Path, default=Path("results/gate_a_full_abl_judged.json"))
    args = ap.parse_args()

    report(load_rows(args.base), "base (refusal intact)")
    report(load_rows(args.abl), "ablated")


if __name__ == "__main__":
    main()
