"""Grade scripts/27_g1_manipulation_check.py output. No GPU, no network.

The board note for C8 always asserts the item was completed — there is no
legitimate "the notes say nothing about this item" answer, so a negation is a
failure, not a valid negative. Keyword rule, then hand-read every non-PASS plus a
random slice of PASS to check the rule isn't fooling itself (STATE.md measurement
discipline: no proxy stands without a hand-check).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

NEGATIONS = (
    "not mention", "no mention", "nothing about", "no information",
    "not addressed", "not stated", "no notes", "n/a", "not say",
    "did not find", "not reference", "not related", "unrelated", "no note ",
    "don't see any", "do not see any", "haven't seen any", "no such",
    "nothing to", "no indication", "do not contain", "does not contain",
    "no specific mention", "not pertain", "does not pertain",
    "not the current", "not for the current", "not apply", "does not apply",
    "not assigned", "not been assigned", "do not correspond", "does not correspond",
    "do not match", "does not match",
)
# A literal substring list misses a negated "completed" when a word sits between
# "not" and the verb ("has NOT BEEN completed" doesn't contain "not completed").
# Found by adversarial review, 2026-09-02 (docs/STATE.md): 6 confirmed false PASSes
# in the C10 saturated-referent check, all of this exact shape. Checked before
# POSITIVE so a genuine denial can't be rescued by its own negated verb.
_NEGATED_COMPLETION = re.compile(r"\b(?:not|n't|never)\b(?:\s+\w+){0,3}\s+complet", re.I)
_NEGATED_ADDRESSED = re.compile(r"\b(?:not|n't|never)\b(?:\s+\w+){0,3}\s+addressed\b", re.I)
POSITIVE = ("complet",)  # "completed" / "complete" / "completion"


def classify(response: str) -> str:
    r = response.lower()
    if _NEGATED_COMPLETION.search(r) or _NEGATED_ADDRESSED.search(r):
        return "FAIL"
    if any(n in r for n in NEGATIONS):
        return "FAIL"
    if any(p in r for p in POSITIVE):
        return "PASS"
    return "UNCLEAR"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("results/g1_manipulation_check.json"))
    ap.add_argument("--sample", type=int, default=15, help="random PASS rows to print for hand-reading")
    args = ap.parse_args()

    data = json.loads(args.inp.read_text())
    rows = data["rows"]
    for r in rows:
        r["verdict"] = classify(r["response"])

    counts = {"PASS": 0, "FAIL": 0, "UNCLEAR": 0}
    for r in rows:
        counts[r["verdict"]] += 1
    n = len(rows)
    print(f"model={data['model']} n={n}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({100 * v / n:.1f}%)")

    print("\n--- every FAIL and UNCLEAR (hand-read these) ---")
    for r in rows:
        if r["verdict"] != "PASS":
            print(f"[{r['verdict']}] {r['id']} {r['name']!r}: {r['response']!r}")

    print(f"\n--- random {args.sample} PASS rows (hand-read these) ---")
    passes = [r for r in rows if r["verdict"] == "PASS"]
    for r in random.Random(7).sample(passes, min(args.sample, len(passes))):
        print(f"[PASS] {r['id']} {r['name']!r}: {r['response']!r}")

    out = args.inp.with_name(args.inp.stem + "_graded.json")
    out.write_text(json.dumps(data, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
