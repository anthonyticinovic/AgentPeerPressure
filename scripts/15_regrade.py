"""Re-grade stored transcripts with AgentHarm's full rubric, judge included.

Grading is separable from generation: transcripts are on disk, so re-scoring costs
API calls and seconds rather than another model run. That is what makes enabling the
judge cheap now rather than a decision that needed making before the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.grading import grade_with_judge  # noqa: E402
from pressure.judge import MODEL, stats  # noqa: E402
from pressure.loop import Turn, as_inspect_messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.loads(args.path.read_text())
    rows = [r for r in data["rows"] if "turns" in r]
    todo = rows[: args.limit] if args.limit else rows
    print(f"re-grading {len(todo)} rows with {MODEL}")

    changed, t0 = 0, time.time()
    for n, r in enumerate(todo, 1):
        turns = [Turn(text=t["text"], calls=t["calls"], results=t["results"],
                      truncated=t.get("truncated", False)) for t in r["turns"]]
        try:
            sc = grade_with_judge(r, as_inspect_messages(turns))
        except Exception as exc:
            r["judge_error"] = f"{type(exc).__name__}: {exc}"
            continue
        was = r.get("full_compliance")
        r["structural_judged"] = sc.structural
        r["score_judged"] = sc.score
        r["full_compliance_judged"] = sc.full_compliance
        # A declined judgement is not a failure. Record it per row so the analysis
        # can report it separately instead of pooling it as non-compliance.
        r["unscored_criteria"] = sc.unscored
        changed += was != sc.full_compliance
        if n % 25 == 0 or n == len(todo):
            s = stats()
            print(f"  {n}/{len(todo)} {(time.time()-t0)/n:.2f}s/row | calls {s['calls']} "
                  f"cached {s['cached']} invalid {s['invalid']} tok {s['in_tokens']}/{s['out_tokens']}", flush=True)
            args.out.write_text(json.dumps({"meta": {**data["meta"], "judge": MODEL}, "rows": data["rows"]}, indent=1))
    args.out.write_text(json.dumps({"meta": {**data["meta"], "judge": MODEL}, "rows": data["rows"]}, indent=1))
    s = stats()
    cost = s["in_tokens"] / 1e6 * 0.435 + s["out_tokens"] / 1e6 * 0.87
    print(f"\nverdicts changed vs structural-only: {changed}/{len(todo)}")
    print(f"judge calls {s['calls']} (cached {s['cached']}, invalid {s['invalid']}, "
          f"recovered by fallback {s.get('recovered', 0)}, UNSCORED {s['unscored']}) "
          f"tokens {s['in_tokens']} in / {s['out_tokens']} out  ~= ${cost:.3f} at V4-Pro list")
    print(f"models used: {s['by_model']}")
    if s["unscored"]:
        print("  UNSCORED verdicts are NOT failures — report them separately, never pooled.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
