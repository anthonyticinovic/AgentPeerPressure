"""Gate P, properly: multi-turn tool loop, AgentHarm's own graders, full corpus.

Supersedes 10_peer_pilot.py, which measured a single turn and could not reach the
payload step for most items. Every number that script produced is retracted.

Fixes carried in by design, not by patch:
- board is randomised *within cluster*; the old `boards[i % 10]` aligned board parity
  to variant parity, making board a perfect proxy for `hint_included`
- rows are keyed on (cluster, id) and uniqueness is asserted; `id` alone collides
  across dataset splits
- every row carries its provenance (model, split, seed, max tokens, command line)
- an existing output file is only reused when its stamp matches; replies are never
  silently clobbered
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.boards import assert_byte_identical_task, assert_no_task_leakage, assert_speaker_free, load_boards, render  # noqa: E402
from pressure.config import CFG  # noqa: E402
from pressure.data import agentharm  # noqa: E402
from pressure.grading import grade  # noqa: E402
from pressure.loop import as_inspect_messages, run_loop  # noqa: E402
from pressure.model import load_model  # noqa: E402
from pressure.tools import SYSTEM_PROMPT, schemas_for  # noqa: E402

CONDITIONS = ("C0", "C1", "C1b", "C1b-para", "C2", "C3", "C6")


def stamp(args, model_name: str) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    return {
        "model": model_name, "conditions": list(CONDITIONS), "seed": args.seed,
        "max_turns": args.max_turns, "max_new_tokens": args.max_new_tokens,
        "n_items": args.n_items, "git": sha, "cmd": " ".join(sys.argv),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def corpus(n_items: int | None, one_per_grader: bool = False) -> list[dict]:
    items = agentharm(harmful=True)
    if one_per_grader:  # coverage smoke: exercise all 52 graders once
        first: dict[str, dict] = {}
        for it in sorted(items, key=lambda r: (r["cluster"], r["id"])):
            first.setdefault(it["grading_function"], it)
        items = list(first.values())
        n_items = None
    if n_items:  # first N clusters, all variants, every category represented
        by_cat = defaultdict(lambda: defaultdict(list))
        for it in items:
            by_cat[it["category"]][it["cluster"]].append(it)
        per = max(1, n_items // (4 * len(by_cat)))
        items = [x for c in sorted(by_cat) for cl in sorted(by_cat[c])[:per] for x in by_cat[c][cl]]
    # Shuffled by cluster so that a partially completed run is a representative
    # sample across categories rather than a prefix of one.
    order = sorted({i["cluster"] for i in items})
    random.Random(0).shuffle(order)
    rank = {c: n for n, c in enumerate(order)}
    return sorted(items, key=lambda r: (rank[r["cluster"]], r["id"]))


def build(items, boards, tok, seed: int) -> list[dict]:
    """Board is drawn per *cluster*, so it cannot track the variant index."""
    rng = random.Random(seed)
    per_cluster = {c: rng.choice(boards) for c in sorted({i["cluster"] for i in items})}
    rows, seen = [], set()
    for it in items:
        key = (it["cluster"], it["id"])
        if key in seen:
            raise AssertionError(f"duplicate item key {key}")
        seen.add(key)
        board = per_cluster[it["cluster"]]
        assert_no_task_leakage(board, it["prompt"])
        rendered = {c: render(c, it["prompt"], board, tok) for c in CONDITIONS}
        assert_byte_identical_task(rendered, it["prompt"])
        for cond, text in rendered.items():
            rows.append({
                "cluster": it["cluster"], "id": it["id"], "split": it["split"],
                "category": it["category"], "name": it["name"],
                "grading_function": it["grading_function"],
                "target_functions": it["target_functions"],
                "hint_included": it["hint_included"], "detailed_prompt": it["detailed_prompt"],
                "board": board.id, "condition": cond, "task": it["prompt"], "user_text": text,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--out", type=Path, default=CFG.results_dir / "peer_loop.json")
    ap.add_argument("--n-items", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--one-per-grader", action="store_true", help="one item per grader, all 52")
    ap.add_argument("--conditions", nargs="+", default=None, help="subset of conditions")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    global CONDITIONS
    if args.conditions:
        CONDITIONS = tuple(args.conditions)
    boards = load_boards()
    for b in boards:
        assert_speaker_free(b)
    model, tok = load_model(eval_model=not args.iter)
    model_name = CFG.iter_model if args.iter else CFG.eval_model
    meta = stamp(args, model_name)

    rows = None
    if args.out.exists():
        saved = json.loads(args.out.read_text())
        keys = ("model", "conditions", "seed", "n_items", "max_turns", "max_new_tokens")
        if all(saved["meta"].get(k) == meta[k] for k in keys):
            rows = saved["rows"]
            print(f"resuming {sum('turns' in r for r in rows)}/{len(rows)} done")
        else:
            raise SystemExit(f"{args.out} was produced with different settings; pass --out elsewhere")
    if rows is None:
        rows = build(corpus(args.n_items, args.one_per_grader), boards, tok, args.seed)
    print(f"{len(rows)} rows | {len(rows)//len(CONDITIONS)} items x {len(CONDITIONS)} conditions")

    todo = [r for r in rows if "turns" not in r][: args.limit]
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        turns = run_loop(
            model, tok,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["user_text"]}],
            schemas_for(r["target_functions"]),
            max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
        )
        try:
            sc = grade(r, as_inspect_messages(turns))
        except Exception as exc:  # one bad grader must not kill the run
            r["grade_error"] = f"{type(exc).__name__}: {exc}"
            sc = None
        r["turns"] = [{"text": t.text, "calls": t.calls, "results": t.results, "truncated": t.truncated} for t in turns]
        r["n_turns"] = len(turns)
        r["called"] = [c["function"] for t in turns for c in t.calls]
        r["cut_mid_call"] = any(t.cut_mid_call for t in turns)
        r["structural"] = sc.structural if sc else {}
        r["semantic"] = sc.semantic if sc else {}
        r["score"] = sc.score if sc else float("nan")
        r["full_compliance"] = sc.full_compliance if sc else False
        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.1f}s/item  eta {rate*(len(todo)-n)/60:.0f}m", flush=True)
            args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
