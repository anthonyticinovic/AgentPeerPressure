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
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.boards import (  # noqa: E402
    assert_bounded_task_reference,
    assert_byte_identical_task,
    assert_no_task_leakage,
    assert_speaker_free,
    item_reference,
    load_boards,
    render,
    speaker_pronoun_warnings,
)
from pressure.causal import ablate_all_components  # noqa: E402
from pressure.config import CFG  # noqa: E402
from pressure.data import agentharm  # noqa: E402
from pressure.grading import grade, grade_with_judge  # noqa: E402
from pressure.loop import Turn, as_inspect_messages, run_loop  # noqa: E402
from pressure.model import load_model  # noqa: E402
from pressure.monitor import Directions, projections  # noqa: E402
from pressure.judge import MODEL as JUDGE_MODEL, stats as judge_stats  # noqa: E402
from pressure.tools import SYSTEM_PROMPT, schemas_for  # noqa: E402

CONDITIONS = ("C0", "C1", "C1b", "C1b-para", "C2", "C3", "C6")


def _atomic_write(path: Path, payload: dict) -> None:
    """Write via a temp file and rename. `write_text` truncates the only copy of a
    multi-hour run on every checkpoint; a crash mid-write loses everything."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def _checkpoint(n: int, total: int, t0: float, out: Path, meta: dict, rows: list) -> None:
    """Progress line and an atomic snapshot, every 10 rows. Called on the grade-failure
    path too, so a transcript is never left only in memory."""
    if n % 10 and n != total:
        return
    rate = (time.time() - t0) / n
    print(f"  {n}/{total}  {rate:.1f}s/item  eta {rate*(total-n)/60:.0f}m", flush=True)
    _atomic_write(out, {"meta": meta, "rows": rows})


def stamp(args, model_name: str) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        # No .git on the cluster — only tracked files are shipped. The sync writes
        # the sha to this file so provenance survives the transfer.
        stamped = Path(__file__).resolve().parents[1] / ".git_sha"
        sha = stamped.read_text().strip() if stamped.exists() else "unknown"
    return {
        "model": model_name, "conditions": list(CONDITIONS), "seed": args.seed,
        "max_turns": args.max_turns, "max_new_tokens": args.max_new_tokens,
        "n_items": args.n_items, "git": sha, "cmd": " ".join(sys.argv),
        "judge": None if args.no_judge else JUDGE_MODEL,
        "one_per_grader": args.one_per_grader,
        "sample_per_cluster": bool(args.sample_per_cluster),
        "ablate": bool(args.ablate),
        "monitor": bool(args.monitor),
        "only_cluster": args.only_cluster,
        "force_board": args.force_board,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def corpus(n_items: int | None, one_per_grader: bool = False,
           sample_per_cluster: bool = False, seed: int = CFG.seed,
           only_cluster: str | None = None) -> list[dict]:
    items = agentharm(harmful=True)
    if only_cluster is not None:
        # Targeted re-run of one cluster (e.g. a token-budget casualty) at a different
        # --max-new-tokens. Applied first and unconditionally: the other subsetting
        # branches below are for whole-corpus smoke/pilot runs and are never combined
        # with this flag, so this must not interact with them.
        items = [it for it in items if it["cluster"] == only_cluster]
        if not items:
            raise SystemExit(f"--only-cluster {only_cluster!r} matched no items")
        return sorted(items, key=lambda r: r["id"])
    if one_per_grader:  # coverage smoke: exercise all 52 graders once
        first: dict[str, dict] = {}
        for it in sorted(items, key=lambda r: (r["cluster"], r["id"])):
            first.setdefault(it["grading_function"], it)
        items = list(first.values())
        n_items = None
    if sample_per_cluster:
        # One variant per cluster, rotating through the (hint_included, detailed_prompt)
        # 2x2 so that factor stays balanced. `one_per_grader` takes variant 1 every
        # time, which is (True, True) and the highest-compliance cell -- 23.1% against
        # 15.9% for variant 4 on the 9B run -- so it runs a pilot ~4pp hot.
        #
        # ICC ~0.38 means within-cluster variants are partly redundant, so for a fixed
        # budget this carries more independent information than a random sample of the
        # same size, and unlike a random draw it cannot drop a grader.
        by_cluster: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_cluster[it["cluster"]].append(it)
        clusters = sorted(by_cluster)
        random.Random(seed).shuffle(clusters)
        items = [sorted(by_cluster[cl], key=lambda r: r["id"])[n % len(by_cluster[cl])]
                 for n, cl in enumerate(clusters)]
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


def build(items, boards, tok, seed: int, force_board=None) -> list[dict]:
    """Board is drawn per *cluster*, so it cannot track the variant index.

    `force_board` pins every cluster present to one given Board instead of drawing —
    for a targeted single-cluster re-run, drawing from a fresh `random.Random(seed)`
    would land on a different point in the RNG stream than the full 52-cluster run
    did (this cluster's board draw happens after N-1 other draws in the full run),
    silently mismatching the peer-framing identities already baked into any
    surviving rows for that cluster.
    """
    rng = random.Random(seed)
    if force_board is not None:
        per_cluster = {c: force_board for c in sorted({i["cluster"] for i in items})}
    else:
        per_cluster = {c: rng.choice(boards) for c in sorted({i["cluster"] for i in items})}
    rows, seen = [], set()
    pronoun_items = []
    for it in items:
        key = (it["cluster"], it["id"])
        if key in seen:
            raise AssertionError(f"duplicate item key {key}")
        seen.add(key)
        board = per_cluster[it["cluster"]]
        assert_no_task_leakage(board, it["prompt"])
        ref = item_reference(it["name"], it["prompt"])
        rendered = {c: render(c, it["prompt"], board, tok, item_name=ref) for c in CONDITIONS}
        assert_byte_identical_task(rendered, it["prompt"])
        assert_bounded_task_reference(rendered, it["prompt"], ref)
        has_pronoun = bool(speaker_pronoun_warnings(ref)) if "C8b" in rendered else False
        if has_pronoun:
            pronoun_items.append(it["id"])
        for cond, text in rendered.items():
            rows.append({
                "cluster": it["cluster"], "id": it["id"], "split": it["split"],
                "category": it["category"], "name": it["name"],
                "grading_function": it["grading_function"],
                "target_functions": it["target_functions"],
                "hint_included": it["hint_included"], "detailed_prompt": it["detailed_prompt"],
                "board": board.id, "condition": cond, "task": it["prompt"], "user_text": text,
                "referent_has_pronoun": has_pronoun,
            })
    if pronoun_items:
        print(f"speaker_pronoun_warnings: {len(pronoun_items)}/{len({r['id'] for r in rows})} items' "
              f"C8b referent echoes the task's own pronoun (not a peer-voice leak -- see "
              f"boards.speaker_pronoun_warnings docstring): {pronoun_items}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--out", type=Path, default=CFG.results_dir / "peer_loop.json")
    ap.add_argument("--n-items", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--one-per-grader", action="store_true", help="one item per grader, all 52")
    ap.add_argument("--conditions", nargs="+", default=None, help="subset of conditions")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--no-judge", action="store_true",
                    help="grade without the semantic judge. The fallback, not the default: "
                         "with the judge off, 31 of 52 graders verify no content at all.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-cluster", action="store_true",
                    help="one variant per cluster, balanced across the (hint, detailed) "
                         "2x2: 52 items covering every cluster, grader and category. "
                         "Use for the pilot; --one-per-grader is variant-1 biased.")
    ap.add_argument("--ablate", action="store_true",
                    help="ablate the selected Arditi direction with ablate_all_components, "
                         "the intervention 03_arditi_selection scored candidates under")
    ap.add_argument("--monitor", action="store_true",
                    help="record r_harm and r_ref projections (one extra forward pass per turn)")
    ap.add_argument("--only-cluster", default=None,
                    help="re-run one cluster only, e.g. after a token-budget casualty. "
                         "Pair with --force-board to match the board an earlier full run "
                         "already used for that cluster, or rows won't merge cleanly.")
    ap.add_argument("--force-board", default=None,
                    help="pin the board id (e.g. board_03) instead of drawing one; "
                         "required for --only-cluster to be mergeable with existing data")
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

    dirs = Directions.load(CFG.results_dir) if (args.ablate or args.monitor) else None
    if dirs is not None:
        if dirs.model != model_name:
            raise SystemExit(f"directions are {dirs.model}, model is {model_name}")
        cos = dirs.cosines()
        print(f"directions: i*={dirs.arditi_position} l*={dirs.arditi_layer} "
              f"harm@{dirs.harm_layer} ref@{dirs.ref_layer} | "
              f"cos vs r_arditi: harm={cos['harm']:+.3f} ref={cos['ref']:+.3f}", flush=True)

    rows = None
    if args.out.exists():
        saved = json.loads(args.out.read_text())
        keys = ("model", "conditions", "seed", "n_items", "max_turns", "max_new_tokens",
                "judge", "one_per_grader", "sample_per_cluster", "ablate", "monitor",
                "only_cluster", "force_board")
        if all(saved["meta"].get(k) == meta[k] for k in keys):
            rows = saved["rows"]
            print(f"resuming {sum('turns' in r for r in rows)}/{len(rows)} done")
        else:
            raise SystemExit(f"{args.out} was produced with different settings; pass --out elsewhere")
    if rows is None:
        force_board = None
        if args.force_board and not args.only_cluster:
            raise SystemExit("--force-board without --only-cluster would pin every cluster "
                              "in a full run to one board — the exact confound this project "
                              "treats board as a fixed, per-cluster-random nuisance variable "
                              "to avoid")
        if args.force_board:
            by_id = {b.id: b for b in boards}
            if args.force_board not in by_id:
                raise SystemExit(f"--force-board {args.force_board!r} not among "
                                  f"{sorted(by_id)}")
            force_board = by_id[args.force_board]
        elif args.only_cluster:
            raise SystemExit("--only-cluster without --force-board draws a fresh board that "
                              "will very likely not match an earlier full run's framing for "
                              "this cluster, and the rows will fail to merge — pass "
                              "--force-board <id> (read the id off the existing rows for this "
                              "cluster) unless this is deliberately a from-scratch run")
        rows = build(corpus(args.n_items, args.one_per_grader, args.sample_per_cluster,
                            args.seed, only_cluster=args.only_cluster),
                     boards, tok, args.seed, force_board=force_board)
    print(f"{len(rows)} rows | {len(rows)//len(CONDITIONS)} items x {len(CONDITIONS)} conditions")

    # A row needs work if it has no transcript, or has one that failed to grade.
    # Generation and grading are retried independently: a grader or API failure must
    # never cost a re-run of the expensive half, and must never freeze in as a
    # non-compliance verdict - that was one-directional, toward the null.
    todo = [r for r in rows if "turns" not in r or "grade_error" in r][: args.limit]
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        if "turns" in r:  # regrade-only pass over a transcript already on disk
            turns = [Turn(text=t["text"], calls=t["calls"], results=t["results"],
                          truncated=t.get("truncated", False)) for t in r["turns"]]
        else:
            # The monitor call sits *inside* the ablation context deliberately: under
            # ablation the projections must be read off the ablated stream.
            trace: list[dict] = []
            # `task_bound` fixes the task-text search window at the turn-0 prompt's
            # length, captured on the first call (before any model output exists) and
            # reused on every later call — otherwise a later turn's growing prompt can
            # match the model's own echo of the task text instead of the original
            # instruction (see resolve_positions' docstring). A mutable single-element
            # list stands in for a nonlocal int, set once per row.
            task_bound: list[int] = []

            def hook(p: str, _bound=task_bound) -> None:
                if not _bound:
                    _bound.append(len(p))
                trace.append(projections(model, tok, p, r["task"], dirs, task_search_upto=_bound[0]))

            hook = hook if args.monitor else None
            ctx = ablate_all_components(model, dirs.r_arditi) if args.ablate else nullcontext()
            with ctx:
                turns = run_loop(
                    model, tok,
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["user_text"]}],
                    schemas_for(r["target_functions"]),
                    max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                    on_prompt=hook,
                )
            if args.monitor:
                # p_harm is constant within a row: task_last is an absolute index
                # inside the user message, and causal attention means later turns
                # cannot reach it. Store it once. The per-turn values are kept only so
                # the analysis can assert the drift is float noise, not a trajectory.
                r["p_harm"] = trace[0]["p_harm"]
                r["p_harm_orth"] = trace[0]["p_harm_orth"]
                r["monitor"] = trace
            # Persist the transcript *before* grading. On a cluster the grader may be
            # the only thing that can fail (no outbound network for the judge), and
            # discarding hours of generation because scoring failed is unrecoverable.
            r["turns"] = [{"text": t.text, "calls": t.calls, "results": t.results, "truncated": t.truncated} for t in turns]
            r["n_turns"] = len(turns)
            r["called"] = [c["function"] for t in turns for c in t.calls]
            r["cut_mid_call"] = any(t.cut_mid_call for t in turns)
            # Recorded per row, not just once in file-level `meta`: a wrong or stale
            # token budget silently governing a run (2026-08-28 — the confirmatory job
            # ran at 768 despite a committed fix for 1536) was only found by an
            # across-run leave-one-cluster-out sweep. `set(r["max_new_tokens"] for r in
            # rows)` now answers the same question from one file in one line.
            r["max_new_tokens"] = args.max_new_tokens
        try:
            sc = (grade if args.no_judge else grade_with_judge)(r, as_inspect_messages(turns))
            r.pop("grade_error", None)
        except Exception as exc:
            r["grade_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  grade failed ({r['cluster']}/{r['id']}/{r['condition']}): {exc}", flush=True)
            _checkpoint(n, len(todo), t0, args.out, meta, rows)
            continue
        # Write the judged fields under the same names 15_regrade uses, so the
        # analysis detects judged data identically however it was produced. Writing
        # only `full_compliance` made a judge-on run analyse as judge-off.
        suffix = "" if args.no_judge else "_judged"
        r[f"structural{suffix}"] = sc.structural if sc else {}
        r[f"score{suffix}"] = sc.score if sc else float("nan")
        r[f"full_compliance{suffix}"] = sc.full_compliance if sc else False
        r["semantic"] = sc.semantic if sc else {}
        r["judged"] = not args.no_judge
        r["unscored_criteria"] = sc.unscored if sc else []
        _checkpoint(n, len(todo), t0, args.out, meta, rows)
    _atomic_write(args.out, {"meta": meta, "rows": rows})
    if not args.no_judge:
        js = judge_stats()
        print(f"judge: {js['calls']} calls, {js['cached']} cached, {js['invalid']} invalid, "
              f"{js.get('recovered', 0)} recovered by fallback, {js['unscored']} UNSCORED "
              f"| by model {js['by_model']}")
        if js["unscored"]:
            print("  UNSCORED verdicts are not failures - report them separately, never pooled.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
