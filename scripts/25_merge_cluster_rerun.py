"""Merge a targeted single-cluster re-run back into a main judged results file.

Built for the cluster-27 (paper plagiarism) token-budget casualty: 71/72 rows hit
--max-new-tokens mid tool-call, and the judge correctly scored them non-compliant
(the rewrite was incomplete, the upload never happened) rather than because grading
was skipped. `12_peer_loop.py --only-cluster 27 --force-board board_03
--max-new-tokens 8192` re-runs just that cluster at a budget that should let it
finish; this script replaces the old truncated rows with the new complete ones.

    uv run python scripts/25_merge_cluster_rerun.py \\
        --main results/gate_a_full_base_judged.json \\
        --rerun results/gate_a_cluster27_base_judged.json \\
        --cluster 27

The re-run itself must be produced with the same --conditions and --monitor
setting as --main (check --main's own meta["cmd"] for the exact flags used), then
graded with 15_regrade.py before being passed here as --rerun. A --conditions
mismatch is caught indirectly (as a row-key mismatch); a --monitor mismatch is
checked directly, since row replacement is whole-row and would otherwise silently
drop p_harm/p_harm_orth/monitor fields from rows that already carried them.

Refuses to run unless every row the rerun is meant to replace is actually present
in --main, and every row in --rerun matches --cluster -- a silent partial merge
would be worse than not merging at all. Writes --main in place via a temp file and
rename (matching 12_peer_loop.py's _atomic_write -- a truncate-in-place write would
corrupt the sole copy of the main results file on a crash mid-write), after copying
the untouched original to <main>.pre_merge.json so the merge is reversible. That
backup is only ever written once: it is meant to be the true pre-any-merge state,
not a rolling snapshot a second merge could itself clobber.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True, help="judged file to update in place")
    ap.add_argument("--rerun", type=Path, required=True, help="judged output of the targeted re-run")
    ap.add_argument("--cluster", required=True, help="cluster id being replaced, e.g. 27")
    args = ap.parse_args()

    main_payload = json.loads(args.main.read_text())
    rerun_payload = json.loads(args.rerun.read_text())
    main_rows, rerun_rows = main_payload["rows"], rerun_payload["rows"]

    if rerun_payload["meta"].get("model") != main_payload["meta"].get("model"):
        raise SystemExit(f"model mismatch: rerun is {rerun_payload['meta'].get('model')}, "
                          f"main is {main_payload['meta'].get('model')}")
    if rerun_payload["meta"].get("ablate") != main_payload["meta"].get("ablate"):
        raise SystemExit(f"ablate mismatch: rerun={rerun_payload['meta'].get('ablate')} "
                          f"main={main_payload['meta'].get('ablate')} -- wrong arm's file")
    if rerun_payload["meta"].get("only_cluster") != args.cluster:
        raise SystemExit(f"--rerun was built with only_cluster="
                          f"{rerun_payload['meta'].get('only_cluster')!r}, not {args.cluster!r}")
    if rerun_payload["meta"].get("monitor") != main_payload["meta"].get("monitor"):
        raise SystemExit(f"monitor mismatch: rerun={rerun_payload['meta'].get('monitor')} "
                          f"main={main_payload['meta'].get('monitor')} -- row replacement is "
                          "whole-row, so this would silently drop p_harm/p_harm_orth/monitor "
                          "from every replaced row")

    off_cluster = [r for r in rerun_rows if r.get("cluster") != args.cluster]
    if off_cluster:
        raise SystemExit(f"--rerun contains {len(off_cluster)} rows outside cluster "
                          f"{args.cluster!r} -- refusing to merge")

    rerun_key_list = [(r["cluster"], r["id"], r["condition"]) for r in rerun_rows]
    dupes = {k for k in rerun_key_list if rerun_key_list.count(k) > 1}
    if dupes:
        raise SystemExit(f"--rerun has duplicate row keys: {sorted(dupes)[:3]} -- refusing "
                          "a merge that would silently pick whichever occurrence sorts last")

    by_key = {(r["cluster"], r["id"], r["condition"]): i
              for i, r in enumerate(main_rows) if r.get("cluster") == args.cluster}
    rerun_keys = set(rerun_key_list)
    missing = rerun_keys - set(by_key)
    if missing:
        raise SystemExit(f"{len(missing)} rerun rows have no matching row in --main "
                          f"(e.g. {sorted(missing)[:3]}) -- refusing a partial merge")
    if set(by_key) - rerun_keys:
        left = sorted(set(by_key) - rerun_keys)
        raise SystemExit(f"{len(left)} existing cluster-{args.cluster} rows in --main have no "
                          f"replacement in --rerun (e.g. {left[:3]}) -- refusing a partial merge")

    old_cut = sum(1 for r in main_rows if r.get("cluster") == args.cluster and r.get("cut_mid_call"))
    board_before = {main_rows[i]["board"] for i in by_key.values()}
    board_after = {r["board"] for r in rerun_rows}
    if board_before != board_after:
        raise SystemExit(f"board mismatch: --main cluster {args.cluster} used {board_before}, "
                          f"--rerun used {board_after} -- rows would not be comparable")

    for r in rerun_rows:
        key = (r["cluster"], r["id"], r["condition"])
        main_rows[by_key[key]] = r
    new_cut = sum(1 for r in main_rows if r.get("cluster") == args.cluster and r.get("cut_mid_call"))

    backup = args.main.with_suffix(".pre_merge.json")
    backup_was_fresh = not backup.exists()
    if backup_was_fresh:
        shutil.copy2(args.main, backup)

    tmp = args.main.with_suffix(args.main.suffix + ".tmp")
    tmp.write_text(json.dumps(main_payload, indent=1))
    tmp.replace(args.main)

    print(f"merged {len(rerun_rows)} cluster-{args.cluster} rows into {args.main}")
    print(f"  cut_mid_call in this cluster: {old_cut} -> {new_cut}")
    if backup_was_fresh:
        print(f"  original backed up to {backup}")
    else:
        print(f"  {backup} already existed (kept as the pre-any-merge original, "
              "not refreshed to this run's input)")


if __name__ == "__main__":
    main()
