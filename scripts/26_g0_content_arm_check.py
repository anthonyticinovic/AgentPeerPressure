"""G0 for the content arm (docs/PLAN_content_arm.md #5): render all 208 items x 4
new conditions on their assigned board, run every invariant, and dump a random
sample of prefixes for hand-reading.

Board assignment mirrors `12_peer_loop.build`: one board drawn per cluster from
`random.Random(seed)`, in cluster-id order, so the identities line up with what a
real run would use.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from pressure.boards import (  # noqa: E402
    assert_bounded_task_reference,
    assert_byte_identical_task,
    assert_speaker_free,
    load_boards,
    render,
)
from pressure.config import CFG  # noqa: E402
from pressure.data import agentharm  # noqa: E402

NEW_CONDITIONS = ("C8", "C8b", "C9", "C8n")
OLD_CONDITIONS = ("C0", "C2")  # C2 for the length-delta comparison in step 5


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CFG.iter_model)
    boards = load_boards()
    for b in boards:
        assert_speaker_free(b)
    items = agentharm()
    assert len(items) == 208, f"expected 208 items, got {len(items)}"

    rng = random.Random(CFG.seed)
    clusters = sorted({it["cluster"] for it in items})
    per_cluster = {c: rng.choice(boards) for c in clusters}

    n_checked = 0
    c8_lens, c2_lens = [], []
    samples = []
    for it in items:
        board = per_cluster[it["cluster"]]
        rendered = {
            c: render(c, it["prompt"], board, tok, item_name=it["name"])
            for c in NEW_CONDITIONS + OLD_CONDITIONS
        }
        assert_byte_identical_task(rendered, it["prompt"])
        assert_bounded_task_reference(rendered, it["prompt"], it["name"])
        n_checked += 1
        c8_lens.append(len(tok(rendered["C8"], add_special_tokens=False)["input_ids"]))
        c2_lens.append(len(tok(rendered["C2"], add_special_tokens=False)["input_ids"]))
        samples.append({"id": it["id"], "name": it["name"], "board": board.id, "prompt": it["prompt"], "rendered": rendered})

    print(f"checked {n_checked}/208 items x {len(NEW_CONDITIONS)} new conditions, all assertions passed")
    delta = [a - b for a, b in zip(c8_lens, c2_lens)]
    print(f"C8-C2 token delta: mean {sum(delta)/len(delta):.2f}, min {min(delta)}, max {max(delta)}")

    sample_rng = random.Random(12345)  # independent of build RNG, for the hand-read
    picked = sample_rng.sample(samples, 20)
    out_dir = CFG.results_dir / "g0_content_arm"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = []
    for s in picked:
        prefixes = {c: s["rendered"][c].split(s["prompt"])[0] for c in NEW_CONDITIONS}
        dump.append({"id": s["id"], "name": s["name"], "board": s["board"], "prefixes": prefixes})
    (out_dir / "hand_read_sample.json").write_text(json.dumps(dump, indent=1))
    print(f"wrote {out_dir / 'hand_read_sample.json'} (20 random items, {len(NEW_CONDITIONS)} conditions each)")


if __name__ == "__main__":
    main()
