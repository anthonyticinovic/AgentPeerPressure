"""Build the write-up artefact — narrative and figures, for an interp-literate reader.

Separate from the research log by design. The log is the working record: every wrong
turn, every retraction, the full parameter trail. This is the argument: what was asked,
what was measured, what it means. Implementation detail and the correction history stay
in the log, because at this stage they are noise against a result that is itself only a
precondition for the main experiment.

Every number is read from results/*.json. Nothing here is transcribed by hand.

    uv run python scripts/08_build_writeup.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pressure.config import CFG

OUT = Path("artifacts/writeup.html")


def _load(name):
    p = CFG.results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def build() -> str:
    ana = _load("inversion_analysis.json")
    sweep = _load("inversion_sweep.json")
    gen = _load("arditi_generation_check.json")
    sel = _load("arditi_selection.json")
    hand = _load("HANDLABEL_arditi_selected.json")
    dual = _load("dual_directions.json")

    payload = {
        "head": ana["headline"],
        "arms": {k: v["series"] for k, v in ana["arms"].items()},
        "unreadable": {k: [r["layer"] for r in v["series"] if not r["readable"]]
                       for k, v in ana["arms"].items()},
        "baseline": ana["baseline_p_harmful"],
        "window": ana["clean_window"],
        "question": sweep["question"],
        "coeff": sweep["coeff"],
        "n_layers": sweep["n_layers"],
        "gateb": {
            "position": sel["selected"]["position"],
            "layer": sel["selected"]["layer"],
            "kl": sel["selected"]["kl"],
            "auroc_kl": sel["auroc_choice"]["kl"],
            "n_candidates": len(sel["positions"]) * sel["n_layers"],
            "refusal_base": gen["behaviour"]["baseline"]["refusal_rate"],
            "refusal_abl": gen["behaviour"]["selected"]["refusal_rate"],
            "refusal_shuf": gen["behaviour"]["shuffled"]["refusal_rate"],
            "hand": hand["hand_complied_rate"],
            "auroc_hand": gen["behaviour"]["auroc"]["harmful_rate"],
        },
        "auroc": {
            "ref": dual["positions"]["context_last"]["auroc"],
            "harm": dual["positions"]["task_last"]["auroc"],
        },
        "model": sweep.get("model", "Qwen/Qwen3.5-4B"),
    }
    return TEMPLATE.replace("__DATA__", json.dumps(payload))


TEMPLATE = Path(__file__).with_name("_writeup_template.html").read_text()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
