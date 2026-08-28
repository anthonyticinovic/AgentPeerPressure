"""Build the write-up artefact — narrative and figures, for an interp-literate reader.

Separate from the research log by design. The log is the working record: every wrong
turn, every retraction, the full parameter trail. This is the argument: what was asked,
what was measured, what it means. Implementation detail and the correction history stay
in the log.

What the write-up does carry is *what was verified and how* — the manipulation checks,
the sample that could ever have moved, and the bound the null actually supports. A null
reported without those is not a result, it is an absence.

Every number is read from results/*.json. Nothing here is transcribed by hand.

    uv run python scripts/08_build_writeup.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pressure.config import CFG
from pressure.provenance import assert_same_model

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
    p9 = _load("gate_p_9b.json")
    p4 = _load("gate_p_4b.json")
    pw = _load("power_9b.json")

    # Phase 1 (directions: ana/sweep/gen/sel/hand/dual) is 4B-only by design and must
    # agree internally. Phase 2 (p9/p4/pw) is deliberately cross-scale -- p4 exists to
    # be compared against p9 -- so it is not part of this check.
    assert_same_model({"inversion_analysis.json": ana, "inversion_sweep.json": sweep,
                        "arditi_generation_check.json": gen, "arditi_selection.json": sel,
                        "HANDLABEL_arditi_selected.json": hand, "dual_directions.json": dual})

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
        "gatep": _gatep(p9, p4, pw),
    }
    return TEMPLATE.replace("__DATA__", json.dumps(payload))


def _block(payload, label, ref):
    for b in payload["blocks"]:
        if b["label"].startswith(label) and b["ref"] == ref:
            return b
    return None


def _best_test(p9):
    """The strongest signal anywhere in the family, named. Reporting only the adjusted
    number hides which test produced it; reporting only the raw one hides the
    correction."""
    fam = p9.get("family_holm") or {}
    if not fam:
        return None
    key = min(fam, key=lambda k: fam[k])
    raw = next((a["p"] for b in p9["blocks"] for a in b["arms"]
                if f"{b['label']}|{a['arm']}-{b['ref']}" == key), None)
    label, contrast = key.split("|")
    return {"contrast": contrast, "outcome": label.split(" (")[0],
            "raw": raw, "holm": fam[key], "n_tests": len(fam)}


def _gatep(p9, p4, pw=None):
    """Phase 2 for the write-up: the primary contrast, the two secondary outcomes that
    carry the actual pattern, and the descriptors that keep the null honest."""
    if not p9:
        return None
    full9 = _block(p9, "FULL compliance", "C1b")
    partial9 = _block(p9, "PARTIAL", "C0")
    call9 = _block(p9, "any tool call", "C0")
    out = {
        "model": p9["meta"]["model"],
        "judge": p9["meta"].get("judge"),
        "n_rows": p9["n_rows"], "n_items": p9["n_items"], "n_clusters": p9["n_clusters"],
        "max_turns": p9["meta"]["max_turns"],
        "discordant": p9["discordant_items"],
        "inert": p9["inert_items"],
        "at_cap": p9["rows_at_turn_cap"],
        "unscored": p9["unscored_rows"],
        "full": full9,
        # The speaker-free floor is the whole point of the design: C2 - C0 would credit
        # the entire endorsement effect to peer identity. These two outcomes are where
        # the context effect is visible at all, so both are reported, Holm and all.
        "ladder": [
            {"outcome": "any tool call", "block": call9},
            {"outcome": "partial compliance", "block": partial9},
        ],
        "family": _best_test(p9),
        "power": pw,
    }
    if p4:
        f4 = _block(p4, "FULL compliance", "C1b")
        out["scale"] = {
            "model": p4["meta"]["model"], "n_items": p4["n_items"],
            "n_clusters": p4["n_clusters"],
            "c2": next(a for a in f4["arms"] if a["arm"] == "C2"),
        }
    return out


TEMPLATE = Path(__file__).with_name("_writeup_template.html").read_text()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
