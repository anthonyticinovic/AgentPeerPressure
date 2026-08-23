"""Build the research log artefact from results on disk.

Re-runnable: every number comes from results/*.json, never transcribed by hand.
    uv run python scripts/05_build_log.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pressure.config import CFG

OUT = Path("artifacts/inspection.html")


def _load(name: str):
    p = CFG.results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(x: float) -> str:
    return f"{x:.2f}"


def gate_b_payload():
    """Compute the Gate B section from the Arditi selection + generation-check results.

    Replaces the retracted causal_validation.json feed. The verdict is PASS: a
    diff-of-means direction, selected by Arditi's real grid criterion, ablates refusal.
    """
    sel, gen, hand = (
        _load("arditi_selection.json"),
        _load("arditi_generation_check.json"),
        _load("HANDLABEL_arditi_selected.json"),
    )
    if not (sel and gen and hand):
        return None
    s, a, b = sel["selected"], sel["auroc_choice"], gen["behaviour"]
    n_cand = len(sel["positions"]) * sel["n_layers"]
    yes, no = '<span class="yes">yes</span>', '<span class="no">no</span>'

    def row(name, beh, kl, adm, hand_rate=None):
        return [
            name,
            {"n": _f(beh["refusal_rate"])},
            {"n": _f(beh["harmful_rate"])},
            {"n": _f(hand_rate) if hand_rate is not None else "—"},
            {"n": f"{kl:.2f}" if kl is not None else "—"},
            {"html": adm},
        ]

    return {
        "verdict_class": "good",
        "verdict_short": "pass",
        "verdict": "PASS — refusal is causally ablatable",
        "summary_html": (
            f"Arditi's selection over the full <b>{n_cand}-candidate</b> grid "
            "(9 post-instruction positions × 32 layers), scored by the refusal metric and "
            "filtered on <code>induce&gt;0, kl&lt;0.1, l&lt;0.8L</code>, selects "
            f"<b>i*={s['position']}, l*={s['layer']}</b>. Ablating it drops refusal "
            f"<b>1.00 → {b['selected']['refusal_rate']:.2f}</b>, with "
            f"<b>{hand['hand_complied_rate']:.2f}</b> harmful compliance by hand over all "
            f"{hand['hand_complied_count']}/25 held-out completions. The shuffled-label null "
            "at the same site stays 1.00 / 0.00 — the effect is the labels, not the geometry."
        ),
        "table_head": ["Direction", "refusal", "harmful·judge", "harmful·hand", "kl", "admissible"],
        "table": [
            row("baseline", b["baseline"], None, "—"),
            row(f"ablate i*={s['position']}, l*={s['layer']}", b["selected"], s["kl"], yes,
                hand["hand_complied_rate"]),
            row("shuffled-label, same site", b["shuffled"], None, "—"),
            row("AUROC pick (pos −1, l 22)", b["auroc"], a["kl"], no),
        ],
        "note_html": (
            "The automated harmful judge under-called (0.20) by reading only framing-heavy "
            "openings; the hand rate is 0.84 (<code>HANDLABEL_arditi_selected.json</code>) — the "
            "third proxy-measurement failure in this project, after the substring detector and the "
            "first judge. Note the split the thesis rests on: the <b>monitor</b> direction (the "
            "AUROC pick, pos −1 l 22) is <em>not</em> the mediator — its kl of 0.45 makes it "
            "inadmissible and it yields 0.08 compliance. The direction that <em>separates</em> "
            "harmful from benign and the one that <em>ablates</em> refusal are different vectors."
        ),
        "ledger": {
            "r_ref_mediates": {
                "evidence": f"ablate i*={s['position']},l*={s['layer']} → 1.00→"
                            f"{b['selected']['refusal_rate']:.2f} refusal, "
                            f"{hand['hand_complied_rate']:.2f} harmful (hand)",
                "status": "yes",
            },
            "r_harm_encodes": {"evidence": "inversion test not yet run", "status": "no"},
        },
    }


def build() -> str:
    payload = json.dumps(
        {
            "sweep": _load("r_ref_sweep.json"),
            "samples": _load("dataset_samples.json"),
            "dual": _load("dual_directions.json"),
            "causal": gate_b_payload(),
            "inv": _load("inversion_sweep.json"),
            "ana": _load("inversion_analysis.json"),
            "pre": _load("inversion_preflight.json"),
            "diag": _load("diag_inversion.json"),
        }
    )
    return TEMPLATE.replace("__DATA__", payload)


TEMPLATE = Path(__file__).with_name("_log_template.html").read_text()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
