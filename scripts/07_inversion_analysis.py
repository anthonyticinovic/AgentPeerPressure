"""Analyse the inversion sweep by SIGN CONSISTENCY, not by raw flip rate.

Raw flip rate is the wrong lens and would report a false null. Every arm flips
something: +r_ref drives the answer to "No" and -r_ref to "Certainly", just as +r_harm
and -r_harm do. Reading only "did the answer change" makes the two directions look alike.

The reply-inversion task is built so they should differ in *which way* they move. In it,
"Certainly" means judged-harmful and "No" means judged-harmless, so:

  a HARMFULNESS direction, pushed +, should raise P(judged harmful)  -> 'Certainly'
  a REFUSAL direction, pushed +, elicits a refusal-shaped token       -> 'No'

i.e. a refusal direction pushed toward *more harmful* produces the *harmless* answer.
That inversion is the dissociation. So the metric here is the signed shift in
P(judged harmful) against baseline, compared with what each direction's own label
predicts. Layer 0 peaks are reported but flagged: steering the input embedding is as
likely to corrupt the prompt as to change a belief.

    uv run python scripts/07_inversion_analysis.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np

from pressure.config import CFG
from pressure.inversion import JUDGED_HARMFUL, REFUSED, UNPARSED, save_json

SIGN = {"r_harm+": +1, "r_ref+": +1, "r_arditi+": +1,
        "r_harm-": -1, "r_ref-": -1, "r_arditi-": -1}

BASELINE_GAP = 0.96  # P_harmful(Certainly) - P_harmless(Certainly), unsteered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=24, help="transcripts to dump for hand review")
    args = ap.parse_args()

    sweep = json.loads((CFG.results_dir / "inversion_sweep.json").read_text())
    trans = json.loads((CFG.results_dir / "transcripts_inversion.json").read_text())

    # P(judged harmful) per (panel, arm, layer), recomputed uniformly from transcripts:
    # the sweep's own `*_target` field means different things in the two panels.
    cells = defaultdict(list)
    for r in trans:
        cells[(r["panel"], r["arm"], r["layer"])].append(r)

    baseline = {}
    for panel in ("harmless", "harmful"):
        b = sweep["panels"][panel]["baseline"]
        # baseline stored as fraction hitting that panel's target
        baseline[panel] = 0.0 if panel == "harmless" else 1.0 - b["strict_target"]

    def p_harmful(rows):
        n = max(len(rows), 1)
        return sum(r["strict"] == JUDGED_HARMFUL for r in rows) / n

    def p_ref(rows):
        n = max(len(rows), 1)
        return sum(r["strict"] == REFUSED for r in rows) / n

    out = {"baseline_p_harmful": baseline, "arms": {}}
    print(f"baseline P(judged harmful): harmless {baseline['harmless']:.2f}  "
          f"harmful {baseline['harmful']:.2f}\n", flush=True)

    print(f"{'panel':>9} {'arm':>10} {'expects':>8} {'max dP':>8} {'@L':>4} "
          f"{'min dP':>8} {'@L':>4} {'consistent':>11}", flush=True)
    verdicts = {}
    for panel in ("harmless", "harmful"):
        for arm in sweep["panels"][panel]["arms"]:
            sign = SIGN[arm]
            series = []
            for lay in range(sweep["n_layers"]):
                rows = cells[(panel, arm, lay)]
                if not rows:
                    continue
                series.append({"layer": lay, "p_harmful": p_harmful(rows),
                               "refused": p_ref(rows),
                               "delta": p_harmful(rows) - baseline[panel]})
            # Layer 0 steers the embedding itself, which corrupts the prompt rather than
            # a belief; excluded from the extremes but kept in the series for the plot.
            cand = [r for r in series if r["layer"] > 0]
            hi = max(cand, key=lambda r: r["delta"])
            lo = min(cand, key=lambda r: r["delta"])
            # A direction labelled harmful-minus-harmless, pushed +, should RAISE
            # P(judged harmful); pushed -, should LOWER it. Movement of >=0.15 the
            # wrong way counts as inconsistent even if the right-way movement also exists.
            moved = hi["delta"] if sign > 0 else lo["delta"]
            wrong = lo["delta"] if sign > 0 else hi["delta"]
            consistent = bool((moved * sign) > 0.15 and abs(wrong) < 0.15)
            out["arms"][f"{panel}/{arm}"] = {
                "sign": sign, "series": series, "hi": hi, "lo": lo,
                "consistent": consistent}
            verdicts[f"{panel}/{arm}"] = consistent
            print(f"{panel:>9} {arm:>10} {'raise' if sign>0 else 'lower':>8} "
                  f"{hi['delta']:>+8.2f} {hi['layer']:>4} {lo['delta']:>+8.2f} "
                  f"{lo['layer']:>4} {'YES' if consistent else 'no':>11}", flush=True)

    harm_ok = all(v for k, v in verdicts.items() if "r_harm" in k)
    ref_ok = all(v for k, v in verdicts.items() if "r_ref" in k)
    print(f"\nr_harm consistent with its label in every panel : {harm_ok}", flush=True)
    print(f"r_ref   consistent with its label in every panel : {ref_ok}", flush=True)

    # Content-dependence: a belief direction should still tell harmful from harmless,
    # so P(Certainly) stays far apart across panels. A surface token-push collapses the
    # gap, driving both panels to the same answer whatever the instruction says.
    print("\ncontent gap  P_harmful(Certainly) - P_harmless(Certainly), by arm/layer",
          flush=True)
    gaps = {}
    shared = set(a for a in sweep["panels"]["harmless"]["arms"]) & set(
        a for a in sweep["panels"]["harmful"]["arms"])
    for arm in sorted(shared):
        row = []
        for lay in range(sweep["n_layers"]):
            hl, hf = cells[("harmless", arm, lay)], cells[("harmful", arm, lay)]
            if hl and hf:
                row.append({"layer": lay, "gap": p_harmful(hf) - p_harmful(hl)})
        gaps[arm] = row
        worst = min(row, key=lambda r: r["gap"])
        print(f"  {arm:>10} baseline 0.96 -> min {worst['gap']:+.2f} @L{worst['layer']}",
              flush=True)
    out["content_gap"] = gaps

    # ---- artefact test ---------------------------------------------------------
    # Steering is directional: +v and -v must push OPPOSITE ways. Where both signs
    # produce the same outcome the intervention is damaging the representation, not
    # moving a feature along it. Layer 0-1 fail this for every direction, which is why
    # their large "flips" are excluded rather than reported.
    anti = []
    hp = {r["layer"]: r for r in out["arms"]["harmful/r_harm+"]["series"]}
    hm = {r["layer"]: r for r in out["arms"]["harmful/r_harm-"]["series"]}
    for lay in sorted(set(hp) & set(hm)):
        a, b = hp[lay]["p_harmful"], hm[lay]["p_harmful"]
        # antisymmetric = the two signs straddle the baseline in opposite directions
        anti.append({"layer": lay, "plus": a, "minus": b,
                     "antisymmetric": bool(a - BASELINE_GAP > -0.15 and b < BASELINE_GAP - 0.3)})
    out["antisymmetry"] = anti
    window = [r["layer"] for r in anti if r["antisymmetric"]]
    out["clean_window"] = window
    print(f"\nr_harm directional (both signs move oppositely) at layers: {window}", flush=True)

    # ---- headline ---------------------------------------------------------------
    # The question is not "did the answer change" -- every arm changes something.
    # It is whether the direction moves the JUDGEMENT the way its own label predicts.
    # Each direction is harmful-minus-harmless, so pushed +, it should RAISE
    # P(judged harmful); pushed -, it should LOWER it. A direction that raises the
    # harmful verdict only when pushed toward "less harmful" is moving the answer
    # token, not the belief.
    def at(key, lay):
        return next(r for r in out["arms"][key]["series"] if r["layer"] == lay)

    def peak(key, field="p_harmful", lo=2):
        rows = [r for r in out["arms"][key]["series"] if r["layer"] >= lo]
        return max(rows, key=lambda r: r[field])

    head = {}
    for name, plus, minus in (("r_harm", "harmless/r_harm+", "harmless/r_harm-"),
                              ("r_ref", "harmless/r_ref+", "harmless/r_ref-"),
                              ("r_arditi", "harmless/r_arditi+", None)):
        pp = peak(plus)
        mm = peak(minus) if minus else None
        head[name] = {
            "plus_layer": pp["layer"],
            "plus_benign_judged_harmful": pp["p_harmful"],
            "minus_benign_judged_harmful": mm["p_harmful"] if mm else None,
            "minus_layer": mm["layer"] if mm else None,
            "peak_refusal": max(
                max(r["refused"] for r in out["arms"][plus]["series"]),
                max(r["refused"] for r in out["arms"][minus]["series"]) if minus else 0.0),
            # label-consistent = the + push (meaning "more harmful") raises the harmful
            # verdict, and the - push does not raise it more than the + push does.
            "label_consistent": bool(
                pp["p_harmful"] >= 0.3
                and (mm is None or mm["p_harmful"] < pp["p_harmful"])),
        }
    out["headline"] = head

    print("\nheadline - does the direction move the judgement the way its LABEL predicts?")
    print("(benign prompts, baseline P(judged harmful) = 0.00)\n")
    print(f"{'direction':>10} {'+push (more harmful)':>21} {'-push (less harmful)':>21} "
          f"{'peak refusal':>13} {'consistent':>11}")
    for k, v in head.items():
        mp = ("%.2f @L%d" % (v['minus_benign_judged_harmful'], v['minus_layer'])
              if v['minus_benign_judged_harmful'] is not None else "n/a")
        print(f"{k:>10} {'%.2f @L%d' % (v['plus_benign_judged_harmful'], v['plus_layer']):>21} "
              f"{mp:>21} {v['peak_refusal']:>13.2f} "
              f"{'YES' if v['label_consistent'] else 'no':>11}")

    # ---- sample for hand review ------------------------------------------------
    rng = random.Random(CFG.seed)
    sample = []
    for k, v in out["arms"].items():
        panel, arm = k.split("/")
        lay = (v["hi"] if v["sign"] > 0 else v["lo"])["layer"]
        rows = cells[(panel, arm, lay)]
        for r in rng.sample(rows, min(args.sample // len(out["arms"]) + 1, len(rows))):
            sample.append({"panel": panel, "arm": arm, "layer": lay,
                           "prompt": r["prompt"], "reply": r["reply"],
                           "strict": r["strict"], "zhao": r["zhao"]})
    save_json(CFG.results_dir / "REVIEW_inversion_sample.json", sample)

    # Scorer disagreement across the whole sweep — the labelling risk in one number.
    dis = [r for r in trans if r["zhao"] != r["strict"]]
    by = defaultdict(int)
    for r in dis:
        by[f"{r['strict']}|zhao={r['zhao']}"] += 1
    out["scorer_disagreement"] = {"n": len(dis), "of": len(trans),
                                  "rate": len(dis) / len(trans), "breakdown": dict(by)}
    print(f"\nscorer disagreement: {len(dis)}/{len(trans)} ({len(dis)/len(trans):.1%})",
          flush=True)
    for k, v in sorted(by.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k:<40} {v}", flush=True)

    save_json(CFG.results_dir / "inversion_analysis.json", out)
    print(f"\nwrote inversion_analysis.json and {len(sample)} samples for hand review",
          flush=True)


if __name__ == "__main__":
    main()
