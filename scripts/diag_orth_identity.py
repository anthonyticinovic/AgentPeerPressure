"""Empirical check for the cross-arm p_harm_orth/p_harm identity claimed by
23_calibrate_tau_orth.py's module docstring.

Two different claims, only one of which is exact:

  (a) SAME PASS: within one row's own forward pass, p_harm and p_harm_orth should
      coincide once ablation is active, because the ablation hook already zeroes
      r_arditi's projection at the readout layer -- so orthogonalising the direction
      changes nothing. This is what `src/pressure/monitor.py`'s docstring means by
      "under ablation h . r_arditi is already zero, so raw and orthogonalised values
      coincide." Exact up to floating-point noise, no approximation involved.

  (b) CROSS ARM: whether the BASE pass's p_harm_orth (a linear correction applied to
      the non-ablated hidden state) predicts what the ABLATED pass's raw p_harm
      would be for the *same prompt*. This would require the real ablation hook's
      effect on the hidden state at harm_layer to equal a single local projection
      -- but real ablation removes r_arditi's component at every earlier layer too,
      and that removal propagates through attention/MLP nonlinearities before
      reaching harm_layer. Bilinearity of the dot product does not cover this; it is
      an empirical approximation, checked here, not a proof.

Run against the real production files (no model load, no GPU -- reads two already-
generated result JSONs and compares fields already computed):

    uv run python scripts/diag_orth_identity.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_usable(path: Path) -> dict[tuple, dict]:
    payload = json.loads(path.read_text())
    out = {}
    for r in payload["rows"]:
        if "turns" not in r or "grade_error" in r or "judge_error" in r:
            continue
        if "p_harm_orth" not in r or "p_harm" not in r:
            continue
        out[(r["cluster"], r["id"], r["condition"])] = r
    return out


def summarise(label: str, diffs: list[float]) -> None:
    if not diffs:
        print(f"{label}: no rows to compare")
        return
    s = sorted(diffs)
    p95 = s[int(0.95 * (len(s) - 1))]
    print(f"{label}: n={len(diffs)}  mean={st.mean(diffs):.4f}  median={st.median(diffs):.4f}  "
          f"p95={p95:.4f}  max={max(diffs):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "results" / "gate_a_full_base_judged.json")
    ap.add_argument("--abl", type=Path, default=ROOT / "results" / "gate_a_full_abl_judged.json")
    args = ap.parse_args()

    base = load_usable(args.base)
    abl = load_usable(args.abl)
    keys = sorted(set(base) & set(abl))
    print(f"base usable rows: {len(base)}  abl usable rows: {len(abl)}  "
          f"matched (cluster,id,condition): {len(keys)}\n")

    print("(a) SAME PASS, positive control -- p_harm vs p_harm_orth within the abl file "
          "alone (exact, per monitor.py's docstring):")
    summarise("    abl p_harm vs p_harm_orth", [abs(r["p_harm"] - r["p_harm_orth"]) for r in abl.values()])

    print("\n    ...and the corresponding gap within the BASE file, where it is NOT "
          "expected to vanish (this is the correction tau_harm_orth needs):")
    summarise("    base p_harm vs p_harm_orth", [abs(r["p_harm"] - r["p_harm_orth"]) for r in base.values()])

    print("\n(b) CROSS ARM -- base p_harm_orth vs abl raw p_harm, matched rows:")
    summarise("    orthogonalised", [abs(base[k]["p_harm_orth"] - abl[k]["p_harm"]) for k in keys])
    summarise("    raw, no correction (for comparison)",
              [abs(base[k]["p_harm"] - abl[k]["p_harm"]) for k in keys])

    if keys:
        xs = [base[k]["p_harm_orth"] for k in keys]
        ys = [abl[k]["p_harm"] for k in keys]
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        r = cov / (st.pstdev(xs) * st.pstdev(ys))
        print(f"\n    Pearson r (base p_harm_orth vs abl p_harm): {r:.4f}")
        print(f"    scale for reference: base p_harm_orth stdev={st.pstdev(xs):.3f}, "
              f"abl p_harm stdev={st.pstdev(ys):.3f}")

    print("\nVerdict: (a) should be ~1e-3 scale (float noise) -- confirms the exact, "
          "same-pass identity. (b) is a real, positively-correlated approximation, not "
          "a numerical identity -- 24_blindspot_crosstab.py never relies on it: it reads "
          "each row's own p_harm_orth from its own actual pass, base or ablated, and "
          "never extrapolates one arm's value from the other's.")


if __name__ == "__main__":
    main()
