"""Classification and table logic in scripts/24_blindspot_crosstab.py, against
synthetic rows -- no GPU, no real artefacts. `scripts/` is not a package, so the
module is loaded by path, the same way the script itself is invoked."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "24_blindspot_crosstab.py"
_spec = importlib.util.spec_from_file_location("blindspot_crosstab", _PATH)
crosstab_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_PATH.parents[1] / "src"))
_spec.loader.exec_module(crosstab_mod)


def row(compliant: bool, p_harm_orth: float, condition: str = "C2", cluster: str = "c0") -> dict:
    return {"full_compliance_judged": compliant, "p_harm_orth": p_harm_orth,
            "condition": condition, "cluster": cluster}


def test_comply_and_flagged_read_the_right_fields():
    r = row(True, 5.0)
    assert crosstab_mod.comply(r) is True
    assert crosstab_mod.flagged(r, tau=1.0) is True
    assert crosstab_mod.flagged(r, tau=10.0) is False


def test_flagged_is_a_strict_greater_than_at_the_threshold():
    """A row sitting exactly on tau is not flagged -- tau is calibrated as a
    quantile of the harmless side, so equality belongs to the unflagged majority."""
    assert crosstab_mod.flagged(row(True, 1.0), tau=1.0) is False
    assert crosstab_mod.flagged(row(True, 1.0000001), tau=1.0) is True


def test_crosstab_counts_all_four_cells():
    rows = [
        row(True, 5.0, cluster="c0"),   # comply & flagged
        row(True, 5.0, cluster="c1"),   # comply & flagged
        row(True, -5.0, cluster="c1"),  # comply & unflagged
        row(False, 5.0, cluster="c2"),  # refuse & flagged
        row(False, -5.0, cluster="c2"), # refuse & unflagged
        row(False, -5.0, cluster="c3"), # refuse & unflagged
    ]
    ct = crosstab_mod.crosstab(rows, tau=0.0, n_boot=2000)
    assert ct["n"] == 6
    assert ct["n_clusters"] == 4
    assert ct["comply_and_flagged"] == 2
    assert ct["comply_and_unflagged"] == 1
    assert ct["refuse_and_flagged"] == 1
    assert ct["refuse_and_unflagged"] == 2
    assert ct["comply_and_flagged_rate"] == pytest.approx(2 / 6)
    lo, hi = ct["comply_and_flagged_ci95"]
    assert lo <= ct["comply_and_flagged_rate"] <= hi
    wlo, whi = ct["comply_and_flagged_ci95_wilson_naive"]
    assert wlo <= ct["comply_and_flagged_rate"] <= whi
    # The two intervals need not coincide -- they answer different questions -- but
    # neither should be degenerate at this n.
    assert (hi - lo) > 0.0
    assert (whi - wlo) > 0.0


def test_crosstab_empty_input_does_not_divide_by_zero():
    ct = crosstab_mod.crosstab([], tau=0.0)
    assert ct["n"] == 0
    assert ct["n_clusters"] == 0
    assert list(ct["comply_and_flagged_ci95"]) == [0.0, 1.0]
    assert list(ct["comply_and_flagged_ci95_wilson_naive"]) == [0.0, 1.0]


def test_exploratory_rate_conditions_on_compliance_only():
    """The exploratory statistic's denominator is compliant rows, unlike the cross-
    tab cell's rate, which is out of every row -- the two must not be conflated."""
    rows = [
        row(True, 5.0),   # compliant, flagged
        row(True, -5.0),  # compliant, unflagged
        row(False, 5.0),  # refused, flagged -- excluded from this statistic
    ]
    out = crosstab_mod.exploratory_conditional_flag_rate(rows, tau=0.0, p0=0.05)
    assert out["n_compliant"] == 2
    assert out["n_flagged_among_compliant"] == 1
    assert out["rate"] == pytest.approx(0.5)
    assert 0.0 <= out["p_value_two_sided"] <= 1.0


def test_exploratory_rate_empty_compliant_set():
    out = crosstab_mod.exploratory_conditional_flag_rate([row(False, 5.0)], tau=0.0, p0=0.05)
    assert out["n_compliant"] == 0
    import math
    assert math.isnan(out["rate"])


def test_per_condition_partitions_by_condition_and_sums_to_the_pooled_table():
    rows = [
        row(True, 5.0, "C2"), row(False, -5.0, "C2"),
        row(True, 5.0, "C3"), row(True, -5.0, "C3"), row(False, 5.0, "C3"),
    ]
    by_cond = crosstab_mod.per_condition(rows, tau=0.0)
    assert set(by_cond) == {"C2", "C3"}
    assert by_cond["C2"]["n"] == 2
    assert by_cond["C3"]["n"] == 3
    total_a = sum(c["comply_and_flagged"] for c in by_cond.values())
    pooled = crosstab_mod.crosstab(rows, tau=0.0)
    assert total_a == pooled["comply_and_flagged"]


def test_load_rows_drops_ungraded_and_unmonitored_rows(tmp_path):
    payload = {"rows": [
        {"turns": [1], "full_compliance_judged": True, "p_harm_orth": 1.0, "condition": "C2"},
        {"turns": [1], "grade_error": "boom", "full_compliance_judged": True,
         "p_harm_orth": 1.0, "condition": "C2"},
        {"turns": [1], "judge_error": "boom", "full_compliance_judged": True,
         "p_harm_orth": 1.0, "condition": "C2"},
        {"full_compliance_judged": True, "p_harm_orth": 1.0, "condition": "C2"},  # no turns
        {"turns": [1], "full_compliance_judged": True, "condition": "C2"},  # no p_harm_orth
    ]}
    path = tmp_path / "rows.json"
    path.write_text(__import__("json").dumps(payload))
    rows = crosstab_mod.load_rows(path)
    assert len(rows) == 1
