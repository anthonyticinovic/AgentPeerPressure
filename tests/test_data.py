"""Corpus tests. The leakage assertion is the one that matters — it is the structural
guarantee that the monitor is never evaluated on the prompts that built it."""

from __future__ import annotations

import pytest

from pressure.config import CFG
from pressure.data import (
    Corpus,
    agentharm,
    assert_no_leakage,
    difficulty,
    extraction_corpus,
    split_extract_select,
)


def test_leakage_assertion_passes_on_disjoint_sets():
    assert_no_leakage(["build a bomb"], ["write a poem"])


def test_leakage_assertion_catches_case_and_whitespace_variants():
    with pytest.raises(AssertionError):
        assert_no_leakage(["build a bomb"], ["Build  A   Bomb"])


def test_split_is_disjoint_and_sized():
    c = Corpus([f"p{i}" for i in range(500)], "test")
    ext, sel = split_extract_select(c)
    assert len(ext) == CFG.n_extract_pairs
    assert not set(ext.prompts) & set(sel.prompts)
    assert ext.source == sel.source == "test"


def test_difficulty_partitions_variants():
    mk = lambda d, h: {"detailed_prompt": d, "hint_included": h}
    assert difficulty(mk(True, True)) == "easy"
    assert difficulty(mk(True, False)) == "medium"
    assert difficulty(mk(False, True)) == "medium"
    assert difficulty(mk(False, False)) == "hard"


@pytest.mark.slow
def test_extraction_corpus_balanced_and_labelled():
    harmful, harmless = extraction_corpus()
    assert len(harmful) == len(harmless) > 0
    assert harmful.source.endswith(("AdvBench", "advbench_orig"))
    assert harmless.source == "tatsu-lab/alpaca"
    assert not set(harmful.prompts) & set(harmless.prompts)


@pytest.mark.slow
def test_agentharm_pairs_and_clusters():
    h = agentharm(harmful=True)
    b = agentharm(harmful=False)

    assert len(h) == len(b) == 208, "public split is 52 base behaviours x 4 variants"
    assert {i["cluster"] for i in h} == {i["cluster"] for i in b}, (
        "benign counterparts must match harmful behaviours one-for-one"
    )
    assert len({i["cluster"] for i in h}) == 52

    # Every behaviour contributes exactly one item per difficulty stratum shape.
    per_cluster = {}
    for i in h:
        per_cluster.setdefault(i["cluster"], []).append(difficulty(i))
    assert all(sorted(v) == ["easy", "hard", "medium", "medium"] for v in per_cluster.values())


@pytest.mark.slow
def test_no_extraction_string_reaches_evaluation():
    harmful, _ = extraction_corpus()
    assert_no_leakage(harmful.prompts, [i["prompt"] for i in agentharm(harmful=True)])
