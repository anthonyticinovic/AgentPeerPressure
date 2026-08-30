import pytest

from pressure.stats import (
    binom_exact_p,
    cluster_bootstrap_ci,
    cluster_sign_test,
    contrast,
    holm,
    mcnemar_exact,
    wilson_ci,
)


def test_mcnemar_symmetric_and_bounded():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest_approx(2 / 2**10)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_holm_is_monotone_and_step_down():
    out = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert out["a"] == pytest_approx(0.03)
    assert out["b"] == pytest_approx(0.04)
    assert out["c"] == pytest_approx(0.5)
    assert out["a"] <= out["b"] <= out["c"]


def test_wilson_ci_matches_the_textbook_n10_k0_example():
    """n=10, k=0 is the standard worked example (e.g. Wikipedia's Wilson score
    interval article): 95% CI ~ [0, 0.278]. Independently re-derived here, not just
    re-run through the same formula as the implementation."""
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.27753, abs=1e-4)


def test_wilson_ci_symmetric_at_half():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.40383, abs=1e-4)
    assert hi == pytest.approx(0.59617, abs=1e-4)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)


def test_wilson_ci_degenerate_n_zero():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_contains_the_point_estimate_and_narrows_with_n():
    for k, n in ((0, 5), (3, 5), (5, 5), (7, 200)):
        lo, hi = wilson_ci(k, n)
        assert lo <= k / n <= hi
    lo_small, hi_small = wilson_ci(3, 5)
    lo_big, hi_big = wilson_ci(300, 500)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_binom_exact_p_matches_hand_calculation():
    """5 coin flips, all one way, p0=0.5: the two equally-extreme tails (k=0, k=5)
    are the only outcomes at least as unlikely as the observed one -- 2 * 0.5**5."""
    assert binom_exact_p(0, 5, 0.5) == pytest_approx(2 * 0.5**5)
    assert binom_exact_p(5, 5, 0.5) == pytest_approx(binom_exact_p(0, 5, 0.5))


def test_binom_exact_p_is_one_at_the_mode():
    """k at (or nearest) the expected count under p0 is the single most likely
    outcome, so every outcome is at least as likely and the two-sided p-value is 1."""
    assert binom_exact_p(5, 10, 0.5) == pytest_approx(1.0)


def test_binom_exact_p_detects_a_clear_excess_over_a_small_fpr():
    """20% observed against a 5% baseline, n=100, is a real excess."""
    p = binom_exact_p(20, 100, 0.05)
    assert p < 0.001


def test_binom_exact_p_degenerate_n_zero():
    assert binom_exact_p(0, 0, 0.5) == 1.0


def test_cluster_sign_test_collapses_a_single_cluster():
    """Four variants of one scenario all moving together are one observation."""
    items = {("c1", str(i)): {"ref": {"y": False}, "arm": {"y": True}} for i in range(4)}

    def outcome(r):
        return r["y"]

    b, c, _, item_p = contrast(items, "ref", "arm", outcome)
    assert (b, c) == (4, 0)
    assert item_p < 0.2
    assert cluster_sign_test(items, "ref", "arm", outcome) == 1.0


def _rows(*cluster_sizes: int, hit: bool) -> list[dict]:
    """`n` rows per cluster, all sharing one outcome -- a convenience for building
    synthetic clustered data without hand-writing every dict."""
    return [{"cluster": f"c{ci}", "hit": hit}
            for ci, n in enumerate(cluster_sizes) for _ in range(n)]


def test_cluster_bootstrap_ci_degenerate_no_rows():
    assert cluster_bootstrap_ci([], lambda r: r["cluster"], lambda r: r["hit"]) == (0.0, 1.0)


def test_cluster_bootstrap_ci_all_hit_is_a_point_mass_at_one():
    rows = _rows(4, 4, 4, hit=True)
    lo, hi = cluster_bootstrap_ci(rows, lambda r: r["cluster"], lambda r: r["hit"])
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_cluster_bootstrap_ci_contains_the_point_estimate():
    rows = ([{"cluster": f"c{i}", "hit": True} for i in range(15)]
            + [{"cluster": f"c{i}", "hit": False} for i in range(15, 40)])
    lo, hi = cluster_bootstrap_ci(rows, lambda r: r["cluster"], lambda r: r["hit"], n_boot=2000)
    assert lo <= 15 / 40 <= hi


def test_cluster_bootstrap_ci_is_wider_than_row_level_when_clusters_are_few():
    """52 rows in 4 big clusters (13 rows each) carries far less information than 52
    independent rows -- the cluster-aware interval must be visibly wider than Wilson's,
    which is exactly the overstated-precision failure mode this exists to catch."""
    rows = (
        [{"cluster": "c0", "hit": True}] * 12 + [{"cluster": "c0", "hit": False}] * 1
        + [{"cluster": "c1", "hit": True}] * 12 + [{"cluster": "c1", "hit": False}] * 1
        + [{"cluster": "c2", "hit": False}] * 12 + [{"cluster": "c2", "hit": True}] * 1
        + [{"cluster": "c3", "hit": False}] * 12 + [{"cluster": "c3", "hit": True}] * 1
    )
    k = sum(1 for r in rows if r["hit"])
    n = len(rows)
    wlo, whi = wilson_ci(k, n)
    clo, chi = cluster_bootstrap_ci(rows, lambda r: r["cluster"], lambda r: r["hit"], n_boot=4000)
    assert (chi - clo) > (whi - wlo)


def test_cluster_bootstrap_ci_uses_the_same_clustering_key_as_cluster_sign_test():
    """A row-shuffle that preserves which cluster each row belongs to must not change
    the interval -- only cluster identity, not row order, should matter."""
    rows = (
        [{"cluster": "a", "hit": True}] * 3 + [{"cluster": "a", "hit": False}] * 1
        + [{"cluster": "b", "hit": False}] * 4
    )
    reordered = list(reversed(rows))
    ci1 = cluster_bootstrap_ci(rows, lambda r: r["cluster"], lambda r: r["hit"], seed=1)
    ci2 = cluster_bootstrap_ci(reordered, lambda r: r["cluster"], lambda r: r["hit"], seed=1)
    assert ci1 == ci2
