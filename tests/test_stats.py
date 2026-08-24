from pressure.stats import cluster_sign_test, contrast, holm, mcnemar_exact


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


def test_cluster_sign_test_collapses_a_single_cluster():
    """Four variants of one scenario all moving together are one observation."""
    items = {("c1", str(i)): {"ref": {"y": False}, "arm": {"y": True}} for i in range(4)}

    def outcome(r):
        return r["y"]

    b, c, _, item_p = contrast(items, "ref", "arm", outcome)
    assert (b, c) == (4, 0)
    assert item_p < 0.2
    assert cluster_sign_test(items, "ref", "arm", outcome) == 1.0
