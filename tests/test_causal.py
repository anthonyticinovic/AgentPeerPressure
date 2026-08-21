"""Control-direction tests.

The controls matter as much as the signal: an earlier gate passed for noise because the
null was mis-built. These pin down the two real nulls (shuffled-label diff-of-means and
top principal component) and the random baseline, so a control that silently degenerates
fails here rather than in an analysis.
"""

from __future__ import annotations

import torch

from pressure.causal import (
    random_directions,
    shuffled_label_direction,
    top_pc_direction,
)


def test_random_directions_are_unit_and_reproducible():
    a = random_directions(64, n=3, seed=0)
    b = random_directions(64, n=3, seed=0)
    assert len(a) == 3
    for v, w in zip(a, b):
        assert torch.allclose(v, w), "same seed must give the same vectors"
        assert v.norm().item() == torch.tensor(1.0).item() or abs(v.norm().item() - 1.0) < 1e-5


def test_shuffled_label_direction_shape_and_determinism():
    ah = torch.randn(20, 4, 8)
    ab = torch.randn(20, 4, 8)
    d1 = shuffled_label_direction(ah, ab, seed=0)
    d2 = shuffled_label_direction(ah, ab, seed=0)
    assert d1.shape == (4, 8)
    assert torch.allclose(d1, d2)


def test_shuffled_label_direction_kills_the_class_signal():
    """With a strong class offset, the true diff-of-means is large; the shuffled-label
    version must be much smaller, since it averages the offset away."""
    torch.manual_seed(0)
    offset = torch.zeros(4, 8)
    offset[:, 0] = 10.0  # class signal lives on one coordinate
    ah = torch.randn(50, 4, 8) + offset
    ab = torch.randn(50, 4, 8)

    true_dir = ah.mean(0) - ab.mean(0)
    shuf = shuffled_label_direction(ah, ab, seed=0)
    assert true_dir.norm() > 5 * shuf.norm(), "shuffled null should not carry the class offset"


def test_top_pc_direction_finds_the_dominant_axis():
    """A cloud stretched along one coordinate must yield a top PC aligned with it."""
    torch.manual_seed(0)
    x = torch.randn(200, 2, 8)
    x[:, 0, 3] *= 30.0  # layer 0 dominated by coordinate 3
    pc = top_pc_direction(x)
    assert pc.shape == (2, 8)
    assert pc[0].abs().argmax().item() == 3, "top PC of layer 0 should point along its variance"
    assert abs(pc[0].norm().item() - 1.0) < 1e-4, "PCs are unit vectors"
