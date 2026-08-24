import pytest
import torch

from pressure.monitor import Directions, orthogonal_to


def make_dirs(**kw):
    base = dict(
        r_arditi=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        arditi_position=-3, arditi_layer=3,
        r_harm=torch.eye(4), harm_layer=1,
        r_ref=torch.eye(4), ref_layer=2,
        model="test/model",
    )
    base.update(kw)
    return Directions(**base)


def test_rejects_layer_outside_range():
    with pytest.raises(ValueError, match="layer"):
        make_dirs(arditi_layer=99)


def test_rejects_mismatched_layer_counts():
    with pytest.raises(ValueError, match="layers"):
        make_dirs(r_ref=torch.eye(8)[:, :4])


def test_rejects_mismatched_hidden_size():
    with pytest.raises(ValueError, match="hidden size"):
        make_dirs(r_arditi=torch.ones(8))


def test_orthogonal_to_removes_the_component():
    u = torch.tensor([3.0, 4.0, 0.0, 0.0])
    v = torch.tensor([10.0, 0.0, 0.0, 0.0])  # unnormalised on purpose
    w = orthogonal_to(u, v)
    assert float(w @ v) == pytest.approx(0.0, abs=1e-6)
    assert float(w[1]) == pytest.approx(4.0)


def test_orthogonalisation_only_bites_at_baseline():
    """Under ablation the stream has no r_arditi component, so the raw and
    orthogonalised projections coincide. It is the *baseline* value that
    orthogonalisation corrects, and that is what makes the two levels comparable."""
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    u = torch.tensor([0.6, 0.8, 0.0, 0.0])          # cos(u, v) = 0.6
    h_base = torch.tensor([2.0, 1.0, 0.0, 0.0])     # carries a v component
    h_abl = h_base - (h_base @ v) * v               # ablated: no v component

    assert float(h_base @ u) != pytest.approx(float(h_base @ orthogonal_to(u, v)))
    assert float(h_abl @ u) == pytest.approx(float(h_abl @ orthogonal_to(u, v)), abs=1e-6)


def test_cosines_reports_overlap_with_the_ablated_direction():
    d = make_dirs()
    cos = d.cosines()
    assert cos["harm"] == pytest.approx(0.0, abs=1e-6)  # e2 vs e1
    assert cos["ref"] == pytest.approx(0.0, abs=1e-6)   # e3 vs e1

    aligned = make_dirs(r_arditi=torch.tensor([0.0, 5.0, 0.0, 0.0]))
    assert aligned.cosines()["harm"] == pytest.approx(1.0, abs=1e-6)


def test_capture_under_ablation_sees_the_ablated_stream():
    """`ablate_all_components` hooks submodules; `capture_residual` hooks the blocks.
    Submodule hooks fire first, so the captured block output is already ablated.

    Supporting evidence only. A toy model cannot establish that embed_tokens,
    self_attn/linear_attn and mlp are *all* the residual writers in Qwen3.5 -- that is
    checked against real weights by `17_cluster_preflight.py --gate-a`. But the toy
    must at least carry residual connections, or it does not exercise the property.
    """
    import torch.nn as nn

    from pressure.causal import ablate_all_components
    from pressure.hooks import capture_residual

    hidden = 16

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(hidden, hidden, bias=False)
            self.mlp = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            x = x + self.self_attn(x)   # real residual connections: ablation must
            return x + self.mlp(x)      # cover every writer, not just the last

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, hidden)
            self.layers = nn.ModuleList([Block() for _ in range(2)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.device = torch.device("cpu")

        def forward(self, input_ids):
            h = self.model.embed_tokens(input_ids)
            for b in self.model.layers:
                h = b(h)
            return h

    torch.manual_seed(0)
    m = Model()
    ids = torch.arange(6).unsqueeze(0)
    v = torch.randn(hidden)
    v = v / v.norm()

    before: dict[int, torch.Tensor] = {}
    with capture_residual(m, before, [5]):
        m(ids)
    after: dict[int, torch.Tensor] = {}
    with ablate_all_components(m, v):
        with capture_residual(m, after, [5]):
            m(ids)

    proj_before = abs(float(before[1][0] @ v))
    proj_after = abs(float(after[1][0] @ v))
    assert proj_before > 1e-3, f"direction carries no signal to begin with: {proj_before}"
    assert proj_after < 1e-5, f"capture saw an unablated stream: {proj_after}"
