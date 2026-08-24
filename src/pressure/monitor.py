"""Read three directions off one forward pass per turn.

Nothing is injected here. `r_arditi`'s projection is a fidelity check that the
ablation hook fired, and is never a result.

The three sites are not interchangeable:

    r_arditi   end-relative offset i*, chosen by the causal sweep in 03
    r_harm     task_last    (Zhao t_inst)      -- constant within a row, see below
    r_ref      context_last (Zhao t_post-inst) -- varies per turn

Why every projection is reported twice
--------------------------------------
`ablate_all_components` zeroes the residual stream along `r_arditi` at every layer
and position, so a raw projection onto `r_harm` or `r_ref` loses
`cos(r_arditi, u) * (h . r_arditi)` by arithmetic rather than by measurement. At 4B
those cosines are 0.26 (`r_ref`) and 0.17 (`r_harm`), and the AUROC-optimal candidate
from the same grid sits at 0.997 with `r_ref`.

Note where the correction actually bites. Under ablation `h . r_arditi` is already
zero, so the raw and orthogonalised values coincide; it is the *baseline* value that
orthogonalisation changes. That is exactly what makes the two ablation levels
comparable, and it is why the orthogonalised number is the result and the raw one is
only a diagnostic.

Why r_harm is stored once per row
---------------------------------
`task_last` is an absolute index inside the user message. `run_loop` appends assistant
and tool messages *after* it, so under causal attention the residual at that index
depends only on tokens that never change. `p_harm` is therefore the same at every turn.
Per-turn variation is floating-point noise from attention kernels tiling differently at
different sequence lengths, and would look exactly like a trajectory. Gate A2 measures
its scale; the analysis asserts real data stays under that bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .hooks import capture_residual, resolve_positions


def orthogonal_to(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The component of `u` perpendicular to `v`. `v` need not be normalised."""
    vh = v / v.norm()
    return u - (u @ vh) * vh


@dataclass(frozen=True)
class Directions:
    """The three vectors, with the position and layer each is read at."""

    r_arditi: torch.Tensor   # (hidden,) at (i*, l*), unnormalised: Arditi eq. 3
    arditi_position: int     # end-relative offset i*
    arditi_layer: int
    r_harm: torch.Tensor     # (n_layers, hidden), unit rows
    harm_layer: int
    r_ref: torch.Tensor      # (n_layers, hidden), unit rows
    ref_layer: int
    model: str

    def __post_init__(self) -> None:
        n = self.r_harm.shape[0]
        if self.r_ref.shape[0] != n:
            raise ValueError(f"r_ref has {self.r_ref.shape[0]} layers, r_harm has {n}")
        if self.r_arditi.shape[-1] != self.r_harm.shape[-1]:
            raise ValueError(
                f"hidden size {self.r_arditi.shape[-1]} != {self.r_harm.shape[-1]}"
            )
        for name, layer in (("arditi", self.arditi_layer),
                            ("harm", self.harm_layer),
                            ("ref", self.ref_layer)):
            if not 0 <= layer < n:
                raise ValueError(f"{name} layer {layer} outside 0..{n - 1}")

    @property
    def n_layers(self) -> int:
        return self.r_harm.shape[0]

    def cosines(self) -> dict[str, float]:
        """Overlap of each monitored vector with the one being ablated.

        High overlap means ablation removes the monitored signal by construction and
        orthogonalising leaves too little behind to mean anything. Gate on this before
        running -- and never re-select i* to dodge it, which would fit the direction
        to the result.
        """
        def cos(u: torch.Tensor) -> float:
            return float((u @ self.r_arditi) / (u.norm() * self.r_arditi.norm()))

        return {"harm": cos(self.r_harm[self.harm_layer]),
                "ref": cos(self.r_ref[self.ref_layer])}

    @classmethod
    def load(cls, results_dir: Path) -> "Directions":
        a = torch.load(results_dir / "arditi_selected.pt", map_location="cpu",
                       weights_only=False)
        d = torch.load(results_dir / "dual_raw.pt", map_location="cpu",
                       weights_only=False)
        if a["model"] != d.get("model"):
            raise ValueError(
                f"arditi_selected.pt is {a['model']} but dual_raw.pt is {d.get('model')}. "
                "Rebuild both at the same scale. A layer-count check cannot catch this "
                "because both models have 32 layers."
            )
        return cls(
            r_arditi=a["r_arditi"].float(),
            arditi_position=int(a["position"]),
            arditi_layer=int(a["layer"]),
            r_harm=d["r_harm"].float(),
            harm_layer=int(d["task_last"]["l_star"]),
            r_ref=d["r_ref"].float(),
            ref_layer=int(d["context_last"]["l_star"]),
            model=a["model"],
        )


@torch.no_grad()
def projections(model, tok, prompt: str, task_text: str, dirs: Directions) -> dict:
    """Raw and orthogonalised projections at three sites, from one forward pass.

    Calls `model.model(...)`, not `model(...)`: the hooks live on `model.model.layers`
    and `model.model.embed_tokens`, and computing logits over a ~151k vocabulary at
    every position of an agentic-length context costs gigabytes for nothing.

    Runs its own prefill rather than reusing `generate`'s, because `capture_residual`
    fires on every forward call including each single-token decode step, where
    prompt-relative indices are meaningless.
    """
    pos = resolve_positions(tok, prompt, task_text)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    seq_len = enc["input_ids"].shape[1]

    i_arditi = seq_len + dirs.arditi_position
    if i_arditi < 0:
        raise ValueError(
            f"prompt of {seq_len} tokens is shorter than offset {dirs.arditi_position}"
        )
    # i* = -1 is an admissible outcome of the causal sweep, in which case the Arditi
    # site coincides with context_last. A repeated index is fine here.
    sites = {"arditi": i_arditi, "harm": pos["task_last"], "ref": pos["context_last"]}
    order = sorted(sites, key=lambda k: sites[k])

    store: dict[int, torch.Tensor] = {}
    with capture_residual(model, store, [sites[k] for k in order]):
        model.model(**enc)

    at = {name: {layer: acts[i] for layer, acts in store.items()}
          for i, name in enumerate(order)}

    h_harm, h_ref = at["harm"][dirs.harm_layer], at["ref"][dirs.ref_layer]
    v_harm, v_ref = dirs.r_harm[dirs.harm_layer], dirs.r_ref[dirs.ref_layer]
    return {
        "p_harm": float(h_harm @ v_harm),
        "p_ref": float(h_ref @ v_ref),
        "p_harm_orth": float(h_harm @ orthogonal_to(v_harm, dirs.r_arditi)),
        "p_ref_orth": float(h_ref @ orthogonal_to(v_ref, dirs.r_arditi)),
        "p_arditi": float(at["arditi"][dirs.arditi_layer] @ dirs.r_arditi),
        "seq_len": seq_len,
        "i_arditi": i_arditi,
        "i_harm": pos["task_last"],
        "i_ref": pos["context_last"],
    }
