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

import json
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
    tau_harm_orth: float | None = None  # frozen threshold on p_harm_orth, see
    # scripts/23_calibrate_tau_orth.py. None until that script has run for this model;
    # callers that need it (scripts/24_blindspot_crosstab.py) check and fail loudly
    # themselves rather than this generic loader guessing what each caller requires.

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

        # tau_harm_orth lives in dual_directions.json, not the .pt caches -- it is a
        # scalar written by scripts/23_calibrate_tau_orth.py, not a tensor. The JSON is
        # optional here (older artefact sets predate it) and, unlike the .pt pairing
        # above, a model mismatch is warned rather than fatal: dual_directions.json has
        # been found stale relative to the .pt files in this project before (it is
        # fully rewritten by 02_dual_directions.py and only patched in place by 23's
        # --iter reruns, never touched by 03, so the two can drift out of sync), and
        # most callers of Directions.load() (e.g. 12_peer_loop.py) do not need tau at
        # all. A caller that does need it (24_blindspot_crosstab.py) checks for None
        # itself and fails loudly there, where the message can be specific.
        tau_harm_orth = None
        json_path = results_dir / "dual_directions.json"
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            if meta.get("model") != a["model"]:
                print(f"WARNING: {json_path} is for {meta.get('model')} but the loaded "
                      f"directions are {a['model']}; ignoring its tau_harm_orth (stale).")
            else:
                # 23_calibrate_tau_orth.py only ever writes the dict form (value plus
                # calibration metadata) -- no bare-float branch here, since nothing
                # produces one and testing a shape nothing writes is speculative, not
                # defensive.
                tau = meta.get("tau_harm_orth")
                tau_harm_orth = float(tau["value"]) if isinstance(tau, dict) else None

        return cls(
            r_arditi=a["r_arditi"].float(),
            arditi_position=int(a["position"]),
            arditi_layer=int(a["layer"]),
            r_harm=d["r_harm"].float(),
            harm_layer=int(d["task_last"]["l_star"]),
            r_ref=d["r_ref"].float(),
            ref_layer=int(d["context_last"]["l_star"]),
            model=a["model"],
            tau_harm_orth=tau_harm_orth,
        )


@torch.no_grad()
def projections(model, tok, prompt: str, task_text: str, dirs: Directions,
                 task_search_upto: int | None = None) -> dict:
    """Raw and orthogonalised projections at three sites, from one forward pass.

    Calls `model.model(...)`, not `model(...)`: the hooks live on `model.model.layers`
    and `model.model.embed_tokens`, and computing logits over a ~151k vocabulary at
    every position of an agentic-length context costs gigabytes for nothing.

    Runs its own prefill rather than reusing `generate`'s, because `capture_residual`
    fires on every forward call including each single-token decode step, where
    prompt-relative indices are meaningless.

    `task_search_upto`: see `resolve_positions` — bounds where `task_text` is searched
    for, so a later turn's growing `prompt` can't have `task_last` hijacked by the
    model's own echo of the task text. The caller (`12_peer_loop.py`) passes the
    length of the turn-0 prompt on every call, not just the first.
    """
    pos = resolve_positions(tok, prompt, task_text, search_upto=task_search_upto)
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

    # Every layer's activation at the harm/ref token position is already sitting in
    # `at` — the hook fires on every block regardless of which single layer is the
    # AUROC-selected "official" one. Recording the full profile costs one dot product
    # per layer (no extra forward pass) and is orthogonalisation-valid at every layer,
    # not just the selected one, because `ablate_all_components` zeroes r_arditi's
    # component at every layer's residual stream, not only `arditi_layer`'s. Added
    # 2026-08-28 for future robustness checks (e.g. is harm_layer/ref_layer actually
    # the best choice, does the signal appear earlier/later) that a single-layer trace
    # can't answer without a full rerun.
    by_layer_harm = [float(at["harm"][l] @ dirs.r_harm[l]) for l in range(dirs.n_layers)]
    by_layer_ref = [float(at["ref"][l] @ dirs.r_ref[l]) for l in range(dirs.n_layers)]
    by_layer_harm_orth = [float(at["harm"][l] @ orthogonal_to(dirs.r_harm[l], dirs.r_arditi))
                           for l in range(dirs.n_layers)]
    by_layer_ref_orth = [float(at["ref"][l] @ orthogonal_to(dirs.r_ref[l], dirs.r_arditi))
                          for l in range(dirs.n_layers)]

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
        "p_harm_by_layer": by_layer_harm,
        "p_ref_by_layer": by_layer_ref,
        "p_harm_orth_by_layer": by_layer_harm_orth,
        "p_ref_orth_by_layer": by_layer_ref_orth,
    }
