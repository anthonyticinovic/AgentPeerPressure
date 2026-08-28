"""Guard against building a report from artefacts computed at different model scales.

Same failure `monitor.Directions.load` already guards against: `arditi_selection.json`
and `dual_directions.json` share a results/ path across every model scale, so a later
run at one scale silently overwrites the file an earlier scale's report depends on.
Both 9B and 4B have 32 layers, so a shape check cannot catch it -- only the stamped
`model` field can. An artefact with no `model` field predates the stamp and cannot be
checked; it is reported, not silently trusted.
"""

from __future__ import annotations


def assert_same_model(sources: dict[str, dict | None]) -> None:
    """Raise if any two named, loaded artefacts disagree on their `model` field.

    `sources` maps a label (e.g. "arditi_selection.json") to its parsed JSON, or None
    if the file is missing/not yet loaded -- callers pass through whatever `_load`
    already returned rather than re-reading.
    """
    stamped = {name: d["model"] for name, d in sources.items() if d and "model" in d}
    unstamped = [name for name, d in sources.items() if d and "model" not in d]
    if unstamped:
        print(f"provenance: {', '.join(unstamped)} predate the model stamp -- not checked")
    models = set(stamped.values())
    if len(models) > 1:
        by_model = {m: [n for n, v in stamped.items() if v == m] for m in models}
        raise SystemExit(
            "Artefacts computed at different model scales are being read together:\n"
            + "\n".join(f"  {m}: {', '.join(names)}" for m, names in by_model.items())
            + "\nRegenerate the stale ones at the matching scale before rebuilding."
        )
