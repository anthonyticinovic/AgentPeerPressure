"""The AgentHarm harness. These guard the two ways it silently stops measuring:
a tool-call format that does not match the model, and a schema that does not match
the benchmark's own tool set."""

from __future__ import annotations

import pytest

from pressure.data import agentharm
from pressure.tools import all_tools, schemas_for



def test_every_corpus_tool_is_vendored():
    needed = {f for i in agentharm(harmful=True) for f in i["target_functions"]}
    assert not needed - set(all_tools())


def test_schemas_are_well_formed():
    for name, spec in all_tools().items():
        assert spec.description, name
        fn = spec.as_schema()["function"]
        assert fn["name"] == name
        for arg, meta in fn["parameters"]["properties"].items():
            assert meta["type"] in {"string", "integer", "number", "boolean", "array"}, (name, arg)
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_only_target_tools_are_exposed():
    """Their default is n_irrelevant_tools=0; exposing extras would change the task."""
    schemas = schemas_for(["post_on_x", "download_pdf"])
    assert [s["function"]["name"] for s in schemas] == ["post_on_x", "download_pdf"]


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        schemas_for(["no_such_tool"])












def test_list_annotations_become_arrays():
    """list[str] published as "string" contradicts the tool's own description."""
    for name, param in [("send_email", "attachments"), ("fill_in_story_template", "characters"),
                        ("upload_tiktok_video", "tags")]:
        meta = all_tools()[name].as_schema()["function"]["parameters"]["properties"][param]
        assert meta["type"] == "array", (name, param)
        assert meta["items"]["type"] == "string"


def test_schema_types_match_the_vendored_source():
    """Guards the whole set, not the three we happen to know about."""
    import ast as _ast
    from pressure.tools import TOOLS_DIR
    for name, spec in all_tools().items():
        tree = _ast.parse((TOOLS_DIR / f"{name}.py").read_text())
        run = next(n for n in _ast.walk(tree)
                   if isinstance(n, (_ast.AsyncFunctionDef, _ast.FunctionDef)) and n.name == "run")
        for a in run.args.args:
            declared = spec.as_schema()["function"]["parameters"]["properties"][a.arg]["type"]
            if isinstance(a.annotation, _ast.Subscript):
                assert declared == "array", f"{name}.{a.arg} is subscripted but published as {declared}"
