"""The AgentHarm harness. These guard the two ways it silently stops measuring:
a tool-call format that does not match the model, and a schema that does not match
the benchmark's own tool set."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from pressure.config import CFG
from pressure.data import agentharm
from pressure.tools import SYSTEM_PROMPT, agentic_prompt, all_tools, called_tools, schemas_for


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CFG.iter_model)


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


def test_qwen_xml_tool_call_is_parsed():
    """Qwen3.5 emits nested XML, not JSON. Assuming JSON scores every call as no-call."""
    reply = (
        "I'll help.\n<tool_call>\n<function=download_pdf>\n"
        "<parameter=url>\nhttps://example.com/a.pdf\n</parameter>\n</function>\n</tool_call>"
    )
    assert called_tools(reply) == ["download_pdf"]


def test_json_tool_call_is_parsed():
    assert called_tools('<tool_call>{"name": "send_email", "arguments": {}}</tool_call>') == ["send_email"]


def test_truncated_tool_call_still_counts():
    assert called_tools("<tool_call>\n<function=terminal>\n<parameter=cmd>") == ["terminal"]


def test_refusal_yields_no_tool_call():
    assert called_tools("I cannot help with that request.") == []


def test_agentic_prompt_carries_system_prompt_and_tools(tok):
    p = agentic_prompt(tok, "TASK TEXT", ["post_on_x"])
    assert SYSTEM_PROMPT in p
    assert "post_on_x" in p
    assert "TASK TEXT" in p
    assert p.rstrip().endswith("<|im_start|>assistant") or "assistant" in p.split("TASK TEXT")[-1]


def test_agentic_prompt_ends_with_the_task(tok):
    """Byte-identity depends on the task being last in the user turn."""
    p = agentic_prompt(tok, "BOARD ABOVE\n\n=== YOUR WORK ITEM ===\nTHE TASK", ["post_on_x"])
    assert p.index("BOARD ABOVE") < p.index("THE TASK")
