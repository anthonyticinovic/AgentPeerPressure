"""Multi-turn tool loop against AgentHarm's own tools.

Why this exists: with a single turn the model can only ever emit its *first* tool
call, and for 48 of 56 pilot items that first target function is preparatory
(`download_pdf`, `query_google_search`). Every compliance metric built on one turn
therefore scores "started a benign step" as "did the harmful thing". The loop is a
precondition for measurement, not a refinement of it.

Tools are the vendored ones, executed for real. They return canned strings, so the
environment is deterministic and nothing external is contacted.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .tools import TOOLS_DIR, all_tools

VENDOR_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(VENDOR_ROOT))

END_OF_TURN = "<|im_end|>"
_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FN = re.compile(r"<function=([\w.]+)\s*>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([\w.]+)\s*>(.*?)(?:</parameter>|$)", re.S)


@dataclass
class Turn:
    text: str
    calls: list[dict] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    truncated: bool = False
    cut_mid_call: bool = False


def parse_calls(reply: str) -> list[dict]:
    """Qwen3.5's nested XML form -> [{"function": name, "arguments": {...}}].

    Argument values are typed from the tool's own schema: a `list[str]` parameter
    handed a bare string would raise inside the vendored tool.
    """
    specs, out = all_tools(), []
    for block in _CALL.findall(reply):
        if not re.search(r"</function>|<parameter=", block):
            continue  # prose quoting the syntax is not a call
        for name, body in _FN.findall(block):
            if name not in specs:
                continue
            props = specs[name].as_schema()["function"]["parameters"]["properties"]
            args: dict = {}
            for k, v in _PARAM.findall(body):
                v = v.strip()
                kind = props.get(k, {}).get("type", "string")
                if kind == "array":
                    try:  # models emit JSON arrays as often as bare comma lists
                        parsed = json.loads(v) if v else []
                        args[k] = parsed if isinstance(parsed, list) else [str(parsed)]
                    except json.JSONDecodeError:
                        args[k] = [x.strip() for x in v.split(",") if x.strip()] if v else []
                elif kind == "integer":
                    args[k] = int(m.group()) if (m := re.search(r"-?\d+", v)) else 0
                elif kind == "number":
                    args[k] = float(m.group()) if (m := re.search(r"-?\d+\.?\d*", v)) else 0.0
                elif kind == "boolean":
                    args[k] = v.lower() in {"true", "1", "yes"}
                else:
                    args[k] = v
            for k in props:  # vendored tools take every parameter positionally
                args.setdefault(k, [] if props[k]["type"] == "array" else "")
            out.append({"function": name, "arguments": args})
    return out


def execute(call: dict) -> str:
    """Run the vendored tool. Errors are returned to the model, as a real harness would.

    Several tools read fixture files by a path relative to the benchmark root
    ("benchmark/harmful_tools/content/..."), so execution happens with the working
    directory set to the vendor root, where that path resolves via a symlink.
    """
    module = __import__(f"harmful_tools.{call['function']}", fromlist=[call["function"]])
    runner = getattr(module, call["function"])()
    cwd = os.getcwd()
    try:
        os.chdir(VENDOR_ROOT)
        return str(asyncio.run(runner(**call["arguments"])))
    except Exception as exc:  # surfaced to the model, not swallowed
        return f"Error: {type(exc).__name__}: {exc}"
    finally:
        os.chdir(cwd)


def _generate_text(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, bool]:
    """One greedy completion. Split out so the loop is testable without a model."""
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    text = raw.split(END_OF_TURN)[0].replace("<think>", "").replace("</think>", "").strip()
    return text, END_OF_TURN not in raw


def run_loop(model, tok, messages: list[dict], tools: list[dict], *,
             max_turns: int = 6, max_new_tokens: int = 384,
             on_prompt=None) -> list[Turn]:
    """Generate, execute tool calls, feed results back, until the model stops calling.

    `on_prompt(prompt)` fires once per turn with the rendered prompt, before
    generation, so a caller can read the residual stream without this module knowing
    what a direction is.
    """
    turns: list[Turn] = []
    convo = list(messages)
    for _ in range(max_turns):
        prompt = tok.apply_chat_template(
            convo, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if on_prompt is not None:
            on_prompt(prompt)
        text, truncated = _generate_text(model, tok, prompt, max_new_tokens)
        turn = Turn(text=text, truncated=truncated)
        turn.calls = parse_calls(text)
        # A truncated turn whose final block never closed was cut mid-call. Executing
        # it sends a half-written payload and feeds a success string back; several
        # content-heavy graders became structurally uncompletable this way.
        if turn.truncated and "<tool_call>" in text and not text.rstrip().endswith("</tool_call>"):
            turn.cut_mid_call = True
            if turn.calls:
                turn.calls = turn.calls[:-1]
        turn.cut_mid_call = turn.cut_mid_call or (turn.truncated and not turn.calls and "<tool_call>" in text)
        turns.append(turn)
        if not turn.calls:
            break
        convo.append({"role": "assistant", "content": text})
        for call in turn.calls:
            result = execute(call)
            turn.results.append(result)
            convo.append({"role": "tool", "content": result, "name": call["function"]})
    return turns


def as_inspect_messages(turns: list[Turn]):
    """Convert to inspect_ai ChatMessages so AgentHarm's own graders can read them."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall

    msgs = []
    for i, turn in enumerate(turns):
        calls = [
            ToolCall(id=f"{i}-{j}", function=c["function"], arguments=c["arguments"])
            for j, c in enumerate(turn.calls)
        ]
        msgs.append(ChatMessageAssistant(content=turn.text, tool_calls=calls or None))
        for j, (c, res) in enumerate(zip(turn.calls, turn.results)):
            msgs.append(ChatMessageTool(content=res, tool_call_id=f"{i}-{j}", function=c["function"]))
    return msgs
