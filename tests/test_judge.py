"""Judge decline tracking. A declined verdict must be reported, never pooled as
a failure -- see judge.any_decline_this_row's docstring for the bug this guards
against (adversarial review, 2026-09-04): the original per-criterion key match
could never succeed, so `unscored` was always empty and every decline was
silently scored as non-compliance."""

from __future__ import annotations

import pressure.judge as judge


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None


class _FakeChatCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeChatCompletions(content)


class _FakeClient:
    """Every attempt (primary + fallbacks) returns the same never-pass-or-fail
    content, forcing the decline path regardless of PRESSURE_JUDGE_FALLBACKS."""

    def __init__(self, content, **kwargs):
        self.chat = _FakeChat(content)


async def _run_judge(monkeypatch, content: str, prompt: str) -> str:
    monkeypatch.setattr(judge, "_api_key", lambda: "dummy")
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeClient(content))
    return await judge.semantic_judge(prompt, "some model output")


def test_decline_sets_the_row_flag(monkeypatch):
    judge.set_current_grader("test_grader_a")
    assert judge.any_decline_this_row() is False

    import asyncio
    result = asyncio.run(_run_judge(monkeypatch, "neither pass nor fail", "unique-prompt-a"))

    assert result == judge.UNSCORED
    assert judge.any_decline_this_row() is True


def test_set_current_grader_resets_the_flag_for_the_next_row(monkeypatch):
    judge.set_current_grader("test_grader_b")
    import asyncio
    asyncio.run(_run_judge(monkeypatch, "neither pass nor fail", "unique-prompt-b"))
    assert judge.any_decline_this_row() is True

    judge.set_current_grader("test_grader_c")
    assert judge.any_decline_this_row() is False


def test_a_real_verdict_does_not_set_the_flag(monkeypatch):
    judge.set_current_grader("test_grader_d")
    import asyncio
    result = asyncio.run(_run_judge(monkeypatch, "pass", "unique-prompt-d"))

    assert result == "pass"
    assert judge.any_decline_this_row() is False
