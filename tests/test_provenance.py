import pytest

from pressure.provenance import assert_same_model


def test_passes_when_all_stamped_sources_agree():
    assert_same_model({"a": {"model": "Qwen/Qwen3.5-4B"}, "b": {"model": "Qwen/Qwen3.5-4B"}})


def test_raises_on_a_real_mismatch():
    with pytest.raises(SystemExit):
        assert_same_model({"a": {"model": "Qwen/Qwen3.5-4B"}, "b": {"model": "Qwen/Qwen3.5-9B"}})


def test_ignores_missing_and_unstamped_sources(capsys):
    """None (file absent) and a dict with no `model` key are both silently skipped --
    the artefact predates the stamp and cannot be checked, which is reported, not
    treated as a mismatch."""
    assert_same_model({"a": {"model": "Qwen/Qwen3.5-4B"}, "b": None, "c": {"no_model_key": 1}})
    assert "c" in capsys.readouterr().out
