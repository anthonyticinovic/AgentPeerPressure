"""Corpora. Extraction and evaluation are deliberately different datasets.

Extraction: AdvBench harmful vs Alpaca harmless, plain chat format.
Evaluation: AgentHarm, whose benign counterparts have equivalent tool use — that
matching is what isolates alignment refusal from capability failure.

Keeping them structurally separate removes the leakage worry by construction rather
than by row-splitting. `assert_no_leakage` still checks it, because "structurally
impossible" has a way of not being.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

from .config import CFG

# Canonical AdvBench is gate-on-click. The mirror is a 520-row copy used only if the
# canonical source is unavailable; which one was used is recorded and reported.
ADVBENCH_CANONICAL = ("walledai/AdvBench", "prompt")
ADVBENCH_MIRROR = ("kelly8tom/advbench_orig", "Behavior")

AGENTHARM = "ai-safety-institute/AgentHarm"
AGENTHARM_SPLITS = ("test_public", "validation")


@dataclass(frozen=True)
class Corpus:
    """Prompts plus the provenance needed to state honestly where they came from."""

    prompts: list[str]
    source: str

    def __len__(self) -> int:
        return len(self.prompts)


def harmful_prompts() -> Corpus:
    """AdvBench harmful behaviours, canonical source preferred."""
    for repo, col in (ADVBENCH_CANONICAL, ADVBENCH_MIRROR):
        try:
            d = load_dataset(repo, split="train")
        except Exception:
            continue
        return Corpus([r[col] for r in d], repo)
    raise RuntimeError(
        f"Neither {ADVBENCH_CANONICAL[0]} nor {ADVBENCH_MIRROR[0]} could be loaded. "
        f"Accept the gate at https://huggingface.co/datasets/{ADVBENCH_CANONICAL[0]}"
    )


def harmless_prompts() -> Corpus:
    """Alpaca instructions with no `input` field, so they stand alone as prompts."""
    d = load_dataset("tatsu-lab/alpaca", split="train")
    return Corpus([r["instruction"] for r in d if not r["input"]], "tatsu-lab/alpaca")


def extraction_corpus() -> tuple[Corpus, Corpus]:
    """Deterministically shuffled harmful/harmless prompts, trimmed to a common size.

    AdvBench has 520 rows against the plan's 400 extraction + 100 selection, so the
    harmful side is the binding constraint and both sides are cut to match it.
    """
    harmful, harmless = harmful_prompts(), harmless_prompts()
    rng = random.Random(CFG.seed)
    h, b = list(harmful.prompts), list(harmless.prompts)
    rng.shuffle(h)
    rng.shuffle(b)
    n = min(CFG.n_extract_pairs + CFG.n_select_pairs, len(h), len(b))
    return Corpus(h[:n], harmful.source), Corpus(b[:n], harmless.source)


def split_extract_select(corpus: Corpus) -> tuple[Corpus, Corpus]:
    """Extraction slice builds the direction; selection slice picks (l*, p*) and tau."""
    k = CFG.n_extract_pairs
    return Corpus(corpus.prompts[:k], corpus.source), Corpus(corpus.prompts[k:], corpus.source)


def agentharm(harmful: bool = True, splits: tuple[str, ...] = AGENTHARM_SPLITS) -> list[dict]:
    """Evaluation items.

    The public release exposes 44 base behaviours in test_public and 8 in validation,
    each with 4 prompt variants (detailed_prompt x hint_included) — 208 items, not the
    110 the plan assumed. `id_original` is the cluster key: variants of one behaviour
    are not independent, so every interval is computed clustered on it.
    """
    config = "harmful" if harmful else "harmless_benign"
    items: list[dict] = []
    for split in splits:
        for r in load_dataset(AGENTHARM, config, split=split):
            items.append(
                {
                    "id": r["id"],
                    "cluster": r["id_original"],
                    "split": split,
                    "prompt": r["prompt"],
                    "name": r["name"],
                    "category": r["category"],
                    "detailed_prompt": r["detailed_prompt"],
                    "hint_included": r["hint_included"],
                    "target_functions": r["target_functions"],
                    "grading_function": r["grading_function"],
                    "harmful": harmful,
                }
            )
    return items


def difficulty(item: dict) -> str:
    """Variant difficulty. `hard` items resist the C2 ceiling effect that would make
    the item-level AUROC range-restricted for a boring statistical reason."""
    if item["detailed_prompt"] and item["hint_included"]:
        return "easy"
    if item["detailed_prompt"] or item["hint_included"]:
        return "medium"
    return "hard"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def assert_no_leakage(extraction: list[str], evaluation: list[str]) -> None:
    """Hard-fail if any extraction string appears in the evaluation set."""
    ext = {_norm(s) for s in extraction}
    overlap = [e for e in evaluation if _norm(e) in ext]
    if overlap:
        raise AssertionError(
            f"{len(overlap)} extraction strings leaked into evaluation: {overlap[:3]}"
        )
