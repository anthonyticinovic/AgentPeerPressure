"""Everything that can only fail on the cluster, checked in one cheap pass.

`device.py` has never executed its CUDA branch. This script exercises it, plus the
handful of environment assumptions that hold silently on the laptop and break on a
compute node: no outbound network, a shared read-only filesystem, an older GPU
without bf16, an attention backend that differs from the one the 4B run used.

Run it on the compute node, inside the job, before the real run. Exit code is
non-zero if any check fails, so an sbatch script can gate on it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from pressure import device as dv  # noqa: E402
from pressure.boards import assert_speaker_free, load_boards  # noqa: E402
from pressure.config import CFG  # noqa: E402
from pressure.data import agentharm  # noqa: E402
from pressure.loop import VENDOR_ROOT, as_inspect_messages, run_loop  # noqa: E402
from pressure.model import load_model  # noqa: E402
from pressure.tools import SYSTEM_PROMPT, schemas_for  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok and fatal:
        FAILURES.append(name)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", action="store_true", help="check the 4B instead of the 9B")
    ap.add_argument("--skip-model", action="store_true", help="environment checks only")
    args = ap.parse_args()

    print("device")
    d = dv.resolve_device()
    check("CUDA is the resolved device", d == "cuda", dv.describe().get("gpu", d))
    if d == "cuda":
        major, minor = torch.cuda.get_device_capability()
        # sm80+ has native bf16. Below that resolve_dtype falls back to fp16, which
        # overflows on some Qwen activations and would not match the 4B bf16 run.
        check(f"compute capability >= 8.0 for native bf16 (got {major}.{minor})", major >= 8)
    check("dtype is bfloat16", dv.resolve_dtype(d) is torch.bfloat16)
    # The 4B run used sdpa. flash-attention-2 is numerically close but not identical,
    # and a scale comparison must not confound model size with attention kernel.
    check("attention backend is sdpa, as on the 4B run", dv.attn_implementation(d) == "sdpa",
          "uninstall flash_attn if this fails")

    need = 18.0 * 1.25
    have = dv.available_memory_gb(d)
    check(f"{have:.0f} GB free covers the 9B's ~{need:.0f} GB", have >= need)

    print("filesystem")
    link = VENDOR_ROOT / "benchmark" / "harmful_tools"
    check("benchmark/harmful_tools symlink resolves", link.is_dir(),
          "several tools read fixtures by a path relative to the benchmark root")
    probe = VENDOR_ROOT / ".preflight_write_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        writable = True
    except OSError as exc:
        writable = False
        print(f"        {exc}")
    check("vendor root is writable", writable, "edit_txt and read_file write there")
    check("results dir is writable", os.access(CFG.results_dir, os.W_OK))

    print("corpus and stimuli")
    items = agentharm(harmful=True)
    check("208 harmful items load", len(items) == 208, f"got {len(items)}")
    boards = load_boards()
    for b in boards:
        assert_speaker_free(b)
    check("boards load and are speaker-free", len(boards) > 0, f"{len(boards)} boards")
    check("tool schemas parse", len(schemas_for(items[0]["target_functions"])) > 0)

    if args.skip_model or FAILURES:
        return _exit()

    print("model and loop (this is the slow part)")
    hf_offline = os.getenv("HF_HUB_OFFLINE") == "1"
    print(f"        HF_HUB_OFFLINE={'1 — weights must already be cached' if hf_offline else 'unset'}")
    model, tok = load_model(eval_model=not args.iter)
    check("weights loaded", True, f"{CFG.eval_model if not args.iter else CFG.iter_model}")
    check("32 decoder layers", len(model.model.layers) == 32, str(len(model.model.layers)))

    # One real item through the real loop: generation, XML tool-call parsing, tool
    # execution against the fixtures, and the transcript conversion the graders read.
    it = next(i for i in items if i["target_functions"])
    turns = run_loop(
        model, tok,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it["prompt"]}],
        schemas_for(it["target_functions"]), max_turns=2, max_new_tokens=128,
    )
    check("loop produced a turn", len(turns) > 0)
    check("generation is non-empty", any(t.text.strip() for t in turns))
    check("transcript converts for the graders", len(as_inspect_messages(turns)) > 0)
    _exit()


def _exit() -> None:
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
