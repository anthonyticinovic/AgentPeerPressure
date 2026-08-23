"""Regenerate results/grader_content_split.json.

Which graders retain a content check when the semantic judge is disabled. Only
meaningful for `--no-judge` runs; with the judge live every grader verifies its own
content and the split selects exactly the graders the judge cannot affect.

Committed as a script because the file is an input to 14_contrast_audit.py and
results/ is gitignored — without this the audit cannot run from a clean clone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressure.config import CFG  # noqa: E402
from pressure.grading import content_verified, judge_dependent  # noqa: E402

out = CFG.results_dir / "grader_content_split.json"
graders = judge_dependent()
split = {
    "primary": sorted(g for g in graders if content_verified(g)),
    "content_unverified": sorted(g for g in graders if not content_verified(g)),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(split, indent=1))
print(f"{len(split['primary'])} primary, {len(split['content_unverified'])} content-unverified -> {out}")
