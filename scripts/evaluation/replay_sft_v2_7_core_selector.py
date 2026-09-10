"""Replay the audit-only selector over stored SFT v2.7 core generations."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import replay_sft_v2_5_selector as runner  # noqa: E402

runner.MANIFEST_VERSION = "sft_v2.7_core_selector_replay.v1"


if __name__ == "__main__":
    raise SystemExit(runner.main())
