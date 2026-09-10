"""Select SFT v2.6 by independent Dev-Audit generation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import select_sft_v2_4_checkpoint as runner  # noqa: E402
from scripts.evaluation import sft_v2_6_protocol as protocol  # noqa: E402

runner.MANIFEST_VERSION = "dev_audit_predictions.v2.6"
runner.SELECTION_VERSION = "dev_audit_checkpoint_selection.v2.6"
runner.render_prompt = protocol.render_prompt
runner.score_answer = protocol.score_answer
runner.prompt_contract = lambda _path="": protocol.prompt_contract(str(Path(__file__).resolve()))

if __name__ == "__main__":
    raise SystemExit(runner.main())
