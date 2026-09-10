"""Run paired E2E evaluation for the SFT v2.7 three-task scope."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import eval_sft_v2_4_e2e as runner  # noqa: E402
from scripts.evaluation import eval_sft_v2_trusted as trusted  # noqa: E402
from scripts.evaluation import sft_v2_7_core_protocol as protocol  # noqa: E402
from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402

runner.MANIFEST_VERSION = "sft_v2.7_core_e2e.v1"
runner.CANDIDATE_REQUEST_NAME = "sft-v2.7-core-candidate"
runner.DEFAULT_ADAPTER = PROJECT_ROOT / "saves/qwen3-8b/lora/sft-v2.7-core-selected"
runner.DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/sft_v2_7_core_trusted_regression.json"
runner.DEFAULT_OUTPUT = PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_trusted_regression_e2e.json"
runner.TARGET_TASKS = CORE_TASKS
trusted.TARGET_TASKS = CORE_TASKS
runner.render_prompt = protocol.render_prompt
runner.score_answer = protocol.score_answer
runner.prompt_contract = lambda _path="": protocol.prompt_contract(str(Path(__file__).resolve()))


if __name__ == "__main__":
    raise SystemExit(runner.main())
