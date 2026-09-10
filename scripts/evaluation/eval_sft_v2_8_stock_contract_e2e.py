"""Run the v2.8 stock-only contract preflight E2E protocol."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import eval_sft_v2_4_e2e as runner  # noqa: E402
from scripts.evaluation import eval_sft_v2_trusted as trusted  # noqa: E402
from scripts.evaluation import sft_v2_8_stock_contract as protocol  # noqa: E402


STOCK_TASK = {"stock_analysis"}
runner.MANIFEST_VERSION = "sft_v2.8_stock_contract_seen_preflight_e2e.v1"
runner.CANDIDATE_REQUEST_NAME = "sft-v2.8-stock-contract-preflight-candidate"
runner.DEFAULT_ADAPTER = PROJECT_ROOT / "saves/qwen3-8b/lora/sft-v2.7-core-selected"
runner.DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/sft_v2_8_stock_contract_seen_preflight.json"
runner.DEFAULT_OUTPUT = PROJECT_ROOT / "saves/eval_results/sft_v2_8_stock_contract_seen_preflight_e2e.json"
runner.TARGET_TASKS = STOCK_TASK
runner.EXPECTED_TASKS = STOCK_TASK
runner.MIN_SAMPLES_PER_TASK = 50
trusted.TARGET_TASKS = STOCK_TASK
runner.render_prompt = protocol.render_prompt
runner.score_answer = protocol.score_answer
runner.prompt_contract = lambda _path="": protocol.prompt_contract(str(Path(__file__).resolve()))


if __name__ == "__main__":
    raise SystemExit(runner.main())
