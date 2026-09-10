# finLLM v2.8.1 Stock Contract Preflight

Date: 2026-09-09 (Asia/Shanghai)

## Objective

This phase implements the next v2.8 gate before training or RL: distinguish a
stock model failure from a mismatch between the old pattern scorer and the
evidence-anchor acceptance contract. It does not change the v2.7 validator and
does not claim new model quality.

## Delivered Evaluation Contract

- `50` stock cases from `25` issuers held out from training, with one normal
  and one adversarial instruction-conflict case per issuer.
- Every case exposes a market document with price, MA5, MA20, and risk data,
  plus a financial document with visible fundamentals.
- A candidate must cite price/return, MA trend, risk, and a financial anchor;
  uncited atomic claims and unsupported forward predictions remain rejected.
- The strict scorer combines normal audit scoring with this explicit contract.
  It is intentionally not the selector: model prompts see requirements, not
  hidden patterns or gold scores.

This is explicitly a **seen contract preflight**, not an untouched final
holdout: all 50 evidence documents overlap the existing trusted regression.
It tests the scorer/protocol implementation only and cannot support a release,
generalization, or RL decision.

## Preflight Results

| Check | Result |
|---|---:|
| Cases | 50 |
| Heldout issuers/source groups | 25 / 25 |
| Adversarial cases | 25 |
| Train/heldout issuer overlap | 0 |
| Seen trusted-regression document overlap | 50 / 50 |
| Dataset schema validation | pass |
| Stock-only E2E validate-only | pass |
| Complete cited reference answer | pass |
| Missing MA/trend anchor | reject |

## Confidence Collection

The E2E runner has a new opt-in `--collect-token-logprobs` flag. It stores
selected-token sequence logprob, normalized logprob, normalized NLL, and token
count for greedy and all eight sampled candidates. The calibration diagnostic
then bins NLL by correct/incorrect and accept/reject after selector replay. It
deliberately does not call a top-k approximation “entropy”. Mean token entropy
remains unavailable unless a future backend exports an adequate probability
distribution.

## Next Execution

Run the seen stock-contract protocol with fixed seed and likelihood capture:

```bash
python scripts/evaluation/eval_sft_v2_8_stock_contract_e2e.py \
  --train-file data/sft_v2_7_core/fin_agentic_sft_v2_7_core_answer_train.json \
  --collect-token-logprobs \
  --no-resume
```

Only after scoring the generated outputs should the team revisit validator
calibration. A passing source split does not authorize DPO/GRPO.

The current Python environment has an RTX 5090 but no `vllm` package, so this
command has not been run here. Installing or selecting a compatible inference
environment is a prerequisite, not evidence of an evaluation failure.
