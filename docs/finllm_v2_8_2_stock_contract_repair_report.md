# finLLM v2.8.2 Stock Contract Repair Report

Date: 2026-09-09 (Asia/Shanghai)

## Decision

**FAIL. Reject `sft-v2.8.2-stock-contract-answer` as a candidate.** Retain the
v2.7-core selected adapter as the last viable model. Do not start DPO/GRPO and
do not continue training from the rejected v2.8.2 adapter.

This is a seen-contract preflight, not a release result: its 50 documents
overlap the trusted regression. It is nevertheless a valid paired regression
test because the pre/post runs use the same 50 cases, model base, decoding
parameters, seed, strict scorer, and selector policy.

## Change

The one-epoch repair continued from `sft-v2.7-core-selected` on 810 balanced
core rows (QA/quant/stock: 270 each). All 270 stock targets were rewritten:

- 166 rows with visible MA5/MA20 received price, MA trend, risk, financial, and
  no-forecast atomic lines.
- 104 rows without MA evidence received a cited trend-evidence abstention.
- Each of 279 stock train/eval targets passed the v2.8.2 task validator; train
  source overlap with the preflight gold was zero.

The missing-trend path is fail-closed: it applies only where both MA5 and MA20
are absent, still requires a cited abstention, and does not relax citation,
numeric-copy, risk, financial-anchor, or forecast rules.

Training completed normally: 51 steps, one epoch, `1.5e-6` learning rate,
final teacher-forced eval loss `0.0401`, no NaN/Inf/OOM. That loss was not used
as evidence of E2E quality.

## Paired Preflight

| Metric | v2.7-core | v2.8.2 | Delta |
|---|---:|---:|---:|
| E2E@1 | 0.16 | 0.08 | -0.08 |
| E2E@8 / oracle@8 | 0.68 | 0.28 | -0.40 |
| Audit@1 | 0.48 | 0.38 | -0.10 |
| Audit@8 | 0.94 | 0.80 | -0.14 |
| Mean primary @1 | 0.2946 | 0.1678 | -0.1268 |
| Paired primary delta @1 | -0.0414 | -0.1682 | -0.1268 |
| Clustered primary 95% CI | [-0.1817, 0.0973] | [-0.2943, -0.0482] | inferior |
| Sampled length finishes | 13 | 9 | not the cause |

All conditions are below the pre-registered thresholds. The expected nonzero
exit status of E2E is therefore a release-gate result, not an inference job
failure.

## Selector And Confidence

| Metric | v2.7-core | v2.8.2 |
|---|---:|---:|
| Candidate-pool TP / FP / TN / FN | 81 / 0 / 319 / 0 | 37 / 0 / 363 / 0 |
| Selector false acceptance | 0 | 0 |
| Selector oracle recovery | 1.00 | 1.00 |
| Selector E2E@8 | 0.68 | 0.28 |

The validator and selector remain calibrated on this evidence-complete stock
contract: all scorer-correct candidates were accepted and no scorer-wrong
candidate was accepted. The failure is candidate-pool quality, not validator
strictness. The selector cannot recover candidates the model does not produce.

Sequence NLL coverage was 100%. Correct greedy outputs had lower mean NLL than
incorrect outputs (`0.0276` vs `0.0904`), but wrong accepted outputs still had
low NLL (`0.0298`); NLL is descriptive only and must not weaken hard gates.

## Failure Analysis

Among 363 rejected sampled candidates, categories overlap:

| Failure category | Candidates | Share |
|---|---:|---:|
| Missing price/return anchor | 284 | 78.24% |
| Missing MA/trend anchor | 251 | 69.15% |
| Missing financial anchor | 228 | 62.81% |
| Unsupported claim | 207 | 57.02% |
| Missing risk anchor | 181 | 49.86% |
| Citation incomplete | 119 | 32.78% |
| Numeric mismatch | 113 | 31.13% |

Representative regression: v2.8.2 generated `收益率或收益为-11.9483%` and
`MA5上穿MA20`, even though the evidence exposes exact price, MA5, and MA20
values. Both lines are unsupported transformations under the contract. This is
not a length issue and cannot be repaired by lowering selector thresholds.

## Next Experiment

Do not repeat the full 270-row target rewrite or train from v2.8.2. Before a
new SFT run, construct a source-disjoint stock repair ablation from
`sft-v2.7-core-selected`:

1. Keep the existing audited stock targets as the control surface.
2. Add only complete-evidence MA rows as a small, separately tagged repair
   slice; do not teach missing-MA abstentions in the same first ablation.
3. Train at least a control and a low-mass repair mixture, then use a small
   source-disjoint Dev-Audit to choose a checkpoint before 50-case @8 runs.
4. Require no regression in price/MA literal-copy rate, audit pass, or primary
   before running the complete preflight; continue to prohibit RL.

Artifacts: [repair build report](../data/sft_v2_8_2_stock_contract/build_report.json),
[training output](../saves/qwen3-8b/lora/sft-v2.8.2-stock-contract-answer),
[preflight E2E](../saves/eval_results/sft_v2_8_2_stock_contract_seen_preflight_e2e.json),
[selector replay](../saves/eval_results/sft_v2_8_2_stock_contract_seen_preflight_task_selector.json),
and [diagnostic](../saves/eval_results/sft_v2_8_2_stock_contract_seen_preflight_diagnostic.json).
