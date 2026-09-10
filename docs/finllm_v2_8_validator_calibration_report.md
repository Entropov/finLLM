# finLLM v2.8 Validator Calibration Report

Date: 2026-09-09 (Asia/Shanghai)

Scope: `financial_qa`, `quant_strategy`, and `stock_analysis` only. No model training, DPO, GRPO, or production deployment was run in this phase.

## Decision

**C. FAIL.** The validator is materially safer than the v2.7 audit-only selector, but it is not qualified as a reward or deployment selector. The remaining blocker is adversarial `stock_analysis`: candidate-pool false acceptance is zero, but correct-candidate rejection is too high and the gold task contract is weaker than the product stock contract. RL remains prohibited.

## Inputs And Boundaries

- Trusted/adversarial results are 150-sample, three-task slices of seen regressions, not an untouched final test.
- Correctness below means the existing hidden gold scorer's `passed` field. It is used only for offline diagnosis, never for selector features.
- The task-aware selector reads only visible task type/evidence, candidate text, and audit-verifier fields. It forbids gold `scoring`, task score, primary score, and `passed`.
- `financial_report`, `risk_assessment`, and `sentiment_analysis` are out of the v2.7/v2.8 validator scope. No acceptance metric is claimed for them.

## Validator Confusion Matrix

Candidate-pool values score all eight sampled candidates per case. Selector values score the one selected candidate, or treat a rejected case with an oracle-valid candidate as a false negative.

| Dataset / layer | Task | TP | FP | TN | FN | Precision | Recall | FA rate | FR rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trusted pool | QA | 125 | 0 | 272 | 3 | 1.0000 | 0.9766 | 0.0000 | 0.0234 |
| trusted pool | stock | 84 | 0 | 220 | 96 | 1.0000 | 0.4667 | 0.0000 | 0.5333 |
| trusted pool | quant | 341 | 0 | 59 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| trusted pool | overall | 550 | 0 | 551 | 99 | 1.0000 | 0.8475 | 0.0000 | 0.1525 |
| adversarial pool | QA | 258 | 0 | 125 | 17 | 1.0000 | 0.9382 | 0.0000 | 0.0618 |
| adversarial pool | stock | 5 | 0 | 321 | 74 | 1.0000 | 0.0633 | 0.0000 | 0.9367 |
| adversarial pool | quant | 337 | 0 | 63 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial pool | overall | 600 | 0 | 509 | 91 | 1.0000 | 0.8683 | 0.0000 | 0.1317 |

The corresponding selected-policy overall recall is `0.8872` on trusted and `0.8205` on adversarial; false acceptance remains `0`. The full machine-readable matrix, including task buckets and the explicit out-of-scope declaration for other tasks, is [validator_confusion_matrix.json](../saves/eval_results/sft_v2_8_validator_calibration/validator_confusion_matrix.json).

## Preference Calibration

The old 810-pair preference set had chosen accept `0.7864`, rejected false accept `0/810`, and 173 rejected chosen examples. Full error mining is in [rejected_chosen_by_reason.csv](../saves/eval_results/sft_v2_8_validator_calibration/rejected_chosen_by_reason.csv).

| Taxonomy | Count | Share | Ownership | Action |
|---|---:|---:|---|---|
| data annotation issue: QA prompt has no visible evidence | 69 | 39.88% | data | remove from verifier-positive preference; retain only for refusal data |
| stock-specific rule / data-contract mismatch: chosen omits MA/trend anchor | 104 | 60.12% | data contract | rebuild stock positives with a cited trend anchor |

This accounts for all `173/810` rejected chosen examples. They are not evidence that citation-drop should be relaxed.

A new 637-pair v2.8 preference set was built only from evidence-complete rows: QA `201`, quant `270`, stock `166`. Its independent check reports chosen accept `1.0000`, rejected false accept `0/637`, and zero chosen false rejects. It intentionally excludes the 69 evidence-empty QA rows and 104 MA-empty stock rows instead of relabeling them as valid positives.

Artifacts: [v2.8 preference](../data/rlhf/sft_v2_8_core_validator_preference.json), [build report](../data/rlhf/sft_v2_8_core_validator_preference_report.json), and [independent validator check](../saves/eval_results/sft_v2_8_validator_preference_check.json).

## Selector Before And After

| Dataset | Policy | E2E@8 | False acceptance | Oracle recovery | E2E delta vs greedy |
|---|---|---:|---:|---:|---:|
| trusted | audit-only | 0.6067 | 0.3681 | 0.6741 | +0.0334 |
| trusted | evidence-anchor task-aware | 0.7867 | 0.0000 | 0.8741 | +0.2133 |
| adversarial | audit-only | 0.6400 | 0.3425 | 0.7442 | +0.0467 |
| adversarial | evidence-anchor task-aware | 0.6400 | 0.0000 | 0.7442 | +0.0467 |

The task-aware policy preserves the citation-drop defence and removes observed false acceptance on seen regression. This is not sufficient for release because stock recall collapses under adversarial prompts.

## Stock Analysis Blocker

| Metric | trusted stock | adversarial stock |
|---|---:|---:|
| Raw correctness @1 | 0.5200 | 0.1400 |
| Raw correctness @8 | 0.9200 | 0.7000 |
| Candidate-pool validator accept rate | 0.2100 | 0.0125 |
| Correct + rejected rate | 0.5333 | 0.9367 |
| Wrong + accepted rate | 0.0000 | 0.0000 |
| Selector E2E@8 | 0.5800 | 0.0400 |

Adversarial stock is not Case A alone. Its low raw @1 and high raw @8 show generation instability, but `74` gold-correct candidates are rejected and `70` of those lack the MA/trend anchor. `32` also lack the selected financial anchor. Sample inspection found answers such as cited price/volatility-only summaries: they pass the older pattern scorer at 2/3 coverage but do not supply the trend and fundamentals requested by the stock product contract.

The evidence supports Cases B, C, and D:

1. **B, validator calibration:** the validator rejects many candidates deemed correct by the existing scorer.
2. **C, protocol incompatibility:** the old scorer can pass answers that omit required stock-analysis dimensions.
3. **D, adversarial concentration:** the mismatch is much stronger on adversarial stock than on trusted stock.

The correct response is not to lower the stock threshold. The next stock scorer and adversarial gold must encode the same cited price/return, trend, risk, and financial-anchor contract as the validator, followed by a new source-disjoint evaluation.

## E2E And Primary Deltas

Raw model evaluation remains unchanged because v2.8 did not train the model.

| Dataset | E2E@1 | E2E@8 | QA primary delta | stock primary delta |
|---|---:|---:|---:|---:|
| trusted | 0.5733 | 0.9000 | -0.0466 | -0.1318 |
| adversarial | 0.5933 | 0.8600 | +0.1185 | -0.2993 |

Quant primary deltas are `+0.4197/+0.4250`. The negative stock deltas and the validator/score mismatch independently block an RL decision.

## Entropy And Confidence

[entropy_confidence_diagnostic.json](../saves/eval_results/sft_v2_8_validator_calibration/entropy_confidence_diagnostic.json) records **unavailable** for mean token entropy, sequence-normalized logprob/NLL, and all correctness/acceptance subgroups. The v2.7 stored vLLM generations contain no token logprob or entropy fields. No length proxy was substituted for uncertainty.

The available pass@1/pass@8 gaps do show instability: stock trusted `0.52 -> 0.92` and stock adversarial `0.14 -> 0.70`. A later fixed-seed evaluation must persist token logprobs or sequence NLL before entropy claims can be made. Low entropy must not be interpreted as correctness because high-confidence errors are a target failure mode.

## Gates And Next Step

| Gate | Result | Status |
|---|---:|---|
| v2.8 aligned preference chosen accept | 1.0000 | pass for the rebuilt data only |
| v2.8 aligned preference rejected false accept | 0/637 | pass for the rebuilt data only |
| legacy preference chosen accept | 0.7864 | fail; diagnosed as contract mismatch |
| seen trusted/adversarial selector false accept | 0 / 0 | pass, not final-test evidence |
| adversarial stock correct-candidate recall | 0.0633 candidate pool | fail |
| adversarial stock selector E2E@8 | 0.0400 | fail |
| untouched final holdout | absent | pending |

Next: build contract-aligned stock gold and adversarial cases, add token-logprob collection, and rerun the same matrix. Do not run DPO/GRPO until the new stock evaluation has low false acceptance, high chosen acceptance, and stock recall no worse than greedy under the agreed task contract.
