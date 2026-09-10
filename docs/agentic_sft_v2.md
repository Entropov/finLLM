# Agentic SFT v2 and Trusted Evaluation

## Objective

SFT v2 trains Qwen3-8B on concise structured reasoning and auditable agent actions. It does not train unrestricted prose chain-of-thought. The training target contains:

- for complex tasks, a structured plan and decision basis inside `<think>`;
- tool/action arguments and observation Evidence IDs;
- a Claim-to-Evidence map;
- the final user-facing answer with Evidence ID citations.

The production trajectory is finalized only after the actual model completion is available. Prompt-enrichment placeholder answers are not training data.

## Audit Contract

The canonical implementation is `scripts/rag/audit_schema.py`; the interoperable JSON Schema is `data/rag/audit_v2.schema.json`.

An `EvidenceRecord` identifies the canonical source, publisher, source class, reliability tier, publication/effective/fetch times, exact quote, content hash, document version, and chunk offsets. An Evidence ID is stable for the source, quote hash, and offsets.

A `ClaimRecord` classifies each conclusion as fact, calculation, inference, or opinion and records supporting and contradicting Evidence IDs, calculation ID, confidence, applicable time, and assumptions.

The system treats `published_at`, `effective_at`, and `fetched_at` as different concepts. Missing dates remain missing. Future or not-yet-observable evidence fails the reward hard gate.

## Reward Contract

`scripts/rag/reward_v2.py` applies non-compensable gates before scoring:

- fabricated or missing citations;
- irrelevant retrieval or entity mismatch;
- unsupported, polarity-reversed, or numerically inconsistent claims;
- future/not-yet-available evidence;
- invalid evidence metadata and unresolved evidence conflicts;
- prohibited guaranteed-return or all-in investment language;
- invalid quantitative code artifacts;
- missing/incomplete trajectories, missing retrieval plans, or normalized Evidence IDs that do not match the audited evidence.

Only a trajectory that passes every gate receives the weighted evidence/task score. Delayed risk-adjusted market feedback is then capped at `0.05`; it cannot rescue a failed fact or compliance gate.

Run the deterministic attack suite before any data build or training:

```bash
python scripts/evaluation/validate_agentic_rewards.py
```

The release condition is zero false accepts across the checked attacks and zero false rejects for control cases. The current suite contains 14 deterministic cases, including missing trajectories, invalid calculations, and future-publication leakage.

## Collect and Build SFT v2

Rebuild the vector index once so new audit metadata is present:

```bash
python scripts/rag/build_vector_db.py \
  --data-dir data/knowledge/v2_train \
  --db-dir saves/chroma_v2_train \
  --chunk-strategy semantic
```

Run the API in agentic mode and collect real completions. The API now retains the prepared LangGraph state and calls `finalize()` with the actual streamed or non-streamed model answer.

```bash
python scripts/inference/api_server.py \
  --model-path Qwen/Qwen3-8B \
  --enable-rag \
  --rag-mode agentic \
  --rag-agentic-config configs/rag_agentic.yaml \
  --rag-db-dir saves/chroma_v2_train
```

Collect the prepared, unique requests through that API. Failed reward/schema
gates remain failed rows and are not replaced with template answers.

```bash
python scripts/data_processing/collect_agentic_v2_trajectories.py \
  --input data/rag/v2_collection_requests.json \
  --output data/rag/v2_collection_responses.jsonl
```

Build the dataset:

```bash
python scripts/data_processing/build_sft_v2_dataset.py
```

The builder accepts only schema-valid, hard-gate-passing trajectories with reward at least `0.65`. It deduplicates questions, splits by source/document group, records both hashed split groups and auditable source/version identities, checks zero train/eval query overlap, and requires at least 100 accepted trajectories for each of the six tasks.

The current pre-v2 trajectory file is intentionally quarantined. See `data/sft_v2/build_report.json`; it must not be migrated by simply adding fields because its answers and rewards were already shown to be unreliable.

## Train

After the data report has `release_gate_passed: true`:

```bash
bash scripts/training/train_sft_v2.sh --dry-run
bash scripts/training/train_sft_v2.sh --skip-build
```

The v2 configuration continues from the selected v1 LoRA checkpoint, uses the Qwen3 thinking template, an 8192-token context, and a lower learning rate for trajectory specialization. It writes a separate adapter to `saves/qwen3-8b/lora/sft-v2`.

## Trusted Model Evaluation

Gold cases must conform to `data/evaluation/trusted_sft_v2.schema.json`. Curate them from held-out issuers/reports/events, include point-in-time evidence and deterministic scoring assertions, and never derive them from the SFT trajectory pool.

Generate both prediction arms from the same process. The generator gives both
models the fixed evidence embedded in each gold case and never queries the
training vector store. It records exact model/adapter identities, decoding
settings, per-case prompt hashes, and resumable partial results. On vLLM 0.10.2
set `VLLM_USE_FLASHINFER_SAMPLER=0` when FlashInfer sampling is unavailable.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluation/generate_sft_v2_trusted_predictions.py
```

Prediction manifests are mandatory. The evaluator rejects non-identical paired
prompts, mismatched generation settings or base models, missing adapter weight
hashes, incomplete/extra IDs, and a dataset fingerprint mismatch.

```bash
python scripts/evaluation/eval_sft_v2_trusted.py \
  --gold data/evaluation/trusted_finance_v2.json \
  --baseline-predictions saves/eval_results/qwen3_base_trusted_predictions.json \
  --candidate-predictions saves/eval_results/qwen3_sft_v2_trusted_predictions.json \
  --train-file data/sft_v2/fin_agentic_sft_v2_train.json
```

Default release gates require at least 50 unique held-out questions per task, no
detected question or evidence URL/version leakage, no overall or per-task
pass-rate regression, no audit regression, at least 95% candidate audit pass
rate, and a source-group-clustered paired bootstrap 95% lower bound above the
-2 percentage-point non-inferiority margin. Reports include both case-level and
source-group-clustered confidence intervals because multiple questions may use
the same held-out issuer evidence.

This paired model evaluation scores supplied answers against fixed point-in-time evidence and therefore explicitly disables the end-to-end trajectory gate. Agent trajectory integrity is evaluated separately by Reward v2 and is mandatory for SFT v2 data admission.

RL remains disabled until SFT v2 passes these gates and the reward attack suite is expanded using real model failures.

## Measured Run (2026-09-07)

The paired Qwen3-8B base/SFT-v2 run completed on all 300 held-out cases (50
unique questions per task). Both arms used identical prompt hashes, base model,
and deterministic decoding settings. The dataset fingerprint is
`f158d5916431702238725a57ddf938a598c59f1e4758f16d2d084c0ad1fab38d`,
the prediction protocol passed, and no train/evaluation question or evidence
URL/version overlap was detected.

- Base pass rate: `0.0167`; audit pass rate: `0.0200`; mean primary score: `0.5674`.
- SFT-v2 pass rate: `0.3867` (95% CI `[0.3333, 0.4429]`); audit pass rate: `0.4667`; mean primary score: `0.6397`.
- Paired primary-score delta: `+0.0724` (case bootstrap 95% CI `[0.0398, 0.1041]`).
- Source-group clustered 95% CI: `[0.0462, 0.0972]` across 30 groups.
- Pass-state improvements/regressions: `112/1`.
- The only configured release-gate failure was candidate audit pass rate below `0.95`.

SFT-v2 also reduced max-token terminations from `112/300` to `15/300` and
reduced mean output length from `1018.95` to `445.95` tokens. It nevertheless
introduced an unacceptable output-language drift: 25 candidate answers contain
Cyrillic text (7 financial-report and 18 sentiment answers), versus zero base
answers. This drift is reported separately because the current deterministic
task/audit gates do not reject it.

The training mix explains part of the behavior: the train split contains 5,970
`agent_policy` samples but only 597 `auditable_answer` samples. A single adapter
is therefore optimized approximately 10:1 for intermediate JSON policy actions
over user-facing answers. It should not be released in this form.

Before RL, build SFT-v2.1 with separate policy/answer heads or adapters, or at
minimum loss-balanced sampling; strengthen final-answer targets for per-claim
citations, supported inferences, concise structured reasoning, valid quant
artifacts, and Chinese output consistency. Calibrate the audit parser against a
human-adjudicated slice of the observed failures and add language-drift plus
per-task continuous-score gates. Re-run the same frozen held-out set after that
change.

Trajectory RL remains useful and likely necessary for retrieval choice, query
planning, evidence selection, verification, and abstention. It should not start
from the current adapter/reward pair: optimizing a model with a 46.67% audit
pass rate and known output-format drift would mostly amplify SFT composition
errors and reward-parser boundaries. Start agentic RL only after SFT-v2.1 passes
the final-answer gates and the expanded reward red-team suite; keep delayed
market feedback capped and subordinate to fact, audit, and compliance rewards.

Artifacts:

- `saves/eval_results/qwen3_base_trusted_predictions.json`
- `saves/eval_results/qwen3_sft_v2_trusted_predictions.json`
- `saves/eval_results/sft_v2_trusted.json`
- `saves/eval_results/sft_v2_trusted.md`
