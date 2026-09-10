# finLLM Financial Agent Repository Audit

> Audit date: 2026-09-09 (Asia/Shanghai)
>
> Target version: `v3.0-agent-harness`
>
> Scope of this change: PHASE 0 repository audit only. No runtime, training,
> model, checkpoint, dataset, or historical evaluation artifact was changed.

## 1. Executive conclusion

finLLM is not only a text-generation training repository. It already contains
an experimental auditable RAG pipeline, typed audit records, deterministic
reward gates, a constrained quant action/renderer, task-aware validators,
trajectory JSONL logging, SFT/DPO/GRPO data builders, and legacy inference
entry points. These capabilities are useful foundations, but they are spread
across `scripts/` and do not yet form the requested independent Financial
Agent Harness.

The safest extension point is a new importable package at `src/agent/` with a
compatibility layer around the existing `scripts/rag/` implementations. The
legacy training and inference paths should remain unchanged by default. Agent
mode can later be integrated explicitly through `--mode agent`, while the
current behavior remains `--mode legacy`.

The present system is research-grade and must not be described as production
ready. In particular, it has no unified risk/approval gate, no common tool
contract, no strict end-to-end AgentState, no independent validator orchestration,
no general repair loop, and no load/replay API for its event logs.

## 2. Repository and Git state

Current branch and upstream state at audit time:

- Branch: `master`.
- Upstream: `finllm/master`.
- Local branch is 2 commits ahead and 0 commits behind.
- HEAD: `c8f133e Improve finance LLM workflows and supporting infrastructure`.
- Other recent relevant commit: `7c32bbd` introduced RAG functionality.
- Remotes `origin`, `finLLM`, and `finllm` all point to
  `Entropov/finLLM` using SSH and/or HTTPS.

Pre-existing uncommitted changes were observed during the audit:

- Modified: `.gitignore`.
- Modified: `docs/qwen3_8b_auditable_financial_agent_research.md`.
- Untracked: `docs/finllm_v2_9_strategy_review.md`.
- Untracked at audit time: `and supporting infrastructure` and
  `how --stat 0e482a4`.

The two oddly named root files were subsequently confirmed to be identical
`less` help-page captures rather than project assets and were excluded from
the repository update.

## 3. Current directory tree

The following is the task-relevant tree. Generated caches, individual model
shards, logs, and the vendored LLaMA-Factory internal tree are collapsed.

```text
finLLM/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- rag_agentic.yaml
|   |-- qwen2.5_*_{sft,dpo,inference,merge_lora}.yaml
|   `-- qwen3_*_{sft,dpo,grpo,inference,merge_lora}.yaml
|-- data/
|   |-- dataset_info.json
|   |-- evaluation/
|   |   |-- trusted_finance_v2.json
|   |   |-- trusted_sft_v2.schema.json
|   |   |-- sft_v2_{3,4,5,6,7,8}*.json
|   |   `-- v2_heldout_corpus/
|   |-- knowledge/
|   |   |-- v2_train/
|   |   `-- web/
|   |-- processed/stock_prices/
|   |-- rag/
|   |   |-- audit_v2.schema.json
|   |   |-- trajectories/*.jsonl
|   |   |-- preferences/
|   |   `-- eval_sets/
|   |-- rlhf/
|   `-- sft_v2*/
|-- docs/
|   |-- agentic_sft_v2.md
|   |-- qwen3_8b_auditable_financial_agent_research.md
|   |-- finllm_v2_8_validator_calibration_report.md
|   |-- finllm_v2_8_1_stock_contract_preflight_report.md
|   |-- finllm_v2_8_2_stock_contract_repair_report.md
|   |-- finllm_v2_9_strategy_review.md
|   |-- rlhf_guide.md
|   `-- rlvr_plan.md
|-- prompts/
|   `-- system_prompts.json
|-- scripts/
|   |-- data_collection/
|   |-- data_processing/
|   |-- evaluation/
|   |-- inference/
|   |-- rag/
|   |-- rlhf/
|   `-- training/
|-- saves/
|   |-- qwen3-8b/merged/
|   |-- qwen3-8b/lora/
|   `-- eval_results/
|-- tests/
|   `-- 18 test modules
|-- test_fineval.py
`-- LLaMA-Factory/                 # vendored external training framework
```

There is no project-owned root `src/` package today. The only existing `src/`
belongs to vendored `LLaMA-Factory`.

## 4. Entrypoint inventory

### 4.1 Training

Primary legacy SFT entry:

- `scripts/training/train.sh` invokes `llamafactory-cli train` and defaults to
  the existing QLoRA config/data path.
- Versioned SFT wrappers exist from v2 through v2.8.2, including
  `train_sft_v2_7_core.sh` and `train_sft_v2_8_2_stock_contract.sh`. These
  wrappers run data/config gates before invoking LLaMA-Factory.

Preference/RL entries:

- `scripts/training/train_rlhf.sh` is the legacy RLHF/DPO wrapper.
- `scripts/training/train_dpo_v2_4.sh` is the hard-negative DPO wrapper.
- `scripts/training/train_grpo_trl.py` is a TRL GRPO entry with a `--dry-run`
  option, but its reward functions consume completions and task metadata, not
  the complete auditable agent trajectory required by the target architecture.
- `scripts/rlhf/alignment_policy.py` contains task-specific offline reward
  heuristics and an older `compute_reward(task_type, response, ...)` interface.

No large training or GRPO job was started during this audit.

### 4.2 Inference

- `scripts/inference/api_server.py`: FastAPI/OpenAI-compatible API. It supports
  `rag_mode` values `basic` and `agentic`, but this is specifically an Agentic
  RAG prompt preparation/finalization path, not the requested full risk/tool/
  validator/repair/approval loop.
- `scripts/inference/chat_demo.py`: Gradio demo with basic and agentic RAG.
- `scripts/inference/batch_inference.py`: task-aware batch generation.
- `scripts/inference/quick_test.py`: local model/adapter smoke generation.
- `scripts/inference/vllm_backend.py`: vLLM generation adapter.

The current interfaces do not expose a general `--mode legacy|agent` switch.
Compatibility integration should happen only after the standalone harness is
tested.

### 4.3 Data

- `data/dataset_info.json` is the LLaMA-Factory registry for legacy SFT,
  preference, GRPO, and versioned agentic SFT datasets.
- `scripts/data_collection/` fetches market prices, financial reports, news,
  and public datasets.
- `scripts/data_processing/` cleans/merges ShareGPT data, collects v2 agentic
  trajectories, builds SFT v2-v2.8.2 datasets, creates hard negatives and
  validator-aligned preference data, and prepares GRPO inputs.
- Point-in-time training and held-out evidence live under `data/knowledge/`
  and `data/evaluation/v2_heldout_corpus/`.
- Existing trajectories live under `data/rag/trajectories/`; they must be
  adapted or quarantined according to their schema version, never silently
  relabeled as new Agent trajectories.

### 4.4 Validator

There is no single runtime validator entry. Current validation is split among:

- `scripts/rag/audit_schema.py`: deterministic schema-level validation for
  Evidence, Claim, Calculation, TrajectoryEvent, and AuditEnvelope records.
- `scripts/rag/reward_v2.py`: non-compensable hard gates for unsupported
  claims, numeric drift, fabricated citations, temporal validity, evidence
  relevance, trajectory integrity, compliance, and quant artifact validity.
- `scripts/evaluation/task_aware_verifier_v2_7_core.py`: task-aware evidence
  anchor validator for core tasks.
- `scripts/evaluation/task_aware_verifier_v2_8_2.py`: stock contract repair
  variant with a cited missing-trend abstention path.
- Version-specific protocol and diagnosis scripts under `scripts/evaluation/`.

These are reusable deterministic validators, but none implements the requested
common `PASS | REPAIR | REFUSE` result or an independent Generator/Validator
boundary in the online runtime.

### 4.5 Evaluation

- `scripts/evaluation/eval_finance_bench.py`: FinEval-style benchmark entry.
- `scripts/evaluation/eval_task_specific.py`: six-task generation evaluation.
- `scripts/evaluation/eval_agentic_rag.py` and
  `scripts/evaluation/validate_agentic_rewards.py`: RAG/reward evaluation.
- v2-v2.8.2 protocol, E2E, checkpoint selection, selector replay, validator
  calibration, and stock contract diagnostics are separate scripts in the
  same directory.
- Machine-readable outputs are preserved under `saves/eval_results/`.

### 4.6 Prompts

- `prompts/system_prompts.json` is the only top-level prompt source. It defines
  six task prompts and a general prompt.
- Agentic SFT builders also embed/version protocol text in Python-generated
  data. There are no standalone risk, planner, tool policy, validator, repair,
  renderer, quant contract, or replay prompt files.
- The current `quant_strategy` prompt explicitly asks the model to write
  complete Python and report backtest metrics. That conflicts with the target
  canonical Strategy JSON contract and must remain legacy-only.

## 5. Current task implementations

### financial_qa

Legacy behavior is prompt-driven generation through batch/API/Gradio. Agentic
RAG classifies general queries as `financial_qa`, retrieves evidence, builds
claims, and applies reward hard gates. v2.7/v2.8 task validators check visible
evidence anchors and citations. It can abstain when evidence is missing, but
there is no explicit RiskDecision or AskEvidence action schema.

### stock_analysis

The current pipeline can retrieve local market/financial documents, optionally
collect fresh web documents, normalize evidence, and validate price/return,
trend, risk, and financial anchors. v2.8.2 added a cited abstention when MA5/
MA20 trend evidence is genuinely absent. The latest reports still reject the
candidate adapter because end-to-end stock quality and validator recall are
insufficient.

### quant_strategy

`scripts/rag/quant_protocol.py` already prevents arbitrary model-generated
Python in the v2.5 agentic path: the model may only emit one fixed JSON action,
which materializes a repository-owned MA5/MA20 code artifact. This is an
important compatibility asset, but it is narrower than the requested Strategy
JSON schema and does not run a deterministic backtest engine or bind every
reported metric to an actual ToolResult. The legacy prompt and older reward
code still assume free-form Python, so legacy and agent contracts must be kept
explicitly separate.

## 6. Existing agent and audit primitives

`scripts/rag/agentic_rag.py` is a LangGraph pipeline with sequential fallback.
Its main graph performs intent classification, claim planning, retrieval policy
selection, query planning, local/web retrieval, reranking, evidence
normalization, source quality checks, generation, quant materialization, claim
graph construction, contradiction handling, compliance checks, grounding
verification, optional refinement, and trajectory logging.

Useful primitives to preserve behind adapters:

| Target concern | Existing asset | Gap |
|---|---|---|
| Evidence/claims/calculations | `scripts/rag/audit_schema.py` dataclasses | Not a unified AgentState; surrounding pipeline passes raw dictionaries |
| Retrieval | `retriever.py`, `knowledge_agent.py`, collection scripts | No common ToolRequest/ToolResult interface |
| Deterministic validation | `reward_v2.py`, task-aware verifiers | No normalized validator status or repair actions |
| Quant constraints | `quant_protocol.py` | Fixed single template; no Strategy JSON/backtest contract |
| Refinement | Agentic RAG refine branch and answer guard | Not a general bounded validator-driven repair loop |
| Audit trail | `TrajectoryEvent`, `AuditEnvelope`, JSONL writes | Event vocabulary differs; no load/replay/timeline API |
| Policy | rule/bandit/learned RAG retrieval policy | No financial risk gate or permission/approval policy |
| Rewards | `compute_auditable_reward` | Component names/output differ from requested trajectory reward interface |

The current audit schema deliberately strips `<think>` content and stores
verifiable artifacts rather than hidden chain-of-thought. This is aligned with
the target replay requirement and should remain invariant.

## 7. SFT, DPO, GRPO, and reward status

SFT has evolved through agentic v2-v2.8.2 builders and versioned training
configs. The current optimization scope documented by the project is
`financial_qa`, `quant_strategy`, and `stock_analysis`; historical
`financial_report`, `risk_assessment`, and `sentiment_analysis` assets remain
available but are explicitly outside the current release gate.

Hard-negative preference builders cover citation drops, numeric drift,
unsupported forecasts, long thinking, and invalid quant actions. v2.8 builds a
637-pair validator-aligned preference set from evidence-complete chosen rows.
The older 810-pair set contains 173 chosen rows that conflict with the newer
validator contract and must not be reused as verifier-positive data without
repair.

GRPO code exists, but the project reports explicitly prohibit a GRPO/DPO run at
the current gate. The TRL reward path is completion-centric and not yet wired
to `compute_auditable_reward` over full trajectories. The existing Reward v2
is the stronger starting point for a future offline adapter because its hard
failures cannot be offset by soft scores.

## 8. README, configuration, models, and checkpoints

`README.md` primarily describes the older Qwen2.5/Fin-Instruct workflow and six
free-form generation tasks. It includes SFT, DPO, RAG, API, Gradio, and batch
commands, but it does not yet document the current v2.7/v2.8 research gate or
the proposed v3 Agent Harness. It should remain the legacy quick start until
agent mode is actually runnable.

Configuration is currently flat under `configs/`. `configs/rag_agentic.yaml`
contains retrieval, refinement, trajectory, and Reward v2 thresholds. There is
no `configs/agent/` hierarchy for risk, validator, quant, replay, or approval.

Model artifacts and evaluation results exist under `saves/`, including the
merged Qwen3-8B model, LoRA/checkpoint material, v2.7 checkpoint-selection
outputs, and v2.8 validator results. They were inspected only by path and were
not changed. The currently documented viable baseline is
`saves/qwen3-8b/lora/sft-v2.7-core-selected`; v2.8.2 is rejected as a candidate.

## 9. Tests and baseline result

Project-owned `tests/` contains 18 modules covering inference/data paths,
Agentic RAG, audit/reward validation, RLVF, SFT v2.2-v2.8.2 protocols,
checkpoint selectors, and task-aware validators.

Commands and results at audit time:

```text
/home/super/.conda/envs/finllm/bin/pytest tests -q
178 passed, 1 failed in 3.27s
```

The failure is pre-existing:

- `tests/test_rlvf_pipeline.py::test_train_rlhf_help_mentions_new_methods`
  expects `scripts/training/train_rlhf.sh --help` to mention
  `--method dpo|grpo|all`, but the script currently exposes only its older DPO
  options.

Running from the repository root does not reach test execution:

```text
/home/super/.conda/envs/finllm/bin/pytest -q
2 collection errors
```

Root collection failures:

- Pytest collects both vendored `LLaMA-Factory/tests` and
  `LLaMA-Factory/tests_v1`, which contain duplicate `test_converter` module
  names and trigger an import-file mismatch.
- `test_fineval.py` downloads `FinEval.zip` from Hugging Face during module
  import. The request failed with SSL `UNEXPECTED_EOF_WHILE_READING`, making
  ordinary offline test collection network-dependent.

The active shell's default Python is from the `torchcu128` environment and has
no pytest installed. The repository's usable test runner is the explicit
`finllm` environment path shown above. Bash also prints a non-fatal
`libtinfo.so.6` version warning on every command.

## 10. Known functional and release failures

1. Release Gate is FAIL. Existing reports prohibit deployment claims and
   DPO/GRPO progression.
2. v2.8 validator calibration has zero observed candidate-pool false accepts,
   but adversarial stock correct-candidate rejection is 93.67%; it is not
   qualified as a deployment or reward selector.
3. v2.8.2 stock contract rewrite regressed seen E2E@8 from 0.68 to 0.28; its
   adapter was rejected.
4. Generator and validator are not clean runtime components. Agentic RAG calls
   generation and validation nodes inside one large pipeline class.
5. `AgenticRAGState` is a `TypedDict`, and module boundaries still exchange
   mutable raw dictionaries. Runtime schema validation is incomplete.
6. There is no requested Risk Gate decision model, permission model, or human
   approval state. Compliance regexes cover some unsafe advice but do not
   authorize actions.
7. There is no uniform tool registry/result contract. Retrieval, web access,
   data collection, calculation, and quant behavior use different APIs.
8. Tool failures are recorded in pipeline error lists but are not universally
   represented as explicit FAILED ToolResults.
9. Repair is limited to pipeline refinement and a task-limited extractive
   answer guard. It is not a maximum-two-round independent validation loop.
10. Trajectories can be saved, but the required event taxonomy, event versions,
    load, replay, and inspect-timeline interfaces are absent.
11. Agentic quant avoids dynamic model code, but its canonical object is a
    fixed renderer action rather than a parameter-complete Strategy JSON, and
    it does not produce actual backtest-bound performance results.
12. Legacy quant prompts and RLVF rewards still reward complete Python output,
    which conflicts with the new agent contract if the modes are mixed.
13. README and default prompts are stale relative to the current Qwen3/v2.8
    research state.
14. Root pytest collection is not hermetic, and the scoped suite already has
    one RLVF CLI contract failure.

## 11. Recommended compatibility boundary

Add a project-owned `src/agent/` package and treat it as an application/runtime
layer. Do not move or rename existing scripts during the first implementation.
The new package should own strict schemas and orchestration; adapters should
translate existing audit/retrieval/validator artifacts at the boundary.

Recommended ownership:

```text
src/agent/
|-- __init__.py
|-- schemas.py       # strict shared models and enums
|-- state.py         # state construction/evolution helpers
|-- events.py        # event model only
|-- risk_gate.py
|-- policy.py
|-- planner.py
|-- tools.py         # interface/registry; adapters live beside it initially
|-- validator.py     # normalized independent validator facade
|-- repair.py
|-- renderer.py
|-- replay.py
`-- loop.py
```

Compatibility rules:

- Keep all legacy commands and dataset registrations working.
- Do not import new Agent types into LLaMA-Factory.
- Do not change existing v2 schema meanings in place; add explicit adapters
  and versions.
- Agent mode owns the new quant Strategy JSON; legacy mode may retain its old
  prompt/output for reproducibility.
- Persist only structured events, tool I/O, evidence, decisions, versions, and
  state snapshots, never hidden model reasoning.
- Fail closed when an adapter cannot validate old data.

## 12. PHASE 1 minimum implementation plan

PHASE 1 should implement State + Schema only. Risk rules, planner behavior,
tools, loop execution, prompts, validators, repair, and inference integration
belong to later phases.

1. Create `src/__init__.py`, `src/agent/__init__.py`,
   `src/agent/schemas.py`, and `src/agent/state.py`.
2. Use Pydantic models with `extra="forbid"` and assignment/default validation.
   Add Pydantic as an explicit project dependency instead of relying on
   FastAPI's transitive installation.
3. Define enums for TaskType, RiskLevel, RiskDecision, PlannerAction,
   ValidationStatus, ApprovalStatus, Permission, ToolStatus, and EventType.
4. Define strict models for RiskState, TaskState, Evidence, Observation,
   PlannerDecision, Claim, Calculation, ValidationResult, ApprovalState,
   FinalOutput, and AgentState. Every collection has a typed element; no
   module-facing `dict[str, Any]` is permitted.
5. Define ToolRequest, ToolExecution, ToolResult, and AgentEvent now because
   AgentState refers to them, but do not implement tool execution or event
   persistence in PHASE 1.
6. Require timezone-aware timestamps, non-empty session/event IDs, bounded
   validator scores, unique evidence IDs, valid cross-references, and a final
   output consistent with validation/approval state.
7. Add explicit `from_legacy_audit(...)` adapter stubs or a narrow implemented
   adapter for `AuditEnvelope`; never reinterpret old JSONL implicitly.
8. Add `tests/agent/test_schemas.py` and `tests/agent/test_state.py` for valid
   construction, rejected unknown fields/enums, invalid timestamps/scores,
   immutable IDs, typed append operations, and serialization round trips.
9. Run the new tests first, then `/home/super/.conda/envs/finllm/bin/pytest
   tests -q`. Record the existing RLVF help failure separately; do not weaken
   schema tests to preserve the baseline.
10. Update this audit with PHASE 1 status only after the schema implementation
    and both test commands have actually completed.

PHASE 1 exit criteria: a consumer can construct, validate, serialize, and load
a complete empty/new AgentState without passing raw dictionaries between new
agent modules, and malformed states are rejected before any planner, tool,
validator, renderer, or replay code can consume them.
