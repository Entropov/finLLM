# RLVF 对齐训练与多维评估重设计

## Summary
将现有单一路径 `SFT -> DPO` 改为任务感知的顺序对齐：`SFT -> DPO 通用偏好对齐 -> TRL-GRPO 规则验证对齐`。本地 LLaMA-Factory 源码当前不支持 `stage: grpo`，因此 DPO 继续由 LLaMA-Factory 执行，GRPO 由 TRL `GRPOTrainer` 执行，并从 DPO adapter 继续训练到 RLVF adapter。

## Alignment Policy
- `stock_analysis`：DPO 主导，偏好支撑/压力、趋势、成交量、MACD/RSI、风险提示完整的回答；可选 GRPO 只用于结构化技术指标覆盖奖励。
- `financial_report`：DPO 主导，偏好财务指标引用、亮点/风险平衡、数值一致性和结构完整性。
- `financial_qa`：DPO 主导开放问答；对 FinEval/选择题/可抽取标准答案子集启用 GRPO，奖励答案抽取正确率和解释简洁性。
- `sentiment_analysis`：GRPO 主导，奖励标签抽取正确、三类标签格式稳定、不过度生成；DPO 仅保留少量“正确标签 vs 错误标签”偏好样本作为兼容补充。
- `quant_strategy`：GRPO 主导，奖励 Python 语法、代码块、函数/import、策略入场出场/仓位/止损、回测指标覆盖；DPO 补充代码可读性和解释质量偏好。
- `risk_assessment`：DPO + GRPO 混合，DPO 偏好完整风险论证，GRPO 奖励风险等级抽取、风险因素、缓释建议、VaR/波动率/回撤等量化指标覆盖。

## Key Changes
- 新增任务策略注册表，统一声明每类任务的对齐方法、数据目标量、奖励函数、评估指标和训练顺序；`train_rlhf.sh` 改为 `--method dpo|grpo|all`、`--tasks`、`--skip-data`，默认执行顺序单产物。
- 扩展偏好数据生成：输出 `data/rlhf/fin_dpo_preference_train.json` / `eval.json`，按任务生成 rejected 策略和过滤规则，避免用通用截断劣化所有任务。
- 新增 GRPO 数据与奖励准备：输出 prompt-only/group-sampling 所需数据，保留 `task_type`、`reward_spec`、`reference`、`answer_key` 等元信息；实现离线奖励函数，并由 TRL `GRPOTrainer` 训练时复用。
- 增加 TRL-GRPO 配置与训练入口：`configs/qwen3_8b_qlora_grpo_trl.yaml` 和 `scripts/training/train_grpo_trl.py`，从 DPO adapter 继续训练到 `saves/qwen3-8b/lora/rlvf`。
- 更新 `dataset_info.json`：保留现有 `fin_preference_*` 兼容项，新增 DPO/GRPO 数据集注册，命名与任务策略一致。
- 更新文档：把“RLHF(DPO)”改为“RLVF(DPO + TRL-GRPO)”，说明本地默认可复现路径、GRPO 依赖和推荐训练命令。

## Evaluation Design
- 将 `eval_rlhf_comparison.py` 改为多模型、多任务评估：比较 `sft`、`dpo`、`rlvf` 三个 adapter，按 6 类任务分组汇总。
- 复用并扩展当前任务专项指标：情感 accuracy/macro-F1，QA ROUGE-L/关键词覆盖，股票技术要素覆盖，财报指标/数值引用，量化策略代码语法与策略要素，风险等级/因素/缓释/量化指标。
- 新增对齐专项指标：DPO 偏好胜率、GRPO reward 均值/分位数、格式合规率、过度拒答率、风险提示充分率、各任务相对 SFT/DPO 的提升表。
- LLM Judge 默认关闭，仅评估开放生成任务；使用 `EVAL_JUDGE_API_KEY`、`EVAL_JUDGE_API_BASE`、`EVAL_JUDGE_MODEL`，不参与默认训练数据生成或奖励打分。

## Test Plan
- 数据测试：验证 6 类任务都有明确 alignment policy；DPO/GRPO 输出格式、任务分布、训练/评估切分、dataset_info 注册一致。
- 奖励函数测试：覆盖情感标签正确性、QA 标准答案抽取、代码语法检查、策略要素覆盖、风险等级抽取、技术/财报指标覆盖。
- 训练入口测试：覆盖 `--method dpo|grpo|all` 参数解析；TRL-GRPO dry-run 能检查配置、数据和依赖。
- 评估测试：覆盖多 adapter 加载配置、6 类任务指标汇总、Judge 无 API Key 自动跳过。
- 验证命令：
  - `python scripts/data_processing/synthesize_preference_data.py --max-samples 500 --mode rules`
  - `bash scripts/training/train_rlhf.sh --method all --skip-train`
  - `python scripts/training/train_grpo_trl.py --config configs/qwen3_8b_qlora_grpo_trl.yaml --dry-run`
  - `python scripts/evaluation/eval_rlhf_comparison.py --task all --judge-mode quick`
  - `pytest tests/ -v`

## Assumptions
- 采用用户确认的默认：TRL-GRPO、顺序单产物、Judge 默认只用于评估。
- GRPO 依赖本地 `trl>=0.18` 的 `GRPOTrainer`，环境导入失败时训练入口直接失败并提示修复依赖。
- 不改 SFT 训练入口，RLVF 从现有 SFT adapter 继续训练。
