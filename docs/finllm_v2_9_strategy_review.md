# finLLM v2.9 阶段策略复盘与总体路线

日期：2026-09-09（Asia/Shanghai）  
基座：Qwen3-8B  研究范围：`financial_qa`、`quant_strategy`、`stock_analysis`

## 结论先行

本阶段停止盲目迭代。最后一个可用候选是 `sft-v2.7-core-selected`；v2.8.2
stock contract adapter 已因端到端退化拒绝。当前不允许 DPO/GRPO，也不能把
GRPO 代码中的配置或启发式 reward 当作已经取得的实验效果。

当前最重要的事实是：模型在 `@8` 采样中经常能生成合格候选，但 `@1` 不稳定；
同时严格 validator 已经把 false acceptance 压到 0，却在 adversarial
`stock_analysis` 上产生了严重 false rejection。下一阶段的工作应先统一
数据、任务契约、gold 和 verifier，再决定是否做受控 GRPO。

## 1. 现有训练数据是怎样得到的

### 1.1 原始轨迹层

v2 的设计目标是通过 LangGraph agentic API 取得真实完成轨迹：意图/claim 规划、
查询规划、本地或 Web 检索、证据归一化、来源质量和时间门、生成、claim graph、
合规与 grounding 验证，最后落盘完整 trajectory。入库前由审计 schema 和
`reward_v2` 的 hard gate 检查证据、引用、数值、时间、合规、quant action 和
轨迹完整性，默认还要求 reward 不低于 `0.65`。

这是正确的数据来源原则，但后续核心 SFT 并不是每轮重新调用 API。v2.5、v2.6
由已经审计的 answer 数据转换为紧凑目标；v2.7-core 又从 v2.6 过滤出三项任务。
因此，当前模型主要学习的是“固定证据输入下的最终答案协议”，而不是完整的检索
策略。不能把这些 answer rows 误称为 810 条全新的 agent trajectory。

### 1.2 版本谱系与选择原因

| 层/版本 | 数据处理 | 规模 | 为什么保留或淘汰 |
|---|---|---:|---|
| v2 | 真实 agent policy + answer | 6 类，约 10:1 policy:answer | 证明 schema/轨迹链路可用，但最终答案监督太少且出现语言漂移 |
| v2.3 | 原子主张、数值复制、审计对抗修复 | 6 类 | 关闭部分 citation/numeric 失败，但仍有弱任务非劣问题 |
| v2.5 | 短目标、无 `<think>`、failure repair、quant renderer、task balance | 1620（每类 270） | 目标通过构建期 audit/numeric/citation gate，适合稳定协议学习 |
| v2.6 | 紧凑 claim、逐句引用、拒答边界、hard negatives | 1620（每类 270） | 用于继续压缩输出和强化边界；后期 checkpoint 审计质量下降 |
| v2.7-core | 从 v2.6 只保留 QA/quant/stock | 810（每类 270） | 聚焦产品核心任务，避免无关任务稀释容量；step 153 按 Dev-Audit 选出 |
| v2.8 preference | 从 v2.7 重建 validator-aligned pairs | 637（QA 201、quant 270、stock 166） | 仅保留证据完整 chosen；chosen accept=1、rejected FA=0，旧 173 条 chosen rejection 不再作正例 |
| v2.8.2 repair | 把全部 stock target 改为四锚点/缺 MA 拒答 | 810 | seen preflight 上 E2E@8 从 0.68 降至 0.28，adapter 拒绝 |

数据选择的共同原则是：任务平衡、source/group disjoint、短且确定的目标、
构建期 validator、每条训练样本对应 hard negative。它们防止长答案和单一任务
支配 CE，也避免把不可审计的模板答案混入训练。代价是样本多为同一证据契约下的
派生/复制，覆盖的真实检索决策和跨来源冲突仍不足。

### 1.3 为什么 chosen acceptance 只有 78.64%

旧 810 对 preference 并非单纯“validator 过严”：173 条 rejected chosen 全部
可归因于数据与契约不一致。

- QA：69 条 prompt 没有可见 evidence，却要求 chosen 带引用；
- stock：104 条 chosen 没有 MA/趋势锚点，而新 stock contract 将其设为必需。

重建后的 637 对达到 chosen accept `1.0000`、rejected false accept `0/637`，
但这只证明新数据符合新契约，不证明模型已经学会该契约。旧 preference 必须
保留作错误分析，不得继续作为正例或直接 DPO 数据。

## 2. SFT 相对 Qwen3-8B 基座取得了什么

### 2.1 可确认的收益

在同协议的早期六任务配对评测（300 条）中，基座 pass `0.0167`、audit `0.0200`、
mean primary `0.5674`；SFT v2 pass `0.3867`、audit `0.4667`、mean primary
`0.6397`，paired primary delta `+0.0724`，且 pass 状态改进/回退为 `112/1`。
最大 token 截断从 `112/300` 降到 `15/300`，平均输出长度从 `1018.95` 降到
`445.95`。这说明 SFT 确实学会了更短、更有格式、更多引用的回答。

在三任务 v2.7-core seen regression 中，选定的 step 153 相对基座：

| 指标 | trusted | adversarial |
|---|---:|---:|
| E2E@1（base → SFT） | 0.0333 → 0.5733 | 0.0133 → 0.5933 |
| E2E@8 | 0.9000 | 0.8600 |
| Audit@1 | 0.7400 | 0.8600 |
| Audit@8 | 0.9600 | 0.9733 |
| mean primary delta（clustered CI） | +0.0804 `[+0.0257,+0.1299]` | +0.0814 `[+0.0236,+0.1347]` |

quant 的收益最稳定（trusted/adversarial primary delta `+0.4197/+0.4250`），
说明 deterministic quant action + renderer 比让模型生成 Python 更可靠。

### 2.2 仍未解决的缺口

- Audit@1 仍为 `74%/86%`，远低于产品门槛 `95%`；
- QA trusted primary delta `-0.0466`，stock trusted/adversarial 为
  `-0.1318/-0.2993`，所以整体提升不能掩盖任务级回退；
- stock adversarial E2E@1/@8 只有 `0.14/0.70`，task-aware validator 的
  selector E2E@8 只有 `0.04`；正确候选池 recall 仅 `0.0633`，主要是缺趋势或
  财务锚点，不是简单 threshold 问题；
- selector false acceptance 已为 0，但这是以大量 false rejection 换来的；
  validator 不能直接作为生产 best-of-8、DPO label 或 GRPO reward；
- `<think>` 已压到 0 是协议成功，不代表模型具备更好的内部决策；当前 NLL
  诊断显示 correct mean `0.0276`、incorrect `0.0904`，但 wrong-accepted
  仍为低 NLL `0.0298`，故 confidence 不能替代事实验证；
- v2.8.2 全量 stock target rewrite 使 E2E@8 `0.68 → 0.28`、primary delta
  `-0.0414 → -0.1682`，证明“大规模重写 + 继续训练”可能破坏已有能力。

这些结果回答了“为什么不继续盲训”：SFT 已完成协议塑形和 quant 稳定化，剩余
问题是候选生成、任务契约一致性、证据到主张的语义校验和单次解码稳定性；继续
降低 CE loss 并不能直接解决它们。

## 3. SFT 应达到的合理终点

SFT 的职责是教会模型在固定可见证据下稳定输出“最小充分答案”，而不是让它
通过 RL 学检索探索。建议把 SFT release target 定为：

1. source-disjoint final holdout（三个核心任务各至少 50 条，另设 adversarial）
   上 Audit@1 `>=95%`，hard-negative FA `=0`；
2. overall 与每任务 primary 的 clustered 95% CI 下界不低于 `-0.02`；
3. stock 的逐项 literal-copy、趋势/财务/风险锚点和拒答边界分别报告，不能用
   overall 平均掩盖；
4. E2E@8 `>=0.80` 且最弱任务 `>=0.70`，同时把 E2E@1 提升到至少 `0.70`，
   让 @8 不再依赖昂贵的 oracle best-of-8；
5. 生成完成率、中文语言一致性、过度拒答率、延迟和 token 成本纳入 gate。

达到这些条件后，SFT 可视为“固定证据答案策略”基本完成；若 @8 高而 @1 仍
明显较低，才有充分理由做稳定性型 GRPO，而不是用 RL 修复数据或 parser 错误。

## 4. GRPO 对齐应如何设计

### 4.1 训练对象与前置条件

GRPO 的 rollout 单位应是完整 agent trajectory，而不是只有一段 final answer：
`claim/query plan → retrieval actions → evidence selection → draft → verification →
retry/abstain → final answer`。每个 prompt 采样 `G=4~8` 条轨迹，保存每一步
action、observation/Evidence ID、时间戳、validator 版本和原始模型输出。

只有 SFT 通过上述 release target、建立真正 untouched final holdout、并扩展
reward attack suite 后才启动。当前仓库的 `train_grpo_trl.py` 是训练入口样板；
它目前使用简化的 task/format/length reward，不能代表本项目所需的完整审计 reward。

### 4.2 Reward 分解

采用“硬门不可补偿 + 软分数”的标量奖励，并同时记录分量，便于审计和 credit
assignment：

```text
R(trajectory) = 0                                  if any hard gate fails
R = 0.14 retrieval + 0.20 claim_support
  + 0.12 numeric + 0.12 citation_coverage
  + 0.10 citation_precision + 0.10 source_quality
  + 0.08 temporal_validity + 0.09 task_validity
  + 0.05 trajectory_quality + R_market              otherwise
```

硬门包括伪造/缺失引用、实体或检索不相关、unsupported claim、数值漂移、未来
信息、无效计算、冲突未处理、合规违规、非法 quant artifact、缺失检索/验证轨迹。
即时事实与合规奖励主导；延迟市场反馈只在全部硬门通过且预注册 horizon、benchmark、
sample size、风险量度有效时加入，`|R_market| <= 0.05`，不能把失败轨迹变成正例。

建议增加两个防 reward hacking 的统计量：每个 claim 的 evidence precision/recall，
以及“拒答是否由证据缺失触发”的 abstention calibration。检索过多、重复查询和
无效重试给小幅 cost penalty，但不能用长度奖励惩罚必要证据。

### 4.3 Advantage 与优化

对同一 prompt 的 group rewards 使用 leave-one-out 或 group normalization：

```text
A_i = (R_i - mean(R_group\{i\})) / (std(R_group\{i\}) + 1e-6)
```

当 group 全部为 0 时不制造伪梯度，记录为“无可学习信号”并进入 SFT/retrieval
数据回放。对 token-level policy gradient 使用 completion mask 和长度归一化，
避免长轨迹凭 token 数获得额外权重；采用 PPO/GRPO ratio clipping `epsilon=0.1~0.2`，
并相对冻结的 SFT reference 加 KL 惩罚（初始 `beta≈0.01~0.03`，按 held-out
KL 漂移调节）。每个 batch 按 task 分桶，避免 quant 的高可验证 reward 淹没 QA/stock。

训练顺序建议分三阶段：

1. **事实/协议阶段**：只用 deterministic QA、quant 和 stock evidence contract，
   不加市场 reward；先验证 reward attack zero-FA。
2. **轨迹阶段**：加入查询选择、证据重排、验证/补检索和有依据拒答的过程奖励；
   仅对通过 final hard gate 的轨迹保留正向 terminal reward。
3. **弱市场阶段**：仅在合规且事实正确的 stock 轨迹上加入 capped、风险调整后
   的 posterior return；使用时间切分和多个 benchmark，报告收益 reward 的置信区间，
   不把它作为发布主指标。

每阶段都要与 SFT paired baseline 比较 `Audit@1/@8`、FA/FR、primary、stock recall、
轨迹成本和 KL；任何 hard gate 回退立即停止 RL，而不是调高 reward 权重。

## 5. 对 GRPO 效果的真实判断

截至本报告没有完成任何 GRPO/真实 RLVF 训练，因此不存在“GRPO 已提升多少”的
实测结论。现有 `configs/qwen3_8b_qlora_grpo_trl.yaml` 的 `task_reward=1.0`、
`format=0.15`、`length=0.05` 以及 `num_generations=4` 只是样板配置；其 reward
函数更接近关键词/代码格式检查，未消费完整 `reward_v2` trajectory，也没有证明
Claim-to-Evidence 对齐或 stock false acceptance 已解决。

合理的成功判据应是相对“通过 SFT gate 的固定 checkpoint”，而不是相对基座：

- Audit@1 不下降且达到 `>=95%`，hard-negative FA 继续为 0；
- primary overall/per-task clustered CI 不劣，stock adversarial recall 不下降；
- 若目标是稳定性，主要看 E2E@1 提升和 `E2E@8-E2E@1` 缩小，而不是只看 reward 均值；
- retrieval recall、证据 precision、拒答校准、轨迹长度/成本和 KL 漂移必须同时报告；
- 市场 reward 只作辅助分层分析，不得成为“模型变得更可信”的证据。

若 GRPO 不能在这些条件下改善 @1，说明问题仍在数据、协议或 verifier，应该回退
到 SFT/工具链修复。

## 6. 项目的可用理想状态

理想产品不是“预测涨跌的聊天模型”，而是一个可拒答、可复核、可回放的金融决策
辅助系统：

- **输入与检索**：请求带 as-of 时间；Agent 规划 claim，检索并记录 query、来源、
  publisher、发布时间/生效时间/获取时间、版本和内容 hash；
- **分析与输出**：每个原子结论末尾有 Evidence ID；数字逐字复制；计算有受限表达式
  和输入证据；证据不足则明确拒答；quant 由固定 action + deterministic renderer
  生成，不让模型自由写代码；
- **验证与选择**：task-aware validator 同时检查审计和任务有效性，selector 不读取
  gold；不确定或冲突时升级人工，而不是生成貌似完整的答案；
- **运营指标**：持续监控 audit precision/recall、FA/FR、primary 非劣、stock recall、
  abstention calibration、语言/格式、延迟和成本；每次模型、数据、检索库或 validator
  变更均有 manifest、哈希和可复现实验；
- **责任边界**：输出是有时间界限的研究辅助，不承诺收益、不替代投资决策；市场后验
  只能用于离线监控与小权重对齐。

在该状态下，SFT 负责“正确、简洁、可审计的条件反射”，GRPO 只负责“在多个可行
轨迹中更稳定地选择检索、验证和拒答路径”。如果没有可靠的证据和验证基础，GRPO
只会更快地优化错误目标。

## 当前决策

**FAIL / HOLD：**保留 `sft-v2.7-core-selected` 作为基准，拒绝 v2.8.2 adapter，
不进入 DPO/GRPO。下一次允许的实验不是大规模训练，而是建立 source-disjoint
stock contract final holdout、统一 gold/verifier、完成 reward attack 扩展和
小规模 SFT ablation；只有这些通过后，才重新评估是否需要 GRPO。
