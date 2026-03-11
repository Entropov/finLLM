#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 指令数据合成模块

基于采集的原始数据（K线、新闻、财报），调用大模型 API 生成高质量指令-回答对。
为 6 类金融任务分别设计了 Prompt 模板:
  1. 股票市场分析
  2. 量化策略研发
  3. 财报解读
  4. 金融新闻情感分析
  5. 金融问答/知识
  6. 风险评估

支持:
  - 批量调用 OpenAI 兼容 API（Qwen API / DeepSeek / OpenAI 等）
  - 异步并发提升吞吐
  - 结果缓存与断点续传
  - 输出 ShareGPT 格式

输出: data/processed/synthesized/ 目录
"""

import os
import json
import time
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHESIZED_DIR = PROCESSED_DIR / "synthesized"
CACHE_DIR = SYNTHESIZED_DIR / ".cache"  # 请求结果缓存目录
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"

# ============================================================
# API 配置（从环境变量读取，保护密钥安全）
# ============================================================
API_BASE_URL = os.environ.get("SYNTH_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.environ.get("SYNTH_API_KEY", "")
API_MODEL = os.environ.get("SYNTH_API_MODEL", "qwen-plus")
MAX_CONCURRENT = int(os.environ.get("SYNTH_MAX_CONCURRENT", "5"))  # 最大并发数
REQUEST_TIMEOUT = int(os.environ.get("SYNTH_TIMEOUT", "60"))  # 请求超时（秒）


def _load_system_prompts() -> dict:
    """加载 system prompt 模板。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def _cache_key(task: str, content: str) -> str:
    """生成缓存键（基于任务类型和输入内容的 MD5）。"""
    raw = f"{task}_{content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cached(cache_key: str) -> Optional[str]:
    """从缓存中获取已有结果。"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f).get("response")
    return None


def _save_cache(cache_key: str, response: str) -> None:
    """保存请求结果到缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, ensure_ascii=False)


# ============================================================
# 各任务的 Prompt 模板
# ============================================================

TASK_TEMPLATES = {
    "stock_analysis": {
        "instruction_template": (
            "以下是{stock_name}（{stock_code}）最近的行情数据摘要:\n"
            "{market_data}\n\n"
            "请从技术面角度分析该股票的走势，包括:\n"
            "1. K线形态分析\n"
            "2. 关键技术指标解读（MACD、KDJ、RSI等）\n"
            "3. 支撑位和压力位判断\n"
            "4. 短期趋势研判和操作建议"
        ),
        "data_source": "stock_prices",
    },
    "quant_strategy": {
        "instruction_template": (
            "基于以下A股市场数据特征:\n"
            "{market_context}\n\n"
            "请设计一个{strategy_type}量化交易策略，要求:\n"
            "1. 说明策略核心逻辑和因子选择依据\n"
            "2. 给出完整的 Python 实现代码（使用 pandas/numpy）\n"
            "3. 说明入场/出场条件和仓位管理规则\n"
            "4. 分析策略的风险点和适用市场环境"
        ),
        "data_source": "stock_prices",
    },
    "financial_report": {
        "instruction_template": (
            "以下是{company_name}（{stock_code}）最新一期的关键财务数据:\n"
            "{financial_data}\n\n"
            "请对该公司进行深度财务分析，包括:\n"
            "1. 盈利能力分析（ROE、毛利率、净利率等）\n"
            "2. 偿债能力评估（资产负债率、流动比率等）\n"
            "3. 运营效率分析（应收账款周转率等）\n"
            "4. 同比/环比变化趋势及异常信号\n"
            "5. 综合投资价值评估"
        ),
        "data_source": "financial_reports",
    },
    "sentiment_analysis": {
        "instruction_template": (
            "请分析以下金融新闻的情感倾向和市场影响:\n\n"
            "标题: {title}\n"
            "内容: {content}\n\n"
            "要求:\n"
            "1. 判断情感倾向（积极/中性/消极）并给出置信度\n"
            "2. 识别涉及的公司/行业/政策\n"
            "3. 评估对相关股票/板块的潜在影响（方向和程度）\n"
            "4. 给出关键依据"
        ),
        "data_source": "news",
    },
    "financial_qa": {
        "instruction_template": "{question}",
        "data_source": "knowledge",
    },
    "risk_assessment": {
        "instruction_template": (
            "以下是{company_name}（{stock_code}）的财务和市场数据:\n"
            "{risk_data}\n\n"
            "请进行全面的风险评估:\n"
            "1. 信用风险评估（违约概率、财务健康度）\n"
            "2. 市场风险评估（波动率、Beta、VaR估算）\n"
            "3. 行业风险因素\n"
            "4. 综合风险等级评定（低/中/高/极高）\n"
            "5. 风险缓释建议"
        ),
        "data_source": "financial_reports",
    },
}

# 金融知识问题库（用于生成金融问答数据）
FINANCE_QUESTIONS = [
    "什么是市盈率（PE）？如何用市盈率判断股票估值是否合理？",
    "请解释股票市场中的做空机制是如何运作的？",
    "什么是夏普比率？如何用它来评估投资组合的表现？",
    "请解释A股市场的涨跌停板制度及其设计目的。",
    "什么是可转债？它的投资价值和风险有哪些？",
    "请解释什么是量化交易中的alpha和beta。",
    "CPI和PPI有什么区别？它们对股市有怎样的影响？",
    "什么是北向资金？为什么它被视为A股的风向标？",
    "请解释期权的四种基本策略（买入看涨、卖出看涨、买入看跌、卖出看跌）。",
    "什么是杜邦分析法？如何用它分析企业的ROE？",
    "什么是资产负债表中的商誉？商誉减值意味着什么？",
    "请解释什么是股票的技术分析和基本面分析，各自的优缺点是什么？",
    "什么是ETF基金？它与普通开放式基金有什么区别？",
    "请解释DCF（自由现金流折现）估值模型的基本原理和步骤。",
    "什么是融资融券？融资融券余额的变化对市场意味着什么？",
]


async def call_llm_api(
    system_prompt: str,
    user_prompt: str,
    cache_key: str,
) -> Optional[str]:
    """
    调用 LLM API 生成回答（异步，带缓存和重试）。

    参数:
        system_prompt: 系统提示词
        user_prompt:   用户输入
        cache_key:     缓存键

    返回:
        模型生成的回答文本，失败返回 None
    """
    # 先查缓存
    cached = _get_cached(cache_key)
    if cached:
        return cached

    import aiohttp

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        _save_cache(cache_key, content)
                        return content
                    elif resp.status == 429:
                        # 限流，等待后重试
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"API限流，等待 {wait_time}s 后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await resp.text()
                        logger.error(f"API调用失败 (HTTP {resp.status}): {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"API请求超时 (第{attempt+1}次)")
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return None

    return None


def prepare_stock_analysis_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """
    从行情数据中准备股票分析任务的输入。

    参数:
        data_dir:    清洗后的行情数据目录
        max_samples: 最大样本数

    返回:
        [{"stock_code": ..., "stock_name": ..., "market_data": ...}, ...]
    """
    inputs = []
    stock_dir = data_dir / "stock_prices"
    if not stock_dir.exists():
        logger.warning(f"行情数据目录不存在: {stock_dir}")
        return inputs

    for code_dir in sorted(stock_dir.iterdir()):
        if not code_dir.is_dir():
            continue
        daily_file = code_dir / "daily.csv"
        if not daily_file.exists():
            continue

        try:
            df = pd.read_csv(daily_file)
            if len(df) < 30:
                continue  # 数据太少跳过

            # 取最近 30 个交易日的摘要
            recent = df.tail(30)
            summary_lines = []
            for _, row in recent.iterrows():
                line = (
                    f"日期:{row.get('date','')}, "
                    f"开:{row.get('open','')}, 高:{row.get('high','')}, "
                    f"低:{row.get('low','')}, 收:{row.get('close','')}, "
                    f"量:{row.get('volume','')}, 涨跌幅:{row.get('pct_change','')}%"
                )
                summary_lines.append(line)

            # 添加最新技术指标
            last_row = df.iloc[-1]
            tech_summary = (
                f"\n最新技术指标: "
                f"MACD_DIF={last_row.get('MACD_DIF','N/A'):.4f}, "
                f"MACD_DEA={last_row.get('MACD_DEA','N/A'):.4f}, "
                f"RSI_6={last_row.get('RSI_6','N/A'):.2f}, "
                f"RSI_12={last_row.get('RSI_12','N/A'):.2f}, "
                f"KDJ_K={last_row.get('KDJ_K','N/A'):.2f}, "
                f"KDJ_D={last_row.get('KDJ_D','N/A'):.2f}, "
                f"KDJ_J={last_row.get('KDJ_J','N/A'):.2f}"
            )

            market_data = "\n".join(summary_lines[-10:]) + tech_summary  # 取最近10天+指标

            inputs.append({
                "stock_code": code_dir.name,
                "stock_name": code_dir.name,  # 实际可从成分股表获取
                "market_data": market_data,
            })

        except Exception as e:
            logger.debug(f"处理 {code_dir.name} 时出错: {e}")

        if len(inputs) >= max_samples:
            break

    return inputs


def prepare_sentiment_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """从新闻数据中准备情感分析任务的输入。"""
    inputs = []
    news_dir = data_dir / "news"
    if not news_dir.exists():
        return inputs

    for json_file in news_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                news_list = json.load(f)
            for item in news_list:
                title = item.get("title", "")
                content = item.get("content", "")
                if title and len(title) > 10:
                    inputs.append({"title": title, "content": content[:500]})
                if len(inputs) >= max_samples:
                    break
        except Exception:
            continue
        if len(inputs) >= max_samples:
            break

    return inputs


async def synthesize_task(
    task_name: str,
    inputs: list,
    system_prompt: str,
    template: str,
    output_file: Path,
) -> int:
    """
    为单个任务批量合成指令数据。

    参数:
        task_name:     任务名称
        inputs:        输入数据列表
        system_prompt: 系统提示词
        template:      指令模板
        output_file:   输出文件路径

    返回:
        成功合成的条数
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []

    async def process_one(item: dict, idx: int):
        async with semaphore:
            try:
                user_prompt = template.format(**item)
            except KeyError as e:
                logger.debug(f"模板填充失败: {e}")
                return None

            ck = _cache_key(task_name, user_prompt)
            response = await call_llm_api(system_prompt, user_prompt, ck)
            if response:
                return {
                    "conversations": [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": response},
                    ],
                    "system": system_prompt,
                    "task_type": task_name,
                }
            return None

    tasks = [process_one(item, i) for i, item in enumerate(inputs)]
    results_raw = await tqdm_asyncio.gather(*tasks, desc=f"合成-{task_name}")

    results = [r for r in results_raw if r is not None]

    # 保存到文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_name}] 合成完成: {len(results)}/{len(inputs)} 条")
    return len(results)


async def run_synthesis(
    tasks: Optional[list] = None,
    max_per_task: int = 500,
) -> None:
    """
    运行完整的数据合成流程。

    参数:
        tasks:        要合成的任务列表，None=全部
        max_per_task: 每个任务最大合成数量
    """
    if not API_KEY:
        logger.error(
            "未设置 API Key！请设置环境变量:\n"
            "  export SYNTH_API_KEY='your-api-key'\n"
            "  export SYNTH_API_BASE='https://api.openai.com/v1'  # 或其他兼容API\n"
            "  export SYNTH_API_MODEL='gpt-4o-mini'"
        )
        return

    prompts = _load_system_prompts()
    all_tasks = list(TASK_TEMPLATES.keys()) if tasks is None else tasks

    logger.info(f"开始合成数据，任务列表: {all_tasks}")
    logger.info(f"每任务最大样本数: {max_per_task}")
    logger.info(f"API: {API_BASE_URL}, 模型: {API_MODEL}\n")

    total = 0
    for task_name in all_tasks:
        if task_name not in TASK_TEMPLATES:
            logger.warning(f"未知任务: {task_name}，跳过")
            continue

        config = TASK_TEMPLATES[task_name]
        prompt_info = prompts.get(task_name, prompts["general"])
        system_prompt = prompt_info["system_prompt"]
        template = config["instruction_template"]

        # 准备输入数据
        if task_name == "stock_analysis":
            inputs = prepare_stock_analysis_inputs(PROCESSED_DIR, max_per_task)
        elif task_name == "sentiment_analysis":
            inputs = prepare_sentiment_inputs(PROCESSED_DIR, max_per_task)
        elif task_name == "financial_qa":
            inputs = [{"question": q} for q in FINANCE_QUESTIONS]
        else:
            # 其他任务需要对应的数据准备函数，这里用占位
            logger.info(f"[{task_name}] 数据准备函数待扩展，使用示例数据...")
            inputs = []

        if not inputs:
            logger.warning(f"[{task_name}] 无可用输入数据，跳过")
            continue

        output_file = SYNTHESIZED_DIR / f"{task_name}.json"
        count = await synthesize_task(
            task_name=task_name,
            inputs=inputs,
            system_prompt=system_prompt,
            template=template,
            output_file=output_file,
        )
        total += count

    logger.info(f"\n合成全部完成，共生成 {total} 条指令数据")
    logger.info(f"输出目录: {SYNTHESIZED_DIR}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 指令数据合成")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(TASK_TEMPLATES.keys()),
        help="指定要合成的任务（默认全部）",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=500,
        help="每个任务最大合成样本数 (默认: 500)",
    )
    args = parser.parse_args()

    asyncio.run(run_synthesis(tasks=args.tasks, max_per_task=args.max_per_task))
