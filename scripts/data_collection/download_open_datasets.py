#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公开金融数据集下载

从 HuggingFace Hub 下载金融领域公开数据集:
  1. FinGPT 系列（情感分析、QA、关系抽取）
  2. DISC-FinLLM（中文金融全任务数据）
  3. FinEval（金融评测基准）

支持 HuggingFace 镜像加速，具备断点续传和自动重试能力。
"""

import os
import sys
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# ★★★ 在任何 HuggingFace 相关导入之前设置镜像 ★★★
# ============================================================
# 方案1: hf-mirror.com（国内最稳定的 HF 镜像）
# 方案2: modelscope 替代下载（部分数据集可用）
HF_MIRROR_URL = "https://hf-mirror.com"

# 设置环境变量（必须在 import datasets/huggingface_hub 之前）
os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
# 如果使用代理，取消下面注释并填入你的代理地址
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 设置 HuggingFace 缓存目录（可选，避免默认目录磁盘空间不足）
# os.environ["HF_HOME"] = "/home/super/workspace/finLLM/.cache/huggingface"

import pandas as pd
from tqdm import tqdm

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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================
# 数据集注册表
# ============================================================
# 每个数据集的配置：仓库ID、子集名、分割集、预期条数、描述
DATASET_REGISTRY: List[Dict] = [
    {
        "name": "fingpt_sentiment",
        "repo_id": "FinGPT/fingpt-sentiment-train",
        "subset": None,
        "split": "train",
        "expected_count": 76800,
        "description": "FinGPT 金融情感分析数据集",
        "save_subdir": "fingpt_sentiment",
    },
    {
        "name": "fingpt_fiqa_qa",
        "repo_id": "ZixuanKe/fingpt-fiqa-qa",
        "subset": None,
        "split": "train",
        "expected_count": 17100,
        "description": "FinGPT 金融问答数据集 (FiQA)",
        "save_subdir": "fingpt_fiqa_qa",
    },
    {
        "name": "fingpt_finred",
        "repo_id": "ZixuanKe/fingpt-finred",
        "subset": None,
        "split": "train",
        "expected_count": 27600,
        "description": "FinGPT 金融关系抽取数据集",
        "save_subdir": "fingpt_finred",
    },
    {
        "name": "fingpt-headline",
        "repo_id": "FinGPT/fingpt-headline",
        "subset": None,
        "split": "train",
        "expected_count": 82200,
        "description": "FinGPT 金融新闻标题生成数据集",
        "save_subdir": "fingpt_headline",
    },
    {
        "name": "fingpt-fine",
        "repo_id": "FinGPT/fingpt-fineval",
        "subset": None,
        "split": "train",
        "expected_count": 10600,
        "description": "FinGPT 金融评测数据集",
        "save_subdir": "fingpt_fineval",
    },
    # {
    #     "name": "disc_finllm",
    #     "repo_id": "eggbiscuit/DISC-FIN-SFT",  # 修改这里，指向真正的数据集仓库
    #     "subset": None,
    #     "split": "train",
    #     "expected_count": 246000,
    #     "description": "DISC-FinLLM 中文金融指令数据集 (DISC-FIN-SFT)",
    #     "save_subdir": "disc_finllm"
    # },
]


def check_mirror_connectivity() -> bool:
    """
    检测 HuggingFace 镜像是否可用。

    返回:
        True 表示镜像可访问
    """
    import urllib.request

    test_url = f"{HF_MIRROR_URL}/api/models?limit=1"
    try:
        req = urllib.request.Request(test_url, method="GET")
        req.add_header("User-Agent", "finLLM-downloader/1.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                logger.info(f"✓ HuggingFace 镜像可用: {HF_MIRROR_URL}")
                return True
    except Exception as e:
        logger.warning(f"✗ HuggingFace 镜像不可用: {e}")
    return False


def check_hf_direct_connectivity() -> bool:
    """
    检测 HuggingFace 官方源是否可直连。

    返回:
        True 表示官方源可访问
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/models?limit=1", method="GET"
        )
        req.add_header("User-Agent", "finLLM-downloader/1.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                logger.info("✓ HuggingFace 官方源可直连")
                return True
    except Exception:
        pass
    return False


def setup_best_endpoint() -> str:
    """
    自动选择最佳的 HuggingFace 下载端点。

    优先级: 镜像 > 官方源

    返回:
        选中的端点 URL
    """
    # 优先尝试镜像
    if check_mirror_connectivity():
        os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
        return HF_MIRROR_URL

    # 回退到官方源
    if check_hf_direct_connectivity():
        os.environ.pop("HF_ENDPOINT", None)
        return "https://huggingface.co"

    logger.error(
        "无法连接到 HuggingFace（镜像和官方源均不可用）！\n"
        "请尝试以下解决方案:\n"
        "  1. 设置 HTTP 代理: export HTTPS_PROXY=http://127.0.0.1:7890\n"
        "  2. 使用其他镜像: export HF_ENDPOINT=https://hf-mirror.com\n"
        "  3. 手动下载数据集到 data/raw/ 目录"
    )
    sys.exit(1)


def download_single_dataset_hf(
    config: Dict,
    save_base_dir: Path,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> Tuple[bool, str]:
    """
    从 HuggingFace Hub 下载单个数据集（带健壮的重试机制）。

    参数:
        config:        数据集配置字典
        save_base_dir: 保存根目录
        max_retries:   最大重试次数
        retry_delay:   重试间隔（秒），每次重试翻倍

    返回:
        (是否成功, 结果描述)
    """
    name = config["name"]
    repo_id = config["repo_id"]
    save_dir = save_base_dir / config["save_subdir"]
    save_file = save_dir / "data.parquet"

    # 检查是否已下载
    if save_file.exists():
        logger.info(f"  [跳过] {name}: 已存在 {save_file}")
        return True, "已存在，跳过"

    save_dir.mkdir(parents=True, exist_ok=True)

    current_delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"  [{attempt}/{max_retries}] 正在下载 {name} "
                f"(repo: {repo_id})..."
            )

            # ★ 关键：每次重试都重新导入，确保新的 HTTP 客户端
            # 这解决了 "client has been closed" 的问题
            import importlib
            import datasets

            importlib.reload(datasets)

            # 使用 load_dataset 下载
            ds = datasets.load_dataset(
                repo_id,
                name=config.get("subset"),
                split=config.get("split", "train"),
                trust_remote_code=True,
                # 增加下载超时时间
                download_config=datasets.DownloadConfig(
                    max_retries=3,
                    num_proc=1,  # 单线程下载，减少连接问题
                ),
            )

            # 保存为 Parquet（高效存储）
            ds.to_parquet(str(save_file))

            # 同时保存一份 JSON（方便查看）
            json_file = save_dir / "data.json"
            ds.to_json(str(json_file), force_ascii=False)

            actual_count = len(ds)
            expected = config.get("expected_count", 0)
            logger.info(
                f"  [成功] {name}: {actual_count} 条 "
                f"(预期 ~{expected})"
            )

            # 保存元信息
            meta = {
                "name": name,
                "repo_id": repo_id,
                "split": config.get("split"),
                "actual_count": actual_count,
                "expected_count": expected,
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint": os.environ.get("HF_ENDPOINT", "official"),
            }
            meta_file = save_dir / "meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, f"下载成功，{actual_count} 条"

        except Exception as e:
            error_msg = str(e)
            logger.warning(
                f"  [{attempt}/{max_retries}] {name} 下载失败: "
                f"{error_msg[:200]}"
            )

            if attempt < max_retries:
                logger.info(f"  等待 {current_delay:.0f} 秒后重试...")
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, 120)  # 指数退避，最长2分钟

                # ★ 清理可能残留的损坏缓存
                _clear_hf_cache(repo_id)
            else:
                return False, f"重试 {max_retries} 次后仍失败: {error_msg[:200]}"

    return False, "未知错误"


def _clear_hf_cache(repo_id: str) -> None:
    """
    清理指定仓库的 HuggingFace 缓存，避免损坏的缓存影响重试。

    参数:
        repo_id: HuggingFace 仓库 ID
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if repo_id.replace("/", "--") in str(repo_info.repo_path):
                logger.info(f"  清理缓存: {repo_info.repo_path}")
                # 删除不完整的下载
                for revision in repo_info.revisions:
                    if revision.size_on_disk < 1024:  # 小于1KB视为损坏
                        delete_strategy = cache_info.delete_revisions(
                            revision.commit_hash
                        )
                        delete_strategy.execute()
    except Exception:
        pass  # 缓存清理失败不影响主流程


def download_from_modelscope_fallback(
    config: Dict,
    save_base_dir: Path,
) -> Tuple[bool, str]:
    """
    ModelScope 备用下载方案（当 HuggingFace 完全不可用时）。

    注意: 并非所有 HF 数据集都在 ModelScope 上有镜像。

    参数:
        config:        数据集配置字典
        save_base_dir: 保存根目录

    返回:
        (是否成功, 结果描述)
    """
    # ModelScope 上的映射仓库（部分数据集可能没有）
    ms_repo_map = {
        "disc_finllm": "Go4miii/DISC-FinLLM",
        "fineval": "SUFE-AIFLM-Lab/FinEval",
    }

    name = config["name"]
    if name not in ms_repo_map:
        return False, f"ModelScope 无此数据集: {name}"

    try:
        from modelscope.msdatasets import MsDataset

        save_dir = save_base_dir / config["save_subdir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "data.parquet"

        if save_file.exists():
            return True, "已存在，跳过"

        logger.info(f"  [ModelScope 备用] 正在下载 {name}...")
        ds = MsDataset.load(ms_repo_map[name], split=config.get("split", "train"))

        # 转为 HuggingFace Dataset 格式后保存
        hf_ds = ds.to_hf_dataset()
        hf_ds.to_parquet(str(save_file))

        return True, f"ModelScope 下载成功，{len(hf_ds)} 条"

    except ImportError:
        return False, "未安装 modelscope（pip install modelscope）"
    except Exception as e:
        return False, f"ModelScope 下载失败: {e}"


def download_all_datasets(
    max_retries: int = 5,
    use_modelscope_fallback: bool = True,
) -> None:
    """
    批量下载所有注册的金融数据集。

    流程:
      1. 检测并选择最佳下载端点
      2. 逐个下载数据集（带重试）
      3. 失败的尝试 ModelScope 备用方案
      4. 输出汇总报告

    参数:
        max_retries:              每个数据集的最大重试次数
        use_modelscope_fallback:  是否启用 ModelScope 备用下载
    """
    logger.info("=" * 60)
    logger.info("Fin-Instruct 公开数据集下载工具")
    logger.info("=" * 60)

    # Step 1: 选择最佳端点
    endpoint = setup_best_endpoint()
    logger.info(f"使用端点: {endpoint}\n")

    # Step 2: 逐个下载
    results = []
    for config in DATASET_REGISTRY:
        logger.info(f"[{config['name']}] {config['description']}")

        # 先尝试 HuggingFace
        success, msg = download_single_dataset_hf(
            config=config,
            save_base_dir=RAW_DATA_DIR,
            max_retries=max_retries,
        )

        # HF 失败则尝试 ModelScope
        if not success and use_modelscope_fallback:
            logger.info(f"  HuggingFace 下载失败，尝试 ModelScope 备用方案...")
            success, msg = download_from_modelscope_fallback(
                config=config,
                save_base_dir=RAW_DATA_DIR,
            )

        results.append(
            {
                "name": config["name"],
                "description": config["description"],
                "success": success,
                "message": msg,
            }
        )
        logger.info(f"  → {msg}\n")

        # 下载间隔，避免被限流
        if not success:
            time.sleep(5)
        else:
            time.sleep(2)

    # Step 3: 输出汇总报告
    logger.info("=" * 60)
    logger.info("下载汇总报告")
    logger.info("=" * 60)

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(f"  {status} {r['name']}: {r['message']}")

    logger.info(f"\n总计: {success_count} 成功, {fail_count} 失败")
    logger.info(f"数据保存目录: {RAW_DATA_DIR}")

    if fail_count > 0:
        logger.warning(
            "\n失败的数据集可以稍后单独重新下载:\n"
            "  python download_open_datasets.py --only <dataset_name>\n"
            "或手动下载后放入对应的 data/raw/<subdir>/ 目录"
        )

    # 保存下载报告
    report_file = RAW_DATA_DIR / "download_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"下载报告已保存: {report_file}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="公开金融数据集下载工具")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="每个数据集的最大重试次数 (默认: 5)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只下载指定数据集，如: fingpt_sentiment",
    )
    parser.add_argument(
        "--no-modelscope",
        action="store_true",
        help="禁用 ModelScope 备用下载",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="自定义 HuggingFace 镜像地址",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP 代理地址，如: http://127.0.0.1:7890",
    )
    args = parser.parse_args()

    # 设置自定义镜像
    if args.mirror:
        HF_MIRROR_URL = args.mirror
        os.environ["HF_ENDPOINT"] = args.mirror
        logger.info(f"使用自定义镜像: {args.mirror}")

    # 设置代理
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        logger.info(f"使用代理: {args.proxy}")

    # 单独下载某个数据集
    if args.only:
        matched = [d for d in DATASET_REGISTRY if d["name"] == args.only]
        if not matched:
            available = [d["name"] for d in DATASET_REGISTRY]
            logger.error(
                f"未知数据集: {args.only}\n可用: {available}"
            )
            sys.exit(1)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.extend(matched)

    download_all_datasets(
        max_retries=args.max_retries,
        use_modelscope_fallback=not args.no_modelscope,
    )