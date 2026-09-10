#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 检索增强模块

功能:
  - 加载 ChromaDB 向量数据库
  - 加载 Embedding 模型进行在线 Query 向量化
  - 执行知识检索 (Top K)
  - 组装知识注入 Prompt
"""

import os
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None

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
# 默认配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_DIR = PROJECT_ROOT / "saves" / "chroma_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "fin_knowledge"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"环境变量 {name}={value} 不是合法整数，使用默认值 {default}")
        return default


DEFAULT_ENABLE_MULTI_QUERY = _env_bool("RAG_ENABLE_MULTI_QUERY", True)
DEFAULT_MULTI_QUERY_COUNT = _env_int("RAG_MULTI_QUERY_COUNT", 4)
DEFAULT_MULTI_QUERY_CANDIDATE_MULTIPLIER = _env_int("RAG_MULTI_QUERY_CANDIDATE_MULTIPLIER", 3)

QUERY_FILLERS = (
    "请问", "帮我", "请帮我", "分析一下", "分析", "说明一下", "说明", "介绍一下", "介绍",
    "一下", "什么是", "是什么", "为什么", "为何", "怎么", "如何", "有哪些", "多少",
    "是否", "能否", "可以", "请", "吗", "呢",
)

QUESTION_INTENT_REPLACEMENTS = {
    "为什么": "原因",
    "为何": "原因",
    "怎么样": "表现",
    "如何看待": "影响",
    "怎么看": "影响",
    "有哪些": "相关因素",
    "是什么": "",
    "什么是": "",
}

FINANCE_EXPANSION_RULES = [
    (
        ("利润", "净利润", "盈利", "业绩", "增长", "下滑", "亏损"),
        ("净利润", "营业收入", "毛利率", "ROE", "同比", "环比", "业绩驱动因素"),
    ),
    (
        ("收入", "营收", "销售额"),
        ("营业收入", "主营业务", "收入结构", "销量", "价格", "增长驱动"),
    ),
    (
        ("估值", "市盈率", "pe", "市净率", "pb", "股价"),
        ("估值", "市盈率", "市净率", "股价", "业绩预期", "可比公司"),
    ),
    (
        ("现金流", "自由现金流", "经营现金流"),
        ("经营现金流", "自由现金流", "现金流量净额", "应收账款", "资本开支"),
    ),
    (
        ("风险", "违约", "暴雷", "监管", "合规"),
        ("风险因素", "经营风险", "财务风险", "市场风险", "监管风险", "违约风险"),
    ),
    (
        ("银行", "商业银行", "净息差", "不良贷款"),
        ("净息差", "不良贷款率", "拨备覆盖率", "资本充足率", "利息净收入"),
    ),
    (
        ("保险", "保费", "偿付能力"),
        ("保费收入", "赔付率", "投资收益", "偿付能力", "新业务价值"),
    ),
    (
        ("基金", "持仓", "净值"),
        ("基金净值", "重仓股", "资产配置", "回撤", "收益率"),
    ),
    (
        ("债券", "债务", "利差", "信用债"),
        ("债券收益率", "信用利差", "久期", "评级", "偿债能力"),
    ),
    (
        ("宏观", "利率", "通胀", "央行", "货币政策", "汇率"),
        ("宏观经济", "利率", "通胀", "货币政策", "汇率", "流动性"),
    ),
    (
        ("财报", "年报", "季报", "公告"),
        ("财报", "年报", "季报", "公告", "管理层讨论", "财务指标"),
    ),
]

class RAGRetriever:
    """RAG 检索处理器"""
    def __init__(
        self, 
        db_dir: str = str(DEFAULT_DB_DIR), 
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        device: str = "cuda",
        enable_multi_query: bool = DEFAULT_ENABLE_MULTI_QUERY,
        multi_query_count: int = DEFAULT_MULTI_QUERY_COUNT,
        multi_query_candidate_multiplier: int = DEFAULT_MULTI_QUERY_CANDIDATE_MULTIPLIER,
        rrf_k: int = 60,
    ):
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.collection_name = collection_name
        self.enable_multi_query = enable_multi_query
        self.multi_query_count = max(1, multi_query_count)
        self.multi_query_candidate_multiplier = max(1, multi_query_candidate_multiplier)
        self.rrf_k = max(1, rrf_k)
        
        self.embed_model = None
        self.chroma_client = None
        self.collection = None
        self.doc_count = 0
        
        self._initialize(device)

    def _load_collection(self):
        if self.chroma_client is None:
            self.collection = None
            self.doc_count = 0
            return

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            self.doc_count = self.collection.count()
            logger.info(f"RAG: 向量数据库连接成功，当前拥有知识库片段数量: {self.doc_count}")
        except Exception as e:
            logger.error(f"RAG: 加载 ChromaDB 失败或对应的集合未找到: {e}")
            self.collection = None
            self.doc_count = 0

    def _initialize(self, device):
        if chromadb is None or SentenceTransformer is None:
            logger.warning("RAG: 缺少依赖，请先安装 chromadb 和 sentence-transformers。")
            return

        if not self.db_dir.exists():
            logger.warning(f"向量数据库存放目录不存在，如需正常使用 RAG 请先执行 build_vector_db.py: {self.db_dir}")
            return

        logger.info(f"RAG: 初始化 Embedding 模型 ({self.model_name})...")
        try:
            self.embed_model = SentenceTransformer(self.model_name, device=device)
        except Exception as e:
            logger.error(f"RAG: 加载 Embedding 模型失败: {e}")
            return
            
        logger.info(f"RAG: 初始化 ChromaDB 客户端，路径: {self.db_dir}")
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
            self._load_collection()
        except Exception as e:
            logger.error(f"RAG: 加载 ChromaDB 失败或对应的集合未找到: {e}")
            self.collection = None
            self.doc_count = 0

    def refresh(self) -> bool:
        """重新加载 Collection，用于增量入库后热刷新检索器。"""
        if chromadb is None:
            return False

        if self.chroma_client is None:
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
            except Exception as e:
                logger.error(f"RAG: 重建 ChromaDB 客户端失败: {e}")
                return False

        self._load_collection()
        return self.is_ready()

    def is_ready(self) -> bool:
        """检查 RAG  Retriever 是否准备完毕。"""
        return self.collection is not None and self.embed_model is not None

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    def _keywordize_query(self, query: str) -> str:
        keyword_query = query
        for old, new in QUESTION_INTENT_REPLACEMENTS.items():
            keyword_query = keyword_query.replace(old, new)
        for filler in QUERY_FILLERS:
            keyword_query = keyword_query.replace(filler, " ")
        keyword_query = re.sub(r"[，。！？；：、,.!?;:()\[\]{}<>《》\"'“”‘’]", " ", keyword_query)
        keyword_query = re.sub(r"\s+", " ", keyword_query).strip()
        return keyword_query or query

    def _append_query_variant(self, variants: List[str], variant: str, limit: int):
        normalized_variant = self._normalize_query(variant)
        if not normalized_variant:
            return
        if normalized_variant not in variants and len(variants) < limit:
            variants.append(normalized_variant)

    def _contains_trigger(self, query_lower: str, trigger: str) -> bool:
        trigger_lower = trigger.lower()
        if re.fullmatch(r"[a-z0-9]+", trigger_lower):
            pattern = rf"(?<![a-z0-9]){re.escape(trigger_lower)}(?![a-z0-9])"
            return re.search(pattern, query_lower) is not None
        return trigger_lower in query_lower

    def _generate_query_variants(self, query: str, limit: int) -> List[str]:
        """
        规则型多查询扩展：保留原问题，同时补充关键词化查询和金融领域同义/指标查询。
        """
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        variants = [normalized_query]
        if limit <= 1:
            return variants

        keyword_query = self._keywordize_query(normalized_query)
        self._append_query_variant(variants, keyword_query, limit)

        query_lower = normalized_query.lower()
        for triggers, expansions in FINANCE_EXPANSION_RULES:
            if len(variants) >= limit:
                break
            if any(self._contains_trigger(query_lower, trigger) for trigger in triggers):
                self._append_query_variant(
                    variants,
                    f"{keyword_query} {' '.join(expansions)}",
                    limit,
                )

        if len(variants) < limit:
            self._append_query_variant(
                variants,
                f"{keyword_query} 财务指标 影响因素 原因 风险",
                limit,
            )

        return variants

    def _encode_queries(self, queries: List[str]) -> List[List[float]]:
        embeddings = self.embed_model.encode(queries, show_progress_bar=False)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return embeddings

    def _result_key(self, doc_id: Optional[str], metadata: Dict[str, Any], content: str) -> str:
        if doc_id:
            return doc_id

        source = metadata.get("source")
        chunk_index = metadata.get("chunk_index")
        if source is not None and chunk_index is not None:
            return f"{source}:{chunk_index}"

        return content[:200]

    def _merge_query_results(self, results: Dict[str, Any], queries: List[str], top_k: int) -> List[Dict]:
        merged_docs: Dict[str, Dict[str, Any]] = {}

        docs_by_query = results.get("documents") or []
        metadatas_by_query = results.get("metadatas") or []
        distances_by_query = results.get("distances") or []
        ids_by_query = results.get("ids") or []

        for query_idx, docs in enumerate(docs_by_query):
            metadatas = metadatas_by_query[query_idx] if query_idx < len(metadatas_by_query) else []
            distances = distances_by_query[query_idx] if query_idx < len(distances_by_query) else []
            ids = ids_by_query[query_idx] if query_idx < len(ids_by_query) else []
            query_text = queries[query_idx] if query_idx < len(queries) else ""
            query_weight = 1.0 if query_idx == 0 else 0.85

            for rank, doc in enumerate(docs):
                metadata = metadatas[rank] if rank < len(metadatas) and metadatas[rank] else {}
                distance = distances[rank] if rank < len(distances) else float("inf")
                doc_id = ids[rank] if rank < len(ids) else None
                result_key = self._result_key(doc_id, metadata, doc)
                rrf_score = query_weight / (self.rrf_k + rank + 1)

                if result_key not in merged_docs:
                    merged_docs[result_key] = {
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "score": 0.0,
                        "matched_queries": [],
                    }

                item = merged_docs[result_key]
                item["score"] += rrf_score
                item["distance"] = min(item["distance"], distance)
                item["matched_queries"].append({
                    "query": query_text,
                    "rank": rank + 1,
                    "distance": distance,
                })

        return sorted(
            merged_docs.values(),
            key=lambda item: (-item["score"], item["distance"]),
        )[:top_k]

    def retrieve(self, query: str, top_k: int = 3, use_multi_query: Optional[bool] = None) -> List[Dict]:
        """
        检索最能回答 query 的外部知识片段。
        """
        if not self.is_ready() or top_k <= 0:
            return []

        query = self._normalize_query(query)
        if not query:
            return []

        if self.doc_count <= 0:
            return []

        # 1. 生成原始查询 + 扩展查询
        should_use_multi_query = self.enable_multi_query if use_multi_query is None else use_multi_query
        query_limit = self.multi_query_count if should_use_multi_query else 1
        queries = self._generate_query_variants(query, query_limit)
        logger.debug(f"RAG: 查询扩展结果: {queries}")

        # 2. 批量把查询词转换成向量
        query_embeddings = self._encode_queries(queries)

        # 3. 批量查询 ChromaDB，扩大候选集后用 RRF 融合多路结果
        candidate_k = top_k
        if len(queries) > 1:
            candidate_k = top_k * self.multi_query_candidate_multiplier
        candidate_k = min(candidate_k, self.doc_count)
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        # 4. 去重、融合排序并保留 top_k
        return self._merge_query_results(results, queries, top_k)

    def retrieve_by_entity_codes(self, codes: List[str], limit_per_code: int = 12) -> List[Dict]:
        """Recall chunks containing explicit six-digit issuer identifiers."""
        if not self.is_ready() or limit_per_code <= 0:
            return []

        docs: List[Dict] = []
        seen = set()
        for code in sorted({str(item) for item in codes if re.fullmatch(r"\d{6}", str(item))}):
            results = self.collection.get(
                where_document={"$contains": code},
                include=["documents", "metadatas"],
                limit=limit_per_code,
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            ids = results.get("ids") or []
            for index, content in enumerate(documents):
                metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
                doc_id = ids[index] if index < len(ids) else None
                key = self._result_key(doc_id, metadata, content)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(
                    {
                        "content": content,
                        "metadata": metadata,
                        "distance": None,
                        "score": 1.0,
                        "matched_queries": [{"query": code, "rank": 1, "match_type": "entity_exact"}],
                    }
                )
        return docs

    def format_prompt_with_context(self, system_prompt: str, query: str, top_k: int = 3) -> str:
        """
        组装带有 RAG 背景知识的全新 System Prompt。
        如果当前 Retriever 还未就绪（无数据），就降级原样返回原有的 system_prompt。
        """
        if not self.is_ready() or top_k <= 0:
            return system_prompt
            
        retrieved_docs = self.retrieve(query, top_k=top_k)
        if not retrieved_docs:
            return system_prompt

        context_str = "\n".join([f"- {doc['content']}" for doc in retrieved_docs])
        
        # 组装一个新的系统提示词前缀或者后缀
        rag_augmented_prompt = (
            f"{system_prompt}\n\n"
            "【参考知识库】\n"
            "以下是系统检索到的与用户问题高度相关的内部参考资料。你可以结合这些参考资料以及你自身的知识来回答用户的问题。如果参考资料提供了明确的时段及事件数值，请以此为准：\n"
            f"{context_str}\n" 
        )
        return rag_augmented_prompt

if __name__ == "__main__":
    # 测试脚本
    retriever = RAGRetriever(device="cpu")
    if retriever.is_ready():
        res = retriever.retrieve("测试句子")
        print("测试查询结果：", res)
    else:
        print("未准备好")
