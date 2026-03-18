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
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

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

class RAGRetriever:
    """RAG 检索处理器"""
    def __init__(
        self, 
        db_dir: str = str(DEFAULT_DB_DIR), 
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = "fin_knowledge",
        device: str = "cuda"
    ):
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.collection_name = collection_name
        
        self.embed_model = None
        self.chroma_client = None
        self.collection = None
        
        self._initialize(device)

    def _initialize(self, device):
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
            # 获取已存在的 Collection
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            doc_count = self.collection.count()
            logger.info(f"RAG: 向量数据库连接成功，当前拥有知识库片段数量: {doc_count}")
        except Exception as e:
            logger.error(f"RAG: 加载 ChromaDB 失败或对应的集合未找到: {e}")
            self.collection = None

    def is_ready(self) -> bool:
        """检查 RAG  Retriever 是否准备完毕。"""
        return self.collection is not None and self.embed_model is not None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索最能回答 query 的外部知识片段。
        """
        if not self.is_ready():
            return []

        # 1. 把查询词转换成向量
        query_embedding = self.embed_model.encode([query])[0].tolist()

        # 2. 查询 ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 3. 解析结果
        retrieved_docs = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if ("metadatas" in results and results["metadatas"]) else [{}] * len(docs)
            distances = results["distances"][0] if ("distances" in results and results["distances"]) else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist  # 相似度度量（如果在L2空间距离越小越相似）
                })
                
        return retrieved_docs

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
