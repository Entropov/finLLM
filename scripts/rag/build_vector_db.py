#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建向量数据库用于 RAG

功能:
  - 读取指定知识库目录下的文本文档
  - 进行文本切块 (Chunking)
  - 使用 Sentence-Transformers 生成文本 Embedding
  - 持久化存储至 ChromaDB

用法:
  python scripts/rag/build_vector_db.py --data-dir data/knowledge --db-dir saves/chroma_db
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("请先安装依赖：pip install chromadb sentence-transformers")

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "knowledge"
DEFAULT_DB_DIR = PROJECT_ROOT / "saves" / "chroma_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    使用滑动窗口进行简单的纯文本切块。
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk.strip():
            start += (chunk_size - overlap)
            continue
        chunks.append(chunk)
        if end >= text_len:
            break
        start += (chunk_size - overlap)
        
    return chunks

def load_documents(data_dir: Path) -> List[dict]:
    """
    加载指定目录下的所有 .txt 和 .md 格式的文档。
    """
    documents = []
    if not data_dir.exists():
        logger.warning(f"数据存放目录不存在，将自动创建: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        return documents

    for file_path in data_dir.glob("**/*"):
        if file_path.is_file() and file_path.suffix in ['.txt', '.md']:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "id": file_path.name,
                        "source": str(file_path.relative_to(PROJECT_ROOT)),
                        "content": content
                    })
                logger.info(f"读取文档成功: {file_path.name}")
            except Exception as e:
                logger.error(f"无法读取文件 {file_path.name}: {e}")
                
    return documents

def build_vector_db(
    data_dir: str, 
    db_dir: str, 
    model_name: str, 
    chunk_size: int, 
    chunk_overlap: int
):
    data_path = Path(data_dir)
    db_path = Path(db_dir)
    
    logger.info("1. 开始加载文档数据...")
    docs = load_documents(data_path)
    if not docs:
        logger.warning(f"未在目录 {data_dir} 找到任何文档，建库流程结束。请放入 .txt 或 .md 文件后重试。")
        return
        
    logger.info(f"共加载 {len(docs)} 个文档。")

    logger.info("2. 开始文本切分(Chunking)...")
    chunks_data = []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": chunk,
                "metadata": {"source": doc["source"], "chunk_index": i}
            })
    logger.info(f"总计切分出 {len(chunks_data)} 个片段。")

    logger.info(f"3. 正在加载 Embedding 模型: {model_name}...")
    embed_model = SentenceTransformer(model_name)

    logger.info("4. 正在初始化 ChromaDB 客户端...")
    db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # 集合（Collection）概念，类似于表
    collection_name = "fin_knowledge"
    try:
        # 尝试删除旧有的集合确保数据最新（可根据需求调整为追加更新）
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"已清理旧的 ChromaDB 集合: {collection_name}")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    logger.info("5. 正在计算 Embedding 并录入 ChromaDB...")
    
    # 将 chunk 相关数据转成 ChromaDB 接受的格式
    ids = [item["id"] for item in chunks_data]
    documents = [item["text"] for item in chunks_data]
    metadatas = [item["metadata"] for item in chunks_data]
    
    # 通过模型计算 embedding
    # 如果数据量极大，考虑分批（Batch）处理
    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    logger.info(f"建库完毕！向量数据库已持久化保存至: {db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG 本地向量数据库")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="知识库原始文档存储目录")
    parser.add_argument("--db-dir", type=str, default=str(DEFAULT_DB_DIR), help="持久化向量库存储目录")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Embedding 模型名称或路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="文本切片大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="文本切片重叠字数")
    
    args = parser.parse_args()
    build_vector_db(args.data_dir, args.db_dir, args.model_name, args.chunk_size, args.chunk_overlap)
