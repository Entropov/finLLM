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

import argparse
import hashlib
import logging
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "knowledge"
DEFAULT_DB_DIR = PROJECT_ROOT / "saves" / "chroma_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "fin_knowledge"

FINANCE_TERMS = {
    "资产", "负债", "收入", "利润", "现金流", "净利润", "营收", "毛利率", "ROE", "ROA",
    "估值", "市盈率", "市净率", "股息", "分红", "债券", "基金", "股票", "期货", "期权",
    "利率", "汇率", "通胀", "央行", "货币政策", "财政政策", "风险", "违约", "担保",
    "抵押", "贷款", "融资", "并购", "财报", "公告", "审计", "监管", "合规", "税",
}


def infer_document_source_quality(canonical_url: str) -> Tuple[str, str]:
    host = urlparse(canonical_url).netloc.lower()
    if host.endswith(("gov.cn", "pbc.gov.cn", "csrc.gov.cn", "stats.gov.cn")):
        return "regulator", "primary"
    if host.endswith(("sse.com.cn", "szse.cn", "hkex.com.hk")):
        return "exchange", "primary"
    if canonical_url:
        return "news", "unverified_secondary"
    return "internal", "internal"


@dataclass
class Chunk:
    """向量库入库前的文本片段。"""

    text: str
    strategy: str
    start_char: int
    end_char: int
    section_title: str = ""
    semantic_score: Optional[float] = None


@dataclass
class ChunkMetrics:
    """用于比较不同 Chunking 策略的轻量质量指标。"""

    strategy: str
    chunk_count: int
    avg_chars: float
    median_chars: float
    min_chars: int
    max_chars: int
    short_chunk_ratio: float
    sentence_boundary_ratio: float
    punctuation_end_ratio: float
    avg_semantic_cohesion: Optional[float] = None


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


def fixed_window_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    原始固定窗口切分策略，保留用于入库和对比。
    """
    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - overlap)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(
                Chunk(
                    text=chunk.strip(),
                    strategy="fixed",
                    start_char=start,
                    end_char=end,
                )
            )
        if end >= text_len:
            break
        start += step

    return chunks


def normalize_text(text: str) -> str:
    """压缩噪声空白，但保留段落/Markdown 结构信号。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S+", line))


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))


def split_structural_blocks(text: str) -> List[Tuple[str, str, int, int, str]]:
    """
    按 Markdown 标题、表格、列表、自然段切成结构块。

    返回: (block_type, block_text, start_char, end_char, section_title)
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    blocks = []
    current_lines = []
    current_type = "paragraph"
    current_start = 0
    current_section = ""
    active_section = ""
    cursor = 0

    def flush(end_pos: int):
        nonlocal current_lines, current_type, current_start, current_section
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((current_type, block_text, current_start, end_pos, current_section))
        current_lines = []

    for raw_line in normalized.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        line_start = cursor
        line_end = cursor + len(raw_line)
        cursor = line_end

        if not stripped:
            flush(line_start)
            current_type = "paragraph"
            continue

        if is_markdown_heading(stripped):
            flush(line_start)
            active_section = re.sub(r"^\s{0,3}#{1,6}\s+", "", stripped).strip()
            blocks.append(("heading", stripped, line_start, line_start + len(line), active_section))
            current_type = "paragraph"
            continue

        line_type = "table" if is_table_line(stripped) else "list" if is_list_line(stripped) else "paragraph"
        if not current_lines:
            current_start = line_start
            current_type = line_type
            current_section = active_section
        elif line_type != current_type and current_type in {"table", "list"}:
            flush(line_start)
            current_start = line_start
            current_type = line_type
            current_section = active_section

        current_lines.append(line)

    flush(len(normalized))
    return blocks


def split_sentences(text: str) -> List[str]:
    """
    中英混合句子切分，尽量在完整标点后断开。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=[。！？!?；;])", text)
    return [part.strip() for part in parts if part and part.strip()]


def split_long_text_by_punctuation(text: str, chunk_size: int) -> List[str]:
    """
    对超长句/表格做二级切分，优先按标点，不足时退回固定长度。
    """
    pieces = []
    buffer = ""
    units = split_sentences(text) or [text]

    for unit in units:
        if len(unit) > chunk_size:
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            for i in range(0, len(unit), chunk_size):
                piece = unit[i:i + chunk_size].strip()
                if piece:
                    pieces.append(piece)
            continue

        if buffer and len(buffer) + len(unit) + 1 > chunk_size:
            pieces.append(buffer.strip())
            buffer = unit
        else:
            buffer = f"{buffer} {unit}".strip()

    if buffer:
        pieces.append(buffer.strip())
    return pieces


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_embedding_func(embed_model: Optional[Any]) -> Optional[Callable[[List[str]], List[List[float]]]]:
    if embed_model is None:
        return None

    def embed(sentences: List[str]) -> List[List[float]]:
        return embed_model.encode(sentences, show_progress_bar=False).tolist()

    return embed


def finance_continuity_bonus(left: str, right: str) -> float:
    """
    金融语料里数字、证券代码、指标名经常跨句延续；共享信号越多，越不应该硬切。
    """
    left_terms = {term for term in FINANCE_TERMS if term in left}
    right_terms = {term for term in FINANCE_TERMS if term in right}
    shared_terms = left_terms & right_terms

    numeric_pattern = r"(?:\d+(?:\.\d+)?%?|[A-Z]{2,6}\.?[A-Z]?)"
    left_numbers = set(re.findall(numeric_pattern, left))
    right_numbers = set(re.findall(numeric_pattern, right))
    shared_numbers = left_numbers & right_numbers

    bonus = min(0.12, 0.03 * len(shared_terms))
    bonus += min(0.08, 0.02 * len(shared_numbers))
    return bonus


def make_semantic_chunk(
    sentences: List[str],
    strategy: str,
    start_char: int,
    section_title: str,
    semantic_scores: Optional[List[float]] = None,
) -> Chunk:
    text = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
    avg_score = None
    if semantic_scores:
        avg_score = sum(semantic_scores) / len(semantic_scores)
    return Chunk(
        text=text,
        strategy=strategy,
        start_char=start_char,
        end_char=start_char + len(text),
        section_title=section_title,
        semantic_score=avg_score,
    )


def semantic_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    min_chunk_size: int = 120,
    semantic_threshold: float = 0.55,
) -> List[Chunk]:
    """
    定制化智能语义切分策略：
      1. 先按 Markdown/段落/表格/列表保留显式结构；
      2. 对普通段落按中英标点切句；
      3. 用相邻句 embedding 相似度识别主题漂移；
      4. 对金融术语、数字、代码的跨句延续给连续性加权，减少误切；
      5. 对超长块按标点二级切分，避免超过 embedding 模型上下文。
    """
    blocks = split_structural_blocks(text)
    chunks: List[Chunk] = []

    for block_type, block_text, start_char, _end_char, section_title in blocks:
        if block_type in {"heading", "table", "list"}:
            for piece in split_long_text_by_punctuation(block_text, chunk_size):
                chunks.append(
                    Chunk(
                        text=piece,
                        strategy="semantic",
                        start_char=start_char,
                        end_char=start_char + len(piece),
                        section_title=section_title,
                    )
                )
            continue

        sentences = split_sentences(block_text)
        if not sentences:
            continue

        if len(sentences) == 1 and len(sentences[0]) <= chunk_size:
            chunks.append(
                Chunk(
                    text=sentences[0],
                    strategy="semantic",
                    start_char=start_char,
                    end_char=start_char + len(sentences[0]),
                    section_title=section_title,
                )
            )
            continue

        if any(len(sentence) > chunk_size for sentence in sentences):
            for piece in split_long_text_by_punctuation(block_text, chunk_size):
                chunks.append(
                    Chunk(
                        text=piece,
                        strategy="semantic",
                        start_char=start_char,
                        end_char=start_char + len(piece),
                        section_title=section_title,
                    )
                )
            continue

        embeddings = embed_func(sentences) if embed_func and len(sentences) > 1 else None
        adjacent_scores = []
        if embeddings:
            for idx in range(len(sentences) - 1):
                score = cosine_similarity(embeddings[idx], embeddings[idx + 1])
                score += finance_continuity_bonus(sentences[idx], sentences[idx + 1])
                adjacent_scores.append(min(1.0, score))

        current: List[str] = []
        current_scores: List[float] = []
        current_len = 0
        sentence_start = start_char

        for idx, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            if current:
                prev_score = adjacent_scores[idx - 1] if idx - 1 < len(adjacent_scores) else None
                semantic_break = (
                    prev_score is not None
                    and prev_score < semantic_threshold
                    and current_len >= min_chunk_size
                )
                size_break = current_len + sentence_len + 1 > chunk_size

                if semantic_break or size_break:
                    chunks.append(
                        make_semantic_chunk(
                            current,
                            "semantic",
                            sentence_start,
                            section_title,
                            current_scores,
                        )
                    )
                    overlap_sentences = []
                    if overlap > 0 and current and len(current[-1]) <= overlap:
                        overlap_sentences = [current[-1]]
                    current = overlap_sentences
                    current_scores = []
                    current_len = sum(len(item) for item in current)
                    sentence_start = start_char + max(0, block_text.find(sentence))
                elif prev_score is not None:
                    current_scores.append(prev_score)

            current.append(sentence)
            current_len += sentence_len + 1

        if current:
            chunks.append(
                make_semantic_chunk(
                    current,
                    "semantic",
                    sentence_start,
                    section_title,
                    current_scores,
                )
            )

    return merge_small_semantic_chunks(chunks, chunk_size, min_chunk_size)


def merge_small_semantic_chunks(chunks: List[Chunk], chunk_size: int, min_chunk_size: int) -> List[Chunk]:
    """
    语义切分会产生标题、短列表等碎片；在同一章节内做轻量合并，提升检索可读性。
    """
    if not chunks:
        return []

    merged: List[Chunk] = []
    buffer: Optional[Chunk] = None

    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue

        same_section = buffer.section_title == chunk.section_title
        can_merge = len(buffer.text) < min_chunk_size and same_section
        fits = len(buffer.text) + len(chunk.text) + 1 <= chunk_size

        if can_merge and fits:
            merged_text = f"{buffer.text}\n{chunk.text}".strip()
            buffer = Chunk(
                text=merged_text,
                strategy=buffer.strategy,
                start_char=buffer.start_char,
                end_char=chunk.end_char,
                section_title=buffer.section_title or chunk.section_title,
                semantic_score=buffer.semantic_score or chunk.semantic_score,
            )
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def calculate_chunk_metrics(
    chunks: List[Chunk],
    strategy: str,
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    min_chunk_size: int = 120,
    max_cohesion_chunks: int = 200,
) -> ChunkMetrics:
    if not chunks:
        return ChunkMetrics(strategy, 0, 0, 0, 0, 0, 0, 0, 0, None)

    lengths = [len(chunk.text) for chunk in chunks]
    sentence_end_pattern = re.compile(r"[。！？!?；;]$")
    punctuation_end_pattern = re.compile(r"[。！？!?；;，,：:]$")
    sentence_boundary_count = sum(1 for chunk in chunks if sentence_end_pattern.search(chunk.text.strip()))
    punctuation_end_count = sum(1 for chunk in chunks if punctuation_end_pattern.search(chunk.text.strip()))

    cohesion_scores = [chunk.semantic_score for chunk in chunks if chunk.semantic_score is not None]
    if not cohesion_scores and embed_func:
        cohesion_scores = []
        for chunk in chunks[:max_cohesion_chunks]:
            sentences = split_sentences(chunk.text)
            if len(sentences) < 2:
                continue
            embeddings = embed_func(sentences)
            pair_scores = [
                cosine_similarity(embeddings[idx], embeddings[idx + 1])
                for idx in range(len(sentences) - 1)
            ]
            if pair_scores:
                cohesion_scores.append(sum(pair_scores) / len(pair_scores))

    return ChunkMetrics(
        strategy=strategy,
        chunk_count=len(chunks),
        avg_chars=sum(lengths) / len(lengths),
        median_chars=statistics.median(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        short_chunk_ratio=sum(1 for length in lengths if length < min_chunk_size) / len(lengths),
        sentence_boundary_ratio=sentence_boundary_count / len(chunks),
        punctuation_end_ratio=punctuation_end_count / len(chunks),
        avg_semantic_cohesion=(sum(cohesion_scores) / len(cohesion_scores)) if cohesion_scores else None,
    )


def log_chunking_comparison(fixed_metrics: ChunkMetrics, semantic_metrics: ChunkMetrics):
    def fmt(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    logger.info("Chunking 策略质量对比:")
    logger.info(
        "  %-10s chunks=%4d avg=%6.1f median=%5.1f min=%3d max=%4d short=%5.1f%% sent_end=%5.1f%% punct_end=%5.1f%% cohesion=%s",
        fixed_metrics.strategy,
        fixed_metrics.chunk_count,
        fixed_metrics.avg_chars,
        fixed_metrics.median_chars,
        fixed_metrics.min_chars,
        fixed_metrics.max_chars,
        fixed_metrics.short_chunk_ratio * 100,
        fixed_metrics.sentence_boundary_ratio * 100,
        fixed_metrics.punctuation_end_ratio * 100,
        fmt(fixed_metrics.avg_semantic_cohesion),
    )
    logger.info(
        "  %-10s chunks=%4d avg=%6.1f median=%5.1f min=%3d max=%4d short=%5.1f%% sent_end=%5.1f%% punct_end=%5.1f%% cohesion=%s",
        semantic_metrics.strategy,
        semantic_metrics.chunk_count,
        semantic_metrics.avg_chars,
        semantic_metrics.median_chars,
        semantic_metrics.min_chars,
        semantic_metrics.max_chars,
        semantic_metrics.short_chunk_ratio * 100,
        semantic_metrics.sentence_boundary_ratio * 100,
        semantic_metrics.punctuation_end_ratio * 100,
        fmt(semantic_metrics.avg_semantic_cohesion),
    )


def build_chunks(
    docs: List[dict],
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> List[dict]:
    chunks_data = []
    all_fixed_chunks: List[Chunk] = []
    all_semantic_chunks: List[Chunk] = []

    for doc in docs:
        doc_uid = hashlib.sha1(doc["source"].encode("utf-8")).hexdigest()[:16]
        fixed_chunks = fixed_window_chunks(doc["content"], chunk_size, chunk_overlap)
        all_fixed_chunks.extend(fixed_chunks)

        if chunk_strategy == "semantic":
            selected_chunks = semantic_chunks(
                doc["content"],
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                embed_func=embed_func,
            )
            all_semantic_chunks.extend(selected_chunks)
        else:
            selected_chunks = fixed_chunks

        for i, chunk in enumerate(selected_chunks):
            audit_metadata = dict(doc.get("metadata") or {})
            chunks_data.append({
                "id": f"{doc_uid}_{chunk.strategy}_chunk_{i}",
                "text": chunk.text,
                "metadata": {
                    **audit_metadata,
                    "source": doc["source"],
                    "chunk_index": i,
                    "chunk_strategy": chunk.strategy,
                    "section_title": chunk.section_title,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "char_count": len(chunk.text),
                }
            })

    if chunk_strategy == "semantic":
        fixed_metrics = calculate_chunk_metrics(all_fixed_chunks, "fixed", embed_func)
        semantic_metrics = calculate_chunk_metrics(all_semantic_chunks, "semantic", embed_func)
        log_chunking_comparison(fixed_metrics, semantic_metrics)

    return chunks_data

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
                    try:
                        source = str(file_path.resolve().relative_to(PROJECT_ROOT))
                    except ValueError:
                        source = str(file_path.resolve())
                    canonical_url_match = re.search(r"^-\s*来源:\s*\[[^\]]*\]\((https?://[^)]+)\)", content, re.MULTILINE)
                    fetched_match = re.search(r"^-\s*抓取时间:\s*(.+)$", content, re.MULTILINE)
                    published_match = re.search(r"^-\s*(?:发布日期|发布时间):\s*(.+)$", content, re.MULTILINE)
                    effective_match = re.search(r"^-\s*(?:生效时间|数据截至|截至日期):\s*(.+)$", content, re.MULTILINE)
                    publisher_match = re.search(r"^-\s*发布者:\s*(.+)$", content, re.MULTILINE)
                    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                    canonical_url = canonical_url_match.group(1).strip() if canonical_url_match else ""
                    publisher = publisher_match.group(1).strip() if publisher_match else (urlparse(canonical_url).netloc or file_path.name)
                    document_hash = hashlib.sha256(re.sub(r"\s+", " ", content).strip().encode("utf-8")).hexdigest()
                    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    fetched_at = fetched_match.group(1).strip() if fetched_match else datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds")
                    source_type, reliability_tier = infer_document_source_quality(canonical_url)
                    documents.append({
                        "id": file_path.name,
                        "source": source,
                        "content": content,
                        "metadata": {
                            "canonical_url": canonical_url,
                            "publisher": publisher,
                            "source_type": source_type,
                            "reliability_tier": reliability_tier,
                            "published_at": published_match.group(1).strip() if published_match else "",
                            "effective_at": effective_match.group(1).strip() if effective_match else "",
                            "fetched_at": fetched_at,
                            "indexed_at": indexed_at,
                            "document_hash": document_hash,
                            "document_version": document_hash[:16],
                            "title": title_match.group(1).strip() if title_match else file_path.stem,
                        },
                    })
                logger.info(f"读取文档成功: {file_path.name}")
            except Exception as e:
                logger.error(f"无法读取文件 {file_path.name}: {e}")

    return documents


def load_embedding_model(model_name: str, device: Optional[str] = None):
    if SentenceTransformer is None:
        raise ImportError("请先安装依赖：pip install sentence-transformers")

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def get_or_create_collection(chroma_client: Any, collection_name: str):
    try:
        return chroma_client.get_collection(name=collection_name)
    except Exception:
        return chroma_client.create_collection(name=collection_name)


def remove_existing_source_chunks(collection: Any, sources: Sequence[str]) -> int:
    deleted_count = 0
    for source in sorted({source for source in sources if source}):
        try:
            existing = collection.get(where={"source": source}, include=[])
            existing_ids = existing.get("ids") or []
            if existing_ids:
                collection.delete(ids=existing_ids)
                deleted_count += len(existing_ids)
        except Exception as exc:
            logger.warning(f"删除旧片段失败，source={source}: {exc}")
    return deleted_count


def add_chunks_to_collection(collection: Any, embed_model: Any, chunks_data: List[dict]) -> int:
    ids = [item["id"] for item in chunks_data]
    documents = [item["text"] for item in chunks_data]
    metadatas = [item["metadata"] for item in chunks_data]
    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks_data)


def index_documents(
    docs: List[dict],
    db_dir: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str = "fixed",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset_collection: bool = False,
    device: Optional[str] = None,
) -> int:
    if chromadb is None or SentenceTransformer is None:
        raise ImportError("请先安装依赖：pip install chromadb sentence-transformers")

    if not docs:
        logger.warning("没有可入库的文档，跳过向量化。")
        return 0

    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在加载 Embedding 模型: {model_name}...")
    embed_model = load_embedding_model(model_name, device=device)
    embed_func = build_embedding_func(embed_model)

    logger.info(f"开始文本切分(Chunking)，策略: {chunk_strategy}...")
    chunks_data = build_chunks(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        embed_func=embed_func if chunk_strategy == "semantic" else None,
    )
    logger.info(f"总计切分出 {len(chunks_data)} 个片段。")
    if not chunks_data:
        logger.warning("所有文档切分后均为空，跳过写入。")
        return 0

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    if reset_collection:
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"已清理旧的 ChromaDB 集合: {collection_name}")
        except Exception:
            pass
        collection = chroma_client.create_collection(name=collection_name)
    else:
        collection = get_or_create_collection(chroma_client, collection_name)
        deleted_count = remove_existing_source_chunks(
            collection,
            [item["metadata"].get("source", "") for item in chunks_data],
        )
        if deleted_count:
            logger.info(f"已删除 {deleted_count} 个旧片段，准备写入最新内容。")

    logger.info("正在计算 Embedding 并录入 ChromaDB...")
    added_count = add_chunks_to_collection(collection, embed_model, chunks_data)
    logger.info(f"向量库写入完成，本次新增片段数: {added_count}")
    return added_count

def build_vector_db(
    data_dir: str, 
    db_dir: str, 
    model_name: str, 
    chunk_size: int, 
    chunk_overlap: int,
    chunk_strategy: str = "fixed",
):
    if chromadb is None or SentenceTransformer is None:
        raise ImportError("请先安装依赖：pip install chromadb sentence-transformers")

    data_path = Path(data_dir).expanduser().resolve()
    db_path = Path(db_dir).expanduser().resolve()
    
    logger.info("1. 开始加载文档数据...")
    docs = load_documents(data_path)
    if not docs:
        logger.warning(f"未在目录 {data_dir} 找到任何文档，建库流程结束。请放入 .txt 或 .md 文件后重试。")
        return
        
    logger.info(f"共加载 {len(docs)} 个文档。")

    logger.info("2. 开始构建向量库...")
    added_count = index_documents(
        docs=docs,
        db_dir=str(db_path),
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        collection_name=DEFAULT_COLLECTION_NAME,
        reset_collection=True,
    )

    logger.info(f"建库完毕！向量数据库已持久化保存至: {db_path}，本次写入 {added_count} 个片段")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG 本地向量数据库")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="知识库原始文档存储目录")
    parser.add_argument("--db-dir", type=str, default=str(DEFAULT_DB_DIR), help="持久化向量库存储目录")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Embedding 模型名称或路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="文本切片大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="文本切片重叠字数")
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        choices=["fixed", "semantic"],
        default="fixed",
        help="文本切分策略：fixed 为原始滑窗；semantic 为结构感知+embedding 语义切分并输出对比指标",
    )
    
    args = parser.parse_args()
    build_vector_db(
        args.data_dir,
        args.db_dir,
        args.model_name,
        args.chunk_size,
        args.chunk_overlap,
        args.chunk_strategy,
    )
