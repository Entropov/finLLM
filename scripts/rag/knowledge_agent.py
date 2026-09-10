#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据用户问题实时进行网络检索，并将抓取结果保存为 Markdown 知识文档后增量写入向量库。

工作流:
  1. 根据用户问题调用网页搜索。
  2. 抓取搜索结果页正文并整理为 Markdown。
  3. 将 Markdown 文件保存到 data/knowledge/web。
  4. 调用 build_vector_db.index_documents 进行增量入库。
  5. 刷新 retriever，并返回最新检索结果。

示例:
  python scripts/rag/knowledge_agent.py --query "宁德时代最新财报亮点"
"""

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from lxml import html

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from scripts.rag.build_vector_db import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_DB_DIR,
    DEFAULT_MODEL_NAME,
    PROJECT_ROOT,
    infer_document_source_quality,
    index_documents,
)
from scripts.rag.retriever import RAGRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_WEB_KNOWLEDGE_DIR = DEFAULT_DATA_DIR / "web"
DEFAULT_SEARCH_URL = "https://html.duckduckgo.com/html/"
DEFAULT_TIMEOUT = 20

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class KnowledgeDocument:
    title: str
    url: str
    query: str
    fetched_at: str
    content: str
    snippet: str = ""


def slugify(value: str, max_length: int = 80) -> str:
    value = value.lower().strip()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:max_length] or "knowledge"


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_result_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])
    return raw_url


class WebKnowledgeAgent:
    def __init__(
        self,
        knowledge_dir: Path = DEFAULT_WEB_KNOWLEDGE_DIR,
        db_dir: Path = DEFAULT_DB_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "semantic",
        device: str = "cpu",
        timeout: int = DEFAULT_TIMEOUT,
        search_url: str = DEFAULT_SEARCH_URL,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.device = device
        self.timeout = timeout
        self.search_url = search_url

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        self.retriever = RAGRetriever(
            db_dir=str(self.db_dir),
            model_name=self.model_name,
            collection_name=self.collection_name,
            device=self.device,
        )

    def search(self, query: str, top_n: int = 5) -> List[SearchResult]:
        logger.info(f"开始联网搜索: {query}")
        response = self.session.get(
            self.search_url,
            params={"q": query},
            timeout=self.timeout,
        )
        response.raise_for_status()

        tree = html.fromstring(response.text)
        results: List[SearchResult] = []
        for node in tree.xpath("//a[contains(@class, 'result__a')]"):
            url = resolve_result_url((node.get("href") or "").strip())
            title = clean_text("".join(node.itertext()))
            if not url or not title:
                continue

            container = node.getparent().getparent() if node.getparent() is not None and node.getparent().getparent() is not None else None
            snippet = ""
            if container is not None:
                snippet_nodes = container.xpath(".//*[contains(@class, 'result__snippet')]")
                if snippet_nodes:
                    snippet = clean_text(" ".join("".join(item.itertext()) for item in snippet_nodes))

            results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= top_n:
                break

        logger.info(f"搜索完成，获得 {len(results)} 条候选结果")
        return results

    def fetch_page(self, result: SearchResult) -> Optional[KnowledgeDocument]:
        logger.info(f"抓取网页正文: {result.url}")
        response = self.session.get(result.url, timeout=self.timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            logger.warning(f"跳过非 HTML 页面: {result.url} ({content_type})")
            return None

        tree = html.fromstring(response.content)
        for bad in tree.xpath("//script|//style|//noscript|//svg|//form|//footer|//nav"):
            bad.drop_tree()

        title = clean_text("".join(tree.xpath("//title/text()"))) or result.title
        main_nodes = tree.xpath(
            "//main | //article | //*[@id='content'] | "
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')] | //body"
        )
        if not main_nodes:
            return None

        block_texts: List[str] = []
        for node in main_nodes[:1]:
            texts = node.xpath(
                ".//h1//text() | .//h2//text() | .//h3//text() | .//p//text() | .//li//text() | .//blockquote//text()"
            )
            for text in texts:
                cleaned = clean_text(text)
                if len(cleaned) >= 20:
                    block_texts.append(cleaned)

        deduped_blocks: List[str] = []
        seen = set()
        for block in block_texts:
            key = block[:120]
            if key in seen:
                continue
            seen.add(key)
            deduped_blocks.append(block)

        content = "\n\n".join(deduped_blocks)
        content = clean_text(content)
        if len(content) < 200:
            logger.warning(f"正文过短，跳过: {result.url}")
            return None

        return KnowledgeDocument(
            title=title,
            url=result.url,
            query="",
            fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            content=content,
            snippet=result.snippet,
        )

    def render_markdown(self, doc: KnowledgeDocument) -> str:
        domain = urlparse(doc.url).netloc or "unknown"
        summary = doc.snippet or doc.content[:180]
        summary = clean_text(summary).replace("\n", " ")
        body_lines = []
        for paragraph in doc.content.split("\n\n"):
            cleaned = clean_text(paragraph)
            if cleaned:
                body_lines.append(cleaned)

        body = "\n\n".join(body_lines)
        return (
            f"# {doc.title}\n\n"
            f"- 查询: {doc.query}\n"
            f"- 来源: [{domain}]({doc.url})\n"
            f"- 抓取时间: {doc.fetched_at}\n"
            f"- 摘要: {summary}\n\n"
            "## 正文\n\n"
            f"{body}\n"
        )

    def save_markdown(self, doc: KnowledgeDocument) -> Path:
        host = urlparse(doc.url).netloc or "web"
        host_slug = slugify(host, max_length=30)
        title_slug = slugify(doc.title, max_length=40)
        url_hash = hashlib.sha1(doc.url.encode("utf-8")).hexdigest()[:10]
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{host_slug}_{title_slug}_{url_hash}.md"
        output_path = self.knowledge_dir / file_name
        markdown = self.render_markdown(doc)
        output_path.write_text(markdown, encoding="utf-8")
        logger.info(f"Markdown 已保存: {output_path}")
        return output_path

    def _build_index_doc(self, markdown_path: Path) -> Dict[str, Any]:
        try:
            relative_source = str(markdown_path.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_source = str(markdown_path)

        content = markdown_path.read_text(encoding="utf-8")
        url_match = re.search(r"^-\s*来源:\s*\[[^\]]*\]\((https?://[^)]+)\)", content, re.MULTILINE)
        fetched_match = re.search(r"^-\s*抓取时间:\s*(.+)$", content, re.MULTILINE)
        published_match = re.search(r"^-\s*(?:发布日期|发布时间):\s*(.+)$", content, re.MULTILINE)
        url = url_match.group(1).strip() if url_match else ""
        source_type, reliability_tier = infer_document_source_quality(url)
        document_hash = hashlib.sha256(re.sub(r"\s+", " ", content).strip().encode("utf-8")).hexdigest()
        return {
            "id": markdown_path.name,
            "source": relative_source,
            "content": content,
            "metadata": {
                "canonical_url": url,
                "publisher": urlparse(url).netloc or "unknown",
                "source_type": source_type,
                "reliability_tier": reliability_tier,
                "published_at": published_match.group(1).strip() if published_match else "",
                "effective_at": published_match.group(1).strip() if published_match else "",
                "fetched_at": fetched_match.group(1).strip() if fetched_match else "",
                "document_hash": document_hash,
                "document_version": document_hash[:16],
            },
        }

    def index_markdowns(self, markdown_paths: List[Path]) -> int:
        docs = [self._build_index_doc(path) for path in markdown_paths]
        if not docs:
            return 0

        return index_documents(
            docs=docs,
            db_dir=str(self.db_dir),
            model_name=self.model_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_strategy=self.chunk_strategy,
            collection_name=self.collection_name,
            reset_collection=False,
            device=self.device,
        )

    def collect(self, query: str, search_top_n: int = 5, fetch_top_n: int = 3) -> List[Path]:
        results = self.search(query, top_n=search_top_n)
        markdown_paths: List[Path] = []
        for result in results[:fetch_top_n]:
            try:
                doc = self.fetch_page(result)
            except Exception as exc:
                logger.warning(f"抓取失败，url={result.url}: {exc}")
                continue

            if doc is None:
                continue

            doc.query = query
            markdown_path = self.save_markdown(doc)
            markdown_paths.append(markdown_path)

        if markdown_paths:
            self.index_markdowns(markdown_paths)
            self.retriever.refresh()
        return markdown_paths

    def run(self, query: str, search_top_n: int = 5, fetch_top_n: int = 3, rag_top_k: int = 5) -> Dict[str, object]:
        markdown_paths = self.collect(query, search_top_n=search_top_n, fetch_top_n=fetch_top_n)
        retrieved = self.retriever.retrieve(query, top_k=rag_top_k) if self.retriever.is_ready() else []
        return {
            "query": query,
            "saved_files": markdown_paths,
            "retrieved": retrieved,
        }


def format_retrieval_output(items: List[Dict]) -> str:
    if not items:
        return "未检索到可用知识片段。"

    lines = []
    for idx, item in enumerate(items, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source", "")
        section = metadata.get("section_title", "")
        preview = clean_text(item.get("content", ""))[:220]
        if section:
            lines.append(f"{idx}. [{source}] {section} :: {preview}")
        else:
            lines.append(f"{idx}. [{source}] {preview}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="实时联网抓取知识并增量写入本地 knowledge 库")
    parser.add_argument("--query", type=str, required=True, help="用户问题")
    parser.add_argument("--knowledge-dir", type=str, default=str(DEFAULT_WEB_KNOWLEDGE_DIR), help="Markdown 知识落盘目录")
    parser.add_argument("--db-dir", type=str, default=str(DEFAULT_DB_DIR), help="向量库存储目录")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Embedding 模型")
    parser.add_argument("--device", type=str, default="cpu", help="Embedding 运行设备")
    parser.add_argument("--search-top-n", type=int, default=5, help="搜索返回候选数")
    parser.add_argument("--fetch-top-n", type=int, default=3, help="实际抓取网页数")
    parser.add_argument("--rag-top-k", type=int, default=5, help="最终返回的知识片段数")
    parser.add_argument("--chunk-size", type=int, default=500, help="文本切片大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="文本切片重叠字数")
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        choices=["fixed", "semantic"],
        default="semantic",
        help="增量入库的切分策略",
    )
    args = parser.parse_args()

    agent = WebKnowledgeAgent(
        knowledge_dir=Path(args.knowledge_dir),
        db_dir=Path(args.db_dir),
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy,
        device=args.device,
    )
    result = agent.run(
        query=args.query,
        search_top_n=args.search_top_n,
        fetch_top_n=args.fetch_top_n,
        rag_top_k=args.rag_top_k,
    )

    print(f"查询: {result['query']}")
    print("\n保存的 Markdown 文件:")
    saved_files = result["saved_files"]
    if saved_files:
        for path in saved_files:
            print(f"- {path}")
    else:
        print("- 本次没有成功保存新文档")

    print("\n最新检索结果:")
    print(format_retrieval_output(result["retrieved"]))


if __name__ == "__main__":
    main()
