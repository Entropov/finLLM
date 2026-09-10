#!/usr/bin/env python3
"""Versioned audit records for financial agent trajectories.

The schema deliberately stores concise, verifiable reasoning artifacts instead
of private free-form chain-of-thought.  Every factual claim must point to an
evidence record or a reproducible calculation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


AUDIT_SCHEMA_VERSION = "2.0"
CLAIM_TYPES = {"fact", "calculation", "inference", "opinion"}
SOURCE_TYPES = {"regulator", "exchange", "issuer", "market_data", "research", "news", "internal", "unknown"}
RELIABILITY_TIERS = {"primary", "verified_secondary", "unverified_secondary", "internal", "unknown"}
FINANCIAL_TASK_TYPES = {
    "stock_analysis", "quant_strategy", "financial_report", "sentiment_analysis", "financial_qa", "risk_assessment",
}
_CITATION_RE = re.compile(r"\[\s*(E[0-9a-f]{10,64})\s*\]", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ABSTENTION_TERMS = ("无法确认", "资料不足", "证据不足", "未检索到", "需要补充")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    chinese_date = re.fullmatch(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?", text)
    if chinese_date:
        year, month, day = chinese_date.groups()
        text = f"{year}-{int(month):02d}-{int(day or 1):02d}T00:00:00+08:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text += "T00:00:00+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonicalize_url(value: str) -> str:
    value = (value or "").strip()
    if not value.startswith(("http://", "https://")):
        return ""
    parts = urlsplit(value)
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in {"spm", "from", "source"}
        )
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def content_digest(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def strip_thinking(text: str) -> str:
    """Remove private/free-form reasoning before audit persistence and scoring."""
    return _THINK_RE.sub("", text or "").strip()


def infer_source_quality(source_uri: str, metadata: dict[str, Any]) -> tuple[str, str, str]:
    explicit_type = str(metadata.get("source_type", "")).strip()
    explicit_tier = str(metadata.get("reliability_tier", "")).strip()
    publisher = str(metadata.get("publisher", "")).strip()
    host = urlsplit(canonicalize_url(source_uri)).netloc.lower()

    regulator_hosts = ("gov.cn", "pbc.gov.cn", "csrc.gov.cn", "stats.gov.cn")
    exchange_hosts = ("sse.com.cn", "szse.cn", "hkex.com.hk")

    def trusted_host(domains: tuple[str, ...]) -> bool:
        return any(host == domain or host.endswith(f".{domain}") for domain in domains)

    if explicit_type in SOURCE_TYPES:
        source_type = explicit_type
    elif trusted_host(regulator_hosts):
        source_type = "regulator"
    elif trusted_host(exchange_hosts):
        source_type = "exchange"
    elif host:
        source_type = "news"
    elif source_uri:
        source_type = "internal"
    else:
        source_type = "unknown"

    if explicit_tier in RELIABILITY_TIERS:
        tier = explicit_tier
    elif source_type in {"regulator", "exchange", "issuer", "market_data"}:
        tier = "primary"
    elif source_type == "internal":
        tier = "internal"
    elif source_type in {"research", "news"}:
        tier = "unverified_secondary"
    else:
        tier = "unknown"
    return source_type, tier, publisher or host or Path(source_uri).name or "unknown"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_uri: str
    exact_quote: str
    content_hash: str
    fetched_at: str
    canonical_url: str = ""
    publisher: str = "unknown"
    source_type: str = "unknown"
    reliability_tier: str = "unknown"
    published_at: str = ""
    effective_at: str = ""
    document_version: str = ""
    title: str = ""
    chunk_index: int = -1
    start_char: int = -1
    end_char: int = -1
    retrieval_query: str = ""
    retrieval_rank: int = -1
    point_in_time_available: bool = False
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_errors(self) -> list[str]:
        errors = []
        if not re.fullmatch(r"E[0-9A-Fa-f]{10,64}", self.evidence_id):
            errors.append("invalid_evidence_id")
        if not self.source_uri:
            errors.append("missing_source_uri")
        if self.canonical_url and self.canonical_url != canonicalize_url(self.canonical_url):
            errors.append("noncanonical_url")
        if not self.exact_quote.strip():
            errors.append("missing_exact_quote")
        if self.content_hash != content_digest(self.exact_quote):
            errors.append("content_hash_mismatch")
        if parse_timestamp(self.fetched_at) is None:
            errors.append("invalid_fetched_at")
        if self.published_at and parse_timestamp(self.published_at) is None:
            errors.append("invalid_published_at")
        if self.effective_at and parse_timestamp(self.effective_at) is None:
            errors.append("invalid_effective_at")
        if not self.document_version.strip():
            errors.append("missing_document_version")
        if self.source_type not in SOURCE_TYPES:
            errors.append("invalid_source_type")
        if self.reliability_tier not in RELIABILITY_TIERS:
            errors.append("invalid_reliability_tier")
        if self.start_char >= 0 and self.end_char >= 0 and self.end_char < self.start_char:
            errors.append("invalid_chunk_offsets")
        return errors

    @classmethod
    def from_retrieved_doc(
        cls,
        doc: dict[str, Any],
        rank: int,
        retrieval_query: str = "",
        observed_at: Optional[str] = None,
    ) -> "EvidenceRecord":
        metadata = dict(doc.get("metadata") or {})
        quote = str(doc.get("content") or doc.get("text") or "").strip()
        source_uri = str(metadata.get("canonical_url") or metadata.get("url") or metadata.get("source") or "").strip()
        canonical_url = canonicalize_url(str(metadata.get("canonical_url") or metadata.get("url") or source_uri))
        source_type, tier, publisher = infer_source_quality(source_uri, metadata)
        digest = content_digest(quote)
        start_char = int(metadata.get("start_char", -1) or -1)
        end_char = int(metadata.get("end_char", -1) or -1)
        identity = "|".join((canonical_url or source_uri, digest, str(start_char), str(end_char)))
        evidence_id = "E" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
        fetched_at = str(metadata.get("fetched_at") or metadata.get("indexed_at") or observed_at or utc_now())
        published_at = str(metadata.get("published_at") or "")
        effective_at = str(metadata.get("effective_at") or metadata.get("as_of") or published_at)
        available_at = parse_timestamp(fetched_at)
        effective_dt = parse_timestamp(effective_at)
        return cls(
            evidence_id=evidence_id,
            source_uri=source_uri,
            canonical_url=canonical_url,
            publisher=publisher,
            source_type=source_type,
            reliability_tier=tier,
            title=str(metadata.get("title") or metadata.get("section_title") or ""),
            exact_quote=quote,
            content_hash=digest,
            document_version=str(metadata.get("document_version") or metadata.get("document_hash") or digest[:16]),
            published_at=published_at,
            effective_at=effective_at,
            fetched_at=fetched_at,
            chunk_index=int(metadata.get("chunk_index", -1) or 0),
            start_char=start_char,
            end_char=end_char,
            retrieval_query=retrieval_query,
            retrieval_rank=rank,
            point_in_time_available=bool(available_at and (not effective_dt or effective_dt <= available_at)),
        )


@dataclass(frozen=True)
class CalculationRecord:
    calculation_id: str
    expression: str
    inputs: dict[str, Any]
    result: Any
    unit: str = ""
    evidence_ids: tuple[str, ...] = ()
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload

    def validation_errors(self, valid_evidence_ids: set[str]) -> list[str]:
        errors = []
        if not self.calculation_id:
            errors.append("missing_calculation_id")
        if not self.evidence_ids:
            errors.append("missing_calculation_evidence")
        if not set(self.evidence_ids) <= valid_evidence_ids:
            errors.append("unknown_evidence_reference")
        try:
            if not self.expression.strip() or len(self.expression) > 512:
                raise ValueError("invalid expression length")
            parsed = ast.parse(self.expression, mode="eval")
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
            )
            nodes = list(ast.walk(parsed))
            if len(nodes) > 64 or any(not isinstance(node, allowed_nodes) for node in nodes):
                raise ValueError("unsupported expression")
            numeric_inputs = {
                name: float(value)
                for name, value in self.inputs.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if set(self.inputs) != set(numeric_inputs) or not all(math.isfinite(value) for value in numeric_inputs.values()):
                raise ValueError("non-numeric input")
            referenced_names = {node.id for node in nodes if isinstance(node, ast.Name)}
            if referenced_names != set(numeric_inputs):
                raise ValueError("calculation inputs do not match expression")
            calculated = _evaluate_calculation_node(parsed.body, numeric_inputs)
            expected = float(self.result)
            tolerance = 1e-6 * max(1.0, abs(expected))
            if not math.isfinite(calculated) or not math.isfinite(expected) or abs(calculated - expected) > tolerance:
                errors.append("calculation_result_mismatch")
        except (ArithmeticError, SyntaxError, TypeError, ValueError):
            errors.append("invalid_calculation")
        return errors


def _evaluate_calculation_node(node: ast.AST, inputs: dict[str, float]) -> float:
    """Evaluate the small arithmetic grammar used by CalculationRecord."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        value = float(node.value)
    elif isinstance(node, ast.Name):
        if node.id not in inputs:
            raise ValueError("unknown input")
        value = inputs[node.id]
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_calculation_node(node.operand, inputs)
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp):
        left = _evaluate_calculation_node(node.left, inputs)
        right = _evaluate_calculation_node(node.right, inputs)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            value = left / right
        elif isinstance(node.op, ast.Mod):
            value = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("exponent out of range")
            value = left ** right
        else:
            raise ValueError("unsupported operator")
    else:
        raise ValueError("unsupported expression")
    value = float(value)
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("calculation out of range")
    return value


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    statement: str
    claim_type: str
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    calculation_id: str = ""
    confidence: float = 0.0
    as_of: str = ""
    assumptions: tuple[str, ...] = ()
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        payload["contradicting_evidence_ids"] = list(self.contradicting_evidence_ids)
        payload["assumptions"] = list(self.assumptions)
        return payload

    def validation_errors(self, valid_evidence_ids: set[str], valid_calculation_ids: set[str]) -> list[str]:
        errors = []
        if not self.claim_id.strip():
            errors.append("missing_claim_id")
        if self.claim_type not in CLAIM_TYPES:
            errors.append("invalid_claim_type")
        if not self.statement.strip():
            errors.append("empty_claim")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("invalid_confidence")
        cited = set(self.supporting_evidence_ids) | set(self.contradicting_evidence_ids)
        if not cited <= valid_evidence_ids:
            errors.append("unknown_evidence_reference")
        if self.calculation_id and self.calculation_id not in valid_calculation_ids:
            errors.append("unknown_calculation_reference")
        if self.claim_type in {"fact", "calculation", "inference"} and not self.supporting_evidence_ids and not self.calculation_id:
            errors.append("unsupported_claim")
        return errors


@dataclass(frozen=True)
class TrajectoryEvent:
    step_id: int
    timestamp: str
    node: str
    action: str
    action_args: dict[str, Any] = field(default_factory=dict)
    observation_ids: tuple[str, ...] = ()
    verifier_scores: dict[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observation_ids"] = list(self.observation_ids)
        payload["errors"] = list(self.errors)
        return payload

    def validation_errors(self) -> list[str]:
        errors = []
        if self.step_id <= 0:
            errors.append("invalid_step_id")
        if parse_timestamp(self.timestamp) is None:
            errors.append("invalid_event_timestamp")
        if not self.node.strip():
            errors.append("missing_event_node")
        if not self.action.strip():
            errors.append("missing_event_action")
        if any(not re.fullmatch(r"E[0-9A-Fa-f]{10,64}", item) for item in self.observation_ids):
            errors.append("invalid_observation_id")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in self.verifier_scores.values()
        ):
            errors.append("invalid_verifier_score")
        return errors


@dataclass
class AuditEnvelope:
    query_id: str
    query: str
    task_type: str
    request_as_of: str
    final_answer: str
    evidence: list[EvidenceRecord] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    calculations: list[CalculationRecord] = field(default_factory=list)
    trajectory: list[TrajectoryEvent] = field(default_factory=list)
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "query": self.query,
            "task_type": self.task_type,
            "request_as_of": self.request_as_of,
            "final_answer": self.final_answer,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "calculations": [item.to_dict() for item in self.calculations],
            "trajectory": [item.to_dict() for item in self.trajectory],
        }

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        evidence_ids = {item.evidence_id for item in self.evidence}
        calculation_ids = {item.calculation_id for item in self.calculations}
        claim_ids = {item.claim_id for item in self.claims}
        if not self.query_id.strip():
            errors.append("missing_query_id")
        if not self.query.strip():
            errors.append("missing_query")
        if self.task_type not in FINANCIAL_TASK_TYPES:
            errors.append("invalid_task_type")
        if parse_timestamp(self.request_as_of) is None:
            errors.append("invalid_request_as_of")
        if not self.final_answer.strip():
            errors.append("missing_final_answer")
        if strip_thinking(self.final_answer) != self.final_answer.strip():
            errors.append("persisted_thinking_trace")
        if len(evidence_ids) != len(self.evidence):
            errors.append("duplicate_evidence_id")
        if len(calculation_ids) != len(self.calculations):
            errors.append("duplicate_calculation_id")
        if len(claim_ids) != len(self.claims):
            errors.append("duplicate_claim_id")
        for item in self.evidence:
            errors.extend(f"{item.evidence_id}:{error}" for error in item.validation_errors())
        for item in self.calculations:
            errors.extend(f"{item.calculation_id}:{error}" for error in item.validation_errors(evidence_ids))
        for claim in self.claims:
            errors.extend(f"{claim.claim_id}:{error}" for error in claim.validation_errors(evidence_ids, calculation_ids))
        for event in self.trajectory:
            errors.extend(f"step_{event.step_id}:{error}" for error in event.validation_errors())
        if self.trajectory and [event.step_id for event in self.trajectory] != list(range(1, len(self.trajectory) + 1)):
            errors.append("non_sequential_trajectory_steps")
        answer_citations = extract_citation_ids(self.final_answer)
        if not answer_citations <= evidence_ids:
            errors.append("answer_contains_unknown_citation")
        return sorted(set(errors))


def extract_citation_ids(text: str) -> set[str]:
    return {match.upper() for match in _CITATION_RE.findall(text or "")}


def normalize_evidence(docs: Iterable[dict[str, Any]], query: str = "", observed_at: Optional[str] = None) -> list[EvidenceRecord]:
    records = []
    seen = set()
    for rank, doc in enumerate(docs, start=1):
        record = EvidenceRecord.from_retrieved_doc(doc, rank, query, observed_at)
        if record.evidence_id in seen:
            continue
        seen.add(record.evidence_id)
        records.append(record)
    return records


def infer_claim_type(statement: str) -> str:
    if re.match(r"^观点\s*[:：]", statement.strip()):
        return "opinion"
    if any(term in statement for term in _ABSTENTION_TERMS):
        return "opinion"
    if re.search(r"\d+(?:\.\d+)?%|亿元|万元|公式|=", statement):
        return "calculation" if re.search(r"计算|公式|=", statement) else "fact"
    if any(term in statement for term in ("可能", "预计", "推断", "表明", "意味着")):
        return "inference"
    if any(term in statement for term in ("建议", "观点", "认为")):
        return "opinion"
    return "fact"


def build_claims_from_answer(answer: str, request_as_of: str = "") -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    answer_without_code = re.sub(r"```.*?```", "", strip_thinking(answer), flags=re.DOTALL)
    for raw_line in re.split(r"[\n]+", answer_without_code):
        is_heading = bool(re.match(r"^\s*(?:>\s*)?#{1,6}\s+", raw_line))
        line = re.sub(r"^\s*>?\s*(?:#{1,6}\s*)?(?:[-*]|\d+[.、])?\s*", "", raw_line).strip()
        plain_line = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", line)
        plain_line = re.sub(r"[*`~]", "", plain_line).strip()
        if (
            len(plain_line) < 8
            or is_heading
            or plain_line.startswith(
                ("基于检索资料", "以下基于", "以下是", "参考资料", "免责声明", "请结合最新数据", "所有事实")
            )
            or "不构成投资建议" in plain_line
            or "不作收益保证" in plain_line
            or (plain_line.endswith(("：", ":")) and not extract_citation_ids(plain_line))
        ):
            continue
        line_is_opinion = bool(re.match(r"^观点\s*[:：]", plain_line))
        citation_pattern = r"(?:\[\s*E[0-9a-f]{10,64}\s*\])+"
        plain_line = re.sub(
            rf"([。！？!?；;])\s*({citation_pattern})",
            r"\2\1",
            plain_line,
            flags=re.IGNORECASE,
        )
        statements = [item.strip() for item in re.split(r"(?<=[。！？!?；;])", plain_line) if len(item.strip()) >= 8]
        for raw_statement_part in statements:
            citations = tuple(sorted(extract_citation_ids(raw_statement_part)))
            statement_part = _CITATION_RE.sub("", raw_statement_part).strip(" :-")
            statement_part = re.sub(r"（来源:.*?；获取时间:.*?）\s*$", "", statement_part).strip()
            if len(statement_part) < 8 or "不构成投资建议" in statement_part:
                continue
            claim_type = "opinion" if line_is_opinion else infer_claim_type(statement_part)
            claim_hash = hashlib.sha256(f"{statement_part}|{','.join(citations)}".encode("utf-8")).hexdigest()[:16]
            claims.append(
                ClaimRecord(
                    claim_id=f"C{claim_hash}",
                    statement=statement_part,
                    claim_type=claim_type,
                    supporting_evidence_ids=citations,
                    confidence=0.8 if citations else 0.4,
                    as_of=request_as_of,
                )
            )
    return claims


def load_audit_envelope(payload: dict[str, Any]) -> AuditEnvelope:
    """Load a serialized envelope and reject unknown schema versions."""
    version = str(payload.get("schema_version", ""))
    if version != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported audit schema version: {version or 'missing'}")
    evidence = [EvidenceRecord(**item) for item in payload.get("evidence", [])]
    claims = []
    for item in payload.get("claims", []):
        item = dict(item)
        for key in ("supporting_evidence_ids", "contradicting_evidence_ids", "assumptions"):
            item[key] = tuple(item.get(key, ()))
        claims.append(ClaimRecord(**item))
    calculations = []
    for item in payload.get("calculations", []):
        item = dict(item)
        item["evidence_ids"] = tuple(item.get("evidence_ids", ()))
        calculations.append(CalculationRecord(**item))
    trajectory = []
    for item in payload.get("trajectory", []):
        item = dict(item)
        item["observation_ids"] = tuple(item.get("observation_ids", ()))
        item["errors"] = tuple(item.get("errors", ()))
        trajectory.append(TrajectoryEvent(**item))
    return AuditEnvelope(
        query_id=str(payload.get("query_id", "")),
        query=str(payload.get("query", "")),
        task_type=str(payload.get("task_type", "")),
        request_as_of=str(payload.get("request_as_of", "")),
        final_answer=str(payload.get("final_answer", "")),
        evidence=evidence,
        claims=claims,
        calculations=calculations,
        trajectory=trajectory,
    )


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
