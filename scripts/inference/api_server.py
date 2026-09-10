#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 推理服务

基于 FastAPI 构建 OpenAI API 兼容的推理服务:
  - POST /v1/chat/completions  (对话补全，支持流式)
  - GET  /v1/models            (模型列表)
  - GET  /health               (健康检查)

支持:
  - 加载合并模型或基座模型+LoRA adapter
  - 4-bit 量化推理
  - 流式输出 (SSE)
  - 可被 LangChain、OpenAI SDK 等直接调用

用法:
  python scripts/inference/api_server.py
  python scripts/inference/api_server.py --model-path saves/qwen3-8b/merged --port 8000
"""

import json
import asyncio
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Literal, Optional, List
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    # 尝试加载 RAG 检索器
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from scripts.rag.retriever import RAGRetriever
    from scripts.rag.knowledge_agent import WebKnowledgeAgent
    from scripts.rag.agentic_rag import AgenticRAGPipeline, load_agentic_rag_config
    from scripts.inference.chat_template import apply_chat_template
    from scripts.inference.vllm_backend import (
        VLLMAPIEngine,
        VLLMEngineConfig,
        validate_vllm_model_options,
    )
except ImportError:
    RAGRetriever = None
    WebKnowledgeAgent = None
    AgenticRAGPipeline = None
    load_agentic_rag_config = None
    from scripts.inference.chat_template import apply_chat_template
    from scripts.inference.vllm_backend import (
        VLLMAPIEngine,
        VLLMEngineConfig,
        validate_vllm_model_options,
    )

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

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="Fin-Instruct API",
    description="Fin-Instruct 金融大模型推理服务 (OpenAI 兼容)",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 全局变量
# ============================================================
model = None
tokenizer = None
backend_name = "transformers"
model_name = "fin-instruct-qwen3-8b"
rag_retriever = None
web_knowledge_agent = None
web_knowledge_enabled_by_default = False
rag_mode_default = "basic"
agentic_rag_pipeline = None
agentic_rag_config_path = None


# ============================================================
# 请求/响应模型定义（OpenAI API 格式）
# ============================================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="fin-instruct", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False, description="是否流式输出")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    rag_mode: Literal["basic", "agentic"] = Field(default="basic", description="RAG 模式: basic 或 agentic")
    task_type: Optional[
        Literal[
            "stock_analysis",
            "quant_strategy",
            "financial_report",
            "sentiment_analysis",
            "financial_qa",
            "risk_assessment",
        ]
    ] = Field(default=None, description="可选的审计任务类型；省略时自动分类")
    rag_top_k: int = Field(default=3, ge=1, le=10, description="本地知识库检索条数")
    enable_web_knowledge: Optional[bool] = Field(default=None, description="是否启用实时联网补充知识")
    web_search_top_n: int = Field(default=5, ge=1, le=10, description="联网搜索候选数")
    web_fetch_top_n: int = Field(default=3, ge=1, le=10, description="实际抓取网页数")
    return_audit: bool = Field(default=False, description="是否返回 Agentic 审计摘要")
    enable_answer_guard: bool = Field(
        default=False,
        description="审计失败时用证据摘录式保守答案替换，仅用于非流式 Agentic 请求",
    )


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict
    audit_summary: Optional[dict] = None


# ============================================================
# 模型加载
# ============================================================
def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """加载模型和分词器。"""
    global model, tokenizer, model_name, backend_name

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"正在加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        logger.info(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    model_name = Path(model_path).name or "fin-instruct"
    backend_name = "transformers"
    logger.info("模型加载完成！")


def load_vllm_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    quantization: Optional[str] = None,
    load_format: Optional[str] = None,
):
    """加载 vLLM 异步推理引擎和分词器。"""
    global model, tokenizer, model_name, backend_name

    from transformers import AutoTokenizer

    validate_vllm_model_options(adapter_path)
    logger.info(f"正在加载 vLLM 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = VLLMAPIEngine(
        VLLMEngineConfig(
            model_path=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            load_format=load_format,
        )
    )
    model_name = Path(model_path).name or "fin-instruct"
    backend_name = "vllm"
    logger.info("vLLM 模型加载完成！")

def init_rag_system(enable_rag: bool, db_dir: Optional[str] = None):
    global rag_retriever
    if not enable_rag:
        return
        
    if RAGRetriever:
        try:
            logger.info("初始化 RAG 检索器...")
            kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            if db_dir:
                kwargs["db_dir"] = db_dir
                
            rag_retriever = RAGRetriever(**kwargs)
            if not rag_retriever.is_ready():
                 logger.warning("RAG 检索器初始化完成，但底层数据库无数据或未就绪。")
        except Exception as e:
            logger.error(f"初始化 RAG 检索器失败: {e}")
            rag_retriever = None
    else:
        logger.warning("未找到 RAGRetriever 模块。")


def init_agentic_rag_system(
    rag_mode: str = "basic",
    config_path: Optional[str] = None,
):
    global agentic_rag_pipeline, rag_mode_default, agentic_rag_config_path
    rag_mode_default = rag_mode
    agentic_rag_config_path = config_path
    if rag_mode != "agentic":
        return
    if AgenticRAGPipeline is None:
        logger.warning("未找到 AgenticRAGPipeline 模块，agentic RAG 将回退到 basic RAG。")
        return
    try:
        logger.info("初始化 Agentic RAG pipeline...")
        agentic_rag_pipeline = AgenticRAGPipeline(
            retriever=rag_retriever,
            web_agent=web_knowledge_agent,
            config_path=config_path,
        )
    except Exception as e:
        logger.warning(f"初始化 Agentic RAG 失败，自动回退 basic RAG: {e}")
        agentic_rag_pipeline = None


def init_web_knowledge_system(
    enable_web_knowledge: bool,
    db_dir: Optional[str] = None,
    knowledge_dir: Optional[str] = None,
):
    global web_knowledge_agent, web_knowledge_enabled_by_default
    web_knowledge_enabled_by_default = enable_web_knowledge
    if not enable_web_knowledge:
        return

    if WebKnowledgeAgent is None:
        logger.warning("未找到 WebKnowledgeAgent 模块。")
        return

    try:
        logger.info("初始化实时联网知识代理...")
        kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        if db_dir:
            kwargs["db_dir"] = db_dir
        if knowledge_dir:
            kwargs["knowledge_dir"] = knowledge_dir
        web_knowledge_agent = WebKnowledgeAgent(**kwargs)
    except Exception as e:
        logger.error(f"初始化实时联网知识代理失败: {e}")
        web_knowledge_agent = None


def enrich_knowledge_before_generation(
    messages: List[dict],
    rag_mode: str,
    rag_top_k: int,
    enable_web_knowledge: Optional[bool],
    web_search_top_n: int,
    web_fetch_top_n: int,
    task_type: Optional[str] = None,
    enable_answer_guard: bool = False,
    return_agent_state: bool = False,
):
    global rag_retriever, web_knowledge_agent, web_knowledge_enabled_by_default
    global agentic_rag_pipeline

    last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
    sys_msg = next((msg for msg in messages if msg["role"] == "system"), None)
    if last_user_msg is None:
        return (messages, None) if return_agent_state else messages

    should_use_web_knowledge = (
        web_knowledge_enabled_by_default if enable_web_knowledge is None else enable_web_knowledge
    )
    requested_mode = rag_mode or rag_mode_default
    sys_content = sys_msg["content"] if sys_msg else "你是一个专业的金融分析助手。"

    if requested_mode == "agentic" and agentic_rag_pipeline is None and AgenticRAGPipeline is not None:
        try:
            agentic_rag_pipeline = AgenticRAGPipeline(
                retriever=rag_retriever,
                web_agent=web_knowledge_agent,
                config_path=agentic_rag_config_path,
            )
        except Exception as e:
            logger.warning(f"懒加载 Agentic RAG 失败，自动回退 basic RAG: {e}")

    if requested_mode == "agentic" and agentic_rag_pipeline is not None:
        augmented_sys_content, prepared_state = agentic_rag_pipeline.build_augmented_prompt_with_state(
            sys_content,
            last_user_msg["content"],
            task_type=task_type,
            runtime_options={
                "top_k": rag_top_k,
                "allow_web": should_use_web_knowledge,
                "web_search_top_n": web_search_top_n,
                "web_fetch_top_n": web_fetch_top_n,
                "enable_answer_guard": enable_answer_guard,
            },
        )
        if sys_msg:
            sys_msg["content"] = augmented_sys_content
        else:
            messages.insert(0, {"role": "system", "content": augmented_sys_content})
        return (messages, prepared_state) if return_agent_state else messages

    if should_use_web_knowledge and web_knowledge_agent is not None:
        try:
            saved_files = web_knowledge_agent.collect(
                last_user_msg["content"],
                search_top_n=web_search_top_n,
                fetch_top_n=web_fetch_top_n,
            )
            if saved_files:
                logger.info(f"实时联网补充知识完成，本次新增文档数: {len(saved_files)}")
                if rag_retriever is not None:
                    rag_retriever.refresh()
        except Exception as e:
            logger.warning(f"实时联网补充知识失败，自动降级为本地知识库检索: {e}")

    should_apply_rag = should_use_web_knowledge or rag_top_k > 0
    if should_apply_rag and rag_retriever and rag_retriever.is_ready():
        augmented_sys_content = rag_retriever.format_prompt_with_context(
            sys_content, last_user_msg["content"], top_k=rag_top_k
        )

        if sys_msg:
            sys_msg["content"] = augmented_sys_content
        else:
            messages.insert(0, {"role": "system", "content": augmented_sys_content})

    return (messages, None) if return_agent_state else messages


def _finalize_agentic_trajectory(prepared_state: Optional[dict[str, Any]], answer: str) -> Optional[dict[str, Any]]:
    if prepared_state is None or agentic_rag_pipeline is None:
        return None
    try:
        return agentic_rag_pipeline.finalize(prepared_state, answer)
    except Exception as exc:
        logger.warning(f"Agentic RAG completion audit failed: {exc}")
        return None


def _audit_summary(state: Optional[dict[str, Any]], enabled: bool) -> Optional[dict[str, Any]]:
    if not enabled or not state:
        return None
    audit = state.get("audit") or {}
    reward = state.get("reward_breakdown") or {}
    return {
        "query_id": audit.get("query_id", state.get("query_id", "")),
        "answer_guard_applied": bool(state.get("answer_guard_applied")),
        "validation_errors": audit.get("validation_errors", []),
        "hard_gate_passed": bool(reward.get("hard_gate_passed")),
        "hard_failures": reward.get("hard_failures", []),
        "total_reward": reward.get("total_reward", 0.0),
        "reward_version": reward.get("reward_version", ""),
        "scores": {
            key: reward.get(key, 0.0)
            for key in (
                "retrieval_relevance",
                "claim_support",
                "numeric_consistency",
                "citation_coverage",
                "citation_precision",
                "task_validity",
                "trajectory_quality",
            )
        },
    }


# ============================================================
# API 路由
# ============================================================
@app.get("/health")
async def health():
    """健康检查。"""
    return {
        "status": "ok",
        "model": model_name,
        "backend": backend_name,
        "model_loaded": model is not None,
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型（兼容 OpenAI API）。"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fin-instruct",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    对话补全接口（兼容 OpenAI API）。

    支持普通返回和 SSE 流式返回。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 构建消息列表
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # 获取 RAG 功能相关参数 (可通过额外字段传递)
    # 取巧的方式：如果开启了 RAG，我们拦截最新的一条 user message 并进行知识增强
    messages, prepared_agent_state = enrich_knowledge_before_generation(
        messages=messages,
        rag_mode=request.rag_mode,
        rag_top_k=request.rag_top_k,
        enable_web_knowledge=request.enable_web_knowledge,
        web_search_top_n=request.web_search_top_n,
        web_fetch_top_n=request.web_fetch_top_n,
        task_type=request.task_type,
        enable_answer_guard=request.enable_answer_guard,
        return_agent_state=True,
    )

    # Agentic mode records a structured, auditable reasoning trajectory. Do not
    # expose or persist the model's private free-form thinking trace as output.
    enable_thinking = False
    text = apply_chat_template(messages=messages, tokenizer=tokenizer, enable_thinking=enable_thinking)
    if backend_name == "vllm":
        if request.stream:
            return StreamingResponse(
                _stream_generate_vllm(text, request, raw_request, prepared_agent_state),
                media_type="text/event-stream",
            )
        return await _generate_vllm_response(text, request, prepared_agent_state)

    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    if request.stream:
        # 流式输出
        return StreamingResponse(
            _stream_generate(inputs, input_length, request, prepared_agent_state),
            media_type="text/event-stream",
        )
    else:
        # 非流式输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][input_length:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True)
        output_length = len(generated)
        final_state = _finalize_agentic_trajectory(prepared_agent_state, response_text)
        if final_state:
            response_text = str(final_state.get("answer", response_text))

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": output_length,
                "total_tokens": input_length + output_length,
            },
            audit_summary=_audit_summary(final_state, request.return_audit),
        )


async def _stream_generate(inputs, input_length, request, prepared_agent_state=None):
    """
    SSE 流式生成器。

    参数:
        inputs:       编码后的模型输入
        input_length: 输入序列长度
        request:      原始请求
    """
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature if request.temperature > 0 else None,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
        "repetition_penalty": request.repetition_penalty,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    complete_text = ""

    for new_text in streamer:
        if new_text:
            complete_text += new_text
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    thread.join()
    _finalize_agentic_trajectory(prepared_agent_state, complete_text)

    # 结束标记
    end_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"



async def _generate_vllm_response(prompt: str, request: ChatCompletionRequest, prepared_agent_state=None):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    output = await model.generate(
        prompt,
        request_id=chat_id,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    prompt_tokens = len(output.get("prompt_token_ids", []))
    completion_tokens = len(output.get("output_token_ids", []))
    final_state = _finalize_agentic_trajectory(prepared_agent_state, output.get("text", ""))
    response_text = str(final_state.get("answer", output.get("text", ""))) if final_state else output.get("text", "")
    return ChatCompletionResponse(
        id=chat_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": output.get("finish_reason", "stop"),
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        audit_summary=_audit_summary(final_state, request.return_audit),
    )


async def _stream_generate_vllm(
    prompt: str,
    request: ChatCompletionRequest,
    raw_request: Request,
    prepared_agent_state=None,
):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    previous_text = ""
    try:
        async for output in model.stream(
            prompt,
            request_id=chat_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        ):
            if await raw_request.is_disconnected():
                await model.abort(chat_id)
                return
            current_text = output.get("text", "")
            new_text = current_text[len(previous_text):]
            previous_text = current_text
            if not new_text:
                continue
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        await model.abort(chat_id)
        raise

    _finalize_agentic_trajectory(prepared_agent_state, previous_text)

    end_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fin-Instruct API 推理服务")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="推理后端 (默认: vllm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saves/qwen3-8b/merged",
        help="模型路径",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口 (默认: 8000)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Transformers 后端不使用 4-bit 量化",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype (默认: bfloat16)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (默认: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU 显存利用率 (默认: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM 最大上下文长度",
    )
    parser.add_argument(
        "--vllm-quantization",
        type=str,
        default=None,
        help="vLLM 量化方式，例如 bitsandbytes",
    )
    parser.add_argument(
        "--vllm-load-format",
        type=str,
        default=None,
        help="vLLM 权重加载格式，例如 bitsandbytes",
    )
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="启用 RAG 本地知识检索功能",
    )
    parser.add_argument(
        "--rag-mode",
        type=str,
        choices=["basic", "agentic"],
        default="basic",
        help="RAG 模式，basic 保持原有拼接检索，agentic 启用 LangGraph/顺序图编排",
    )
    parser.add_argument(
        "--rag-agentic-config",
        type=str,
        default=None,
        help="Agentic RAG 配置文件路径",
    )
    parser.add_argument(
        "--rag-db-dir",
        type=str,
        default=None,
        help="RAG 向量数据库路径",
    )
    parser.add_argument(
        "--enable-web-knowledge",
        action="store_true",
        help="在用户提问时自动联网检索并增量补充知识库",
    )
    parser.add_argument(
        "--web-knowledge-dir",
        type=str,
        default=None,
        help="实时联网知识 Markdown 落盘目录",
    )
    args = parser.parse_args()

    # 加载模型
    if args.backend == "vllm":
        load_vllm_model(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            quantization=args.vllm_quantization,
            load_format=args.vllm_load_format,
        )
    else:
        load_model(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            quantize=not args.no_quantize,
        )
    
    # 尝试初始化 RAG
    init_rag_system(args.enable_rag or args.enable_web_knowledge or args.rag_mode == "agentic", args.rag_db_dir)
    init_web_knowledge_system(
        args.enable_web_knowledge,
        db_dir=args.rag_db_dir,
        knowledge_dir=args.web_knowledge_dir,
    )
    init_agentic_rag_system(args.rag_mode, args.rag_agentic_config)

    # 启动服务
    logger.info(f"API 服务启动: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
