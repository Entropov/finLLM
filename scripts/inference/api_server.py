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
  python scripts/inference/api_server.py --model-path saves/qwen2.5-7b/merged --port 8000
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, List
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
model_name = "fin-instruct-qwen2.5-7b"


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


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict


# ============================================================
# 模型加载
# ============================================================
def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """加载模型和分词器。"""
    global model, tokenizer, model_name

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
    logger.info("模型加载完成！")


# ============================================================
# API 路由
# ============================================================
@app.get("/health")
async def health():
    """健康检查。"""
    return {
        "status": "ok",
        "model": model_name,
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
async def chat_completions(request: ChatCompletionRequest):
    """
    对话补全接口（兼容 OpenAI API）。

    支持普通返回和 SSE 流式返回。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 构建消息列表
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # 编码输入
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    if request.stream:
        # 流式输出
        return StreamingResponse(
            _stream_generate(inputs, input_length, request),
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
        )


async def _stream_generate(inputs, input_length, request):
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

    for new_text in streamer:
        if new_text:
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

    thread.join()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fin-Instruct API 推理服务")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
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
        help="不使用 4-bit 量化",
    )
    args = parser.parse_args()

    # 加载模型
    load_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        quantize=not args.no_quantize,
    )

    # 启动服务
    logger.info(f"API 服务启动: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
