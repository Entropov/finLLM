"""vLLM backend helpers for API and batch inference."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional


VLLM_INSTALL_HINT = (
    "vLLM is required for --backend vllm. Install it with: "
    "pip install -r requirements.txt"
)


def _import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(VLLM_INSTALL_HINT) from exc
    return LLM, SamplingParams


def _import_async_vllm():
    try:
        from vllm import SamplingParams
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
    except ImportError as exc:
        raise RuntimeError(VLLM_INSTALL_HINT) from exc
    return AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def build_sampling_params(
    *,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    sampling_params_cls: Optional[type] = None,
):
    """Create vLLM SamplingParams with deterministic decoding when temperature is 0."""
    if sampling_params_cls is None:
        _, sampling_params_cls = _import_vllm()

    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if temperature <= 0:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
    return sampling_params_cls(**kwargs)


@dataclass
class VLLMEngineConfig:
    model_path: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    load_format: Optional[str] = None

    def as_engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.model_path,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.quantization:
            kwargs["quantization"] = self.quantization
        if self.load_format:
            kwargs["load_format"] = self.load_format
        return kwargs


class VLLMBatchEngine:
    """Synchronous vLLM engine for offline batch inference."""

    def __init__(self, config: VLLMEngineConfig):
        llm_cls, sampling_params_cls = _import_vllm()
        self._sampling_params_cls = sampling_params_cls
        self.llm = llm_cls(**config.as_engine_kwargs())

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float = 1.1,
    ) -> list[dict[str, Any]]:
        sampling_params = build_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sampling_params_cls=self._sampling_params_cls,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [normalize_request_output(output) for output in outputs]


class VLLMAPIEngine:
    """Async vLLM engine for FastAPI chat completions."""

    def __init__(self, config: VLLMEngineConfig):
        async_engine_args_cls, async_llm_engine_cls, sampling_params_cls = _import_async_vllm()
        self._sampling_params_cls = sampling_params_cls
        engine_args = async_engine_args_cls(**config.as_engine_kwargs())
        self.engine = async_llm_engine_cls.from_engine_args(engine_args)

    async def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> dict[str, Any]:
        final_output = None
        async for output in self.stream(
            prompt,
            request_id=request_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            final_output = output
        return final_output or {"text": "", "output_token_ids": [], "finish_reason": "stop"}

    async def stream(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        sampling_params = build_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sampling_params_cls=self._sampling_params_cls,
        )
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            yield normalize_request_output(output)

    async def abort(self, request_id: str) -> None:
        result = self.engine.abort(request_id)
        if inspect.isawaitable(result):
            await result


def normalize_request_output(output: Any) -> dict[str, Any]:
    """Convert vLLM RequestOutput into a stable dict for the app layer."""
    candidate = output.outputs[0] if getattr(output, "outputs", None) else None
    text = getattr(candidate, "text", "") if candidate is not None else ""
    output_token_ids = list(getattr(candidate, "token_ids", []) or [])
    prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
    finish_reason = getattr(candidate, "finish_reason", None) if candidate is not None else None
    return {
        "text": text,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "finish_reason": finish_reason or "stop",
    }


def validate_vllm_model_options(adapter_path: Optional[str]) -> None:
    if adapter_path:
        raise ValueError(
            "vLLM backend expects a merged model. Merge LoRA weights first, "
            "or use --backend transformers with --adapter-path."
        )
