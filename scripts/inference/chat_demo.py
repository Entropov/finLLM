#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 交互式聊天演示

功能:
  - 基于 Gradio 构建 Web 交互界面
  - 支持选择 6 类金融任务（自动切换 system prompt）
  - 支持多轮对话
  - 支持流式输出
  - 可加载合并模型或基座模型+LoRA adapter

用法:
  python scripts/inference/chat_demo.py
  python scripts/inference/chat_demo.py --model-path saves/qwen2.5-7b/merged
  python scripts/inference/chat_demo.py --share  # 生成公网链接
"""

import json
import logging
from pathlib import Path
from typing import Optional, Generator
from threading import Thread

import torch
import gradio as gr

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
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"

# ============================================================
# 全局变量（模型和分词器）
# ============================================================
model = None
tokenizer = None
system_prompts = {}


def load_prompts() -> dict:
    """加载 system prompt 模板。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return {
        v["task_name"]: v["system_prompt"]
        for k, v in prompts.items()
        if "task_name" in v and "system_prompt" in v
    }


def load_model(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """
    加载模型和分词器。

    参数:
        model_path:   模型路径或 HuggingFace ID
        adapter_path: LoRA adapter 路径
        quantize:     是否使用 4-bit 量化
    """
    global model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

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

    # 加载 LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        logger.info(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    logger.info("模型加载完成！")


def generate_response(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """
    流式生成回复。

    参数:
        message:        用户当前输入
        history:        对话历史 [(user, assistant), ...]
        system_prompt:  系统提示词
        temperature:    温度系数
        max_new_tokens: 最大生成 token 数
        top_p:          核采样概率

    生成器:
        逐步产出的文本片段
    """
    from transformers import TextIteratorStreamer

    # 构建对话消息列表
    messages = [{"role": "system", "content": system_prompt}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    # 编码输入
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 设置流式输出器
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "repetition_penalty": 1.1,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    # 在后台线程中生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式产出文本
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

    thread.join()


def create_demo() -> gr.Blocks:
    """
    创建 Gradio 交互界面。

    返回:
        Gradio Blocks 应用
    """
    # 加载任务列表
    task_names = list(system_prompts.keys())
    default_task = "通用金融助手" if "通用金融助手" in task_names else task_names[0]

    with gr.Blocks(
        title="Fin-Instruct 金融AI助手",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🏦 Fin-Instruct 金融AI助手
            基于 Qwen2.5-7B-Instruct 微调的专业金融分析模型，
            支持股票分析、量化策略、财报解读、情感分析、金融问答和风险评估。

            **⚠️ 免责声明：所有分析仅供参考，不构成投资建议。**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_selector = gr.Dropdown(
                    choices=task_names,
                    value=default_task,
                    label="选择任务类型",
                    info="切换任务会自动更新系统提示词",
                )
                system_prompt_box = gr.Textbox(
                    value=system_prompts.get(default_task, ""),
                    label="系统提示词（可编辑）",
                    lines=6,
                    max_lines=10,
                )

                with gr.Accordion("高级参数", open=False):
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature（温度）",
                        info="越高越有创意，越低越确定",
                    )
                    max_tokens = gr.Slider(
                        minimum=128, maximum=4096, value=2048, step=128,
                        label="最大生成长度",
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-P（核采样）",
                    )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                    show_copy_button=True,
                )
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的金融分析问题...",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

        # 示例问题
        gr.Examples(
            examples=[
                "请分析贵州茅台(600519)的投资价值",
                "帮我设计一个基于动量因子的量化交易策略",
                "请解读比亚迪最新一季度的财务报表",
                "分析'央行降准0.5个百分点'这条新闻对A股市场的影响",
                "什么是市盈率？如何用来判断股票估值？",
                "请评估宁德时代的信用风险等级",
            ],
            inputs=msg,
            label="示例问题",
        )

        # 事件绑定
        def update_system_prompt(task_name):
            return system_prompts.get(task_name, "")

        task_selector.change(
            fn=update_system_prompt,
            inputs=[task_selector],
            outputs=[system_prompt_box],
        )

        def user_submit(message, history, sys_prompt, temp, max_tok, tp):
            if not message.strip():
                return "", history
            return "", history + [[message, None]]

        def bot_response(history, sys_prompt, temp, max_tok, tp):
            if not history or history[-1][1] is not None:
                return history
            user_message = history[-1][0]
            history_pairs = history[:-1]

            for partial in generate_response(
                message=user_message,
                history=history_pairs,
                system_prompt=sys_prompt,
                temperature=temp,
                max_new_tokens=max_tok,
                top_p=tp,
            ):
                history[-1][1] = partial
                yield history

        submit_btn.click(
            fn=user_submit,
            inputs=[msg, chatbot, system_prompt_box, temperature, max_tokens, top_p],
            outputs=[msg, chatbot],
        ).then(
            fn=bot_response,
            inputs=[chatbot, system_prompt_box, temperature, max_tokens, top_p],
            outputs=[chatbot],
        )

        msg.submit(
            fn=user_submit,
            inputs=[msg, chatbot, system_prompt_box, temperature, max_tokens, top_p],
            outputs=[msg, chatbot],
        ).then(
            fn=bot_response,
            inputs=[chatbot, system_prompt_box, temperature, max_tokens, top_p],
            outputs=[chatbot],
        )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fin-Instruct Gradio 交互演示")
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
        "--port",
        type=int,
        default=7860,
        help="服务端口 (默认: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="生成 Gradio 公网分享链接",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="不使用 4-bit 量化（需要更多显存）",
    )
    args = parser.parse_args()

    # 加载 system prompts
    system_prompts = load_prompts()

    # 加载模型
    load_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        quantize=not args.no_quantize,
    )

    # 启动 Demo
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
