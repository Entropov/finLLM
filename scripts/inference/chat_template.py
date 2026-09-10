"""Chat template helpers shared by local inference and evaluation scripts."""


def apply_chat_template(tokenizer, messages, *, add_generation_prompt=True, enable_thinking=False):
    """Render chat messages, disabling Qwen3 thinking mode when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
