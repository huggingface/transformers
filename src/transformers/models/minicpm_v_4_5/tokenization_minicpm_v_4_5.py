from transformers.models.qwen2 import Qwen2Tokenizer


class MiniCPM_V_4_5Tokenizer(Qwen2Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = ["MiniCPM_V_4_5Tokenizer"]
