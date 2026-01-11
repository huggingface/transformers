from transformers import BertTokenizerFast

from .custom_tokenization import CustomTokenizer


class CustomTokenizerFast(BertTokenizerFast):
    slow_tokenizer_class = CustomTokenizer
    _auto_map = {
        "AutoTokenizer": ("custom_tokenization.CustomTokenizer", "custom_tokenization_fast.CustomTokenizerFast")
    }
