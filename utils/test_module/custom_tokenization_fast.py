from transformers import BertTokenizer

from .custom_tokenization import CustomTokenizer


class CustomTokenizerFast(BertTokenizer):
    slow_tokenizer_class = CustomTokenizer
