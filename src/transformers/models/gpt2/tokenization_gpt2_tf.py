import os
from typing import Dict, List, Union

import tensorflow as tf

from keras_nlp.tokenizers import BytePairTokenizer

from .tokenization_gpt2 import GPT2Tokenizer


class TFGPT2Tokenizer(tf.keras.layers.Layer):
    def __init__(self, vocab: Dict[str, int], merges: List[str]) -> None:
        super().__init__()
        self.tf_tokenizer = BytePairTokenizer(vocab, merges)

    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer):
        merges = [" ".join(m) for m in tokenizer.bpe_ranks.keys()]
        vocab = tokenizer.get_vocab()
        return cls(vocab, merges)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer)

    def call(self, x):
        input_ids = self.tf_tokenizer(x)
        attention_mask = tf.ones_like(input_ids)
        return {"attention_mask": attention_mask, "input_ids": input_ids}
