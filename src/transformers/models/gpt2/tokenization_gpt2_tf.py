import os
from typing import Dict, List, Union

import tensorflow as tf

from keras_nlp.tokenizers import BytePairTokenizer

from .tokenization_gpt2 import GPT2Tokenizer


class TFGPT2Tokenizer(tf.keras.layers.Layer):
    """
    This is an in-graph tokenizer for GPT2. It should be initialized similarly to other tokenizers, using the
    `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
    from an existing standard tokenizer object.

    In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
    when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
    than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
    straight from `tf.string` inputs to outputs.

    Args:
        vocab (Dict[str, int]): Vocabulary dict for Byte Pair Tokenizer
        merges (List[str]): Merges list for Byte Pair Tokenizer
    """

    def __init__(self, vocab: Dict[str, int], merges: List[str]):
        super().__init__()
        self.tf_tokenizer = BytePairTokenizer(vocab, merges)

    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer):
        """Creates TFGPT2Tokenizer from GPT2Tokenizer

        Args:
            tokenizer (GPT2Tokenizer)

        Examples:

        ```python
        from transformers import AutoTokenizer, TFGPT2Tokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tf_tokenizer = TFGPT2Tokenizer.from_tokenizer(tokenizer)
        ```
        """
        merges = [" ".join(m) for m in tokenizer.bpe_ranks.keys()]
        vocab = tokenizer.get_vocab()
        return cls(vocab, merges)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Creates TFGPT2Tokenizer from pretrained GPT2Tokenizer

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Path to pretrained model

        Examples:

        ```python
        from transformers import TFGPT2Tokenizer

        tf_tokenizer = TFGPT2Tokenizer.from_pretrained("gpt2")
        ```
        """
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer)

    def call(self, x):
        input_ids = self.tf_tokenizer(x)
        attention_mask = tf.ones_like(input_ids)
        return {"attention_mask": attention_mask, "input_ids": input_ids}
