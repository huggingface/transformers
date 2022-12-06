import os
from typing import Dict, List, Union

import tensorflow as tf

from keras_nlp.tokenizers import BytePairTokenizer
from keras_nlp.tokenizers.byte_pair_tokenizer import split_strings_for_bpe
from tensorflow_text import pad_model_inputs
from transformers import CLIPTokenizer


class CLIPKerasNLPTokenizer(BytePairTokenizer):
    def __init__(self, vocab, merges, bos_token_id=None, eos_token_id=None, sequence_length=None):
        self.append_token = tf.convert_to_tensor("</w>")
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(vocab, merges, sequence_length)

    def tokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        raw_tokens = split_strings_for_bpe(inputs)
        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""

        has_unseen_words = tf.math.reduce_any((cache_lookup == "") & (flat_tokens != ""))

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )

        tokens = tf.strings.split(tokenized_words, sep=" ") + self.append_token
        if self.compute_dtype != tf.string:
            # Encode merged tokens.
            tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input in scalar
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens


class TFCLIPTokenizer(tf.keras.layers.Layer):
    """
    This is an in-graph tokenizer for CLIP. It should be initialized similarly to other tokenizers, using the
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

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[str],
        max_length: int = None,
        pad_token_id: int = None,
        eos_token_id: int = None,
        bos_token_id: int = None,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.vocab = vocab
        self.merges = merges
        self.tf_tokenizer = CLIPKerasNLPTokenizer(vocab, merges, sequence_length=max_length)

    @classmethod
    def from_tokenizer(cls, tokenizer: CLIPTokenizer, *args, **kwargs):
        """Creates TFCLIPTokenizer from CLIPTokenizer

        Args:
            tokenizer (CLIPTokenizer)

        Examples:

        ```python
        from transformers import AutoTokenizer, TFCLIPTokenizer

        tokenizer = AutoTokenizer.from_pretrained("CLIP")
        tf_tokenizer = TFCLIPTokenizer.from_tokenizer(tokenizer)
        ```
        """
        merges = [" ".join(m) for m in tokenizer.bpe_ranks.keys()]
        vocab = tokenizer.get_vocab()

        eos_token_id = None
        bos_token_id = None

        if tokenizer.eos_token is not None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer._eos_token.content)

        if tokenizer.bos_token is not None:
            bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer._bos_token.content)

        return cls(vocab, merges, eos_token_id=eos_token_id, bos_token_id=bos_token_id * args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Creates TFCLIPTokenizer from pretrained CLIPTokenizer

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Path to pretrained model

        Examples:

        ```python
        from transformers import TFCLIPTokenizer

        tf_tokenizer = TFCLIPTokenizer.from_pretrained("CLIP")
        ```
        """
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer, *init_inputs, **kwargs)

    @classmethod
    def from_config(cls, config):
        """Creates TFCLIPTokenizer from configurations

        Args:
            config (Dict): Dictionary with keys such as stated in `get_config`.
        """
        return cls(**config)

    def get_config(self):
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    def call(self, x, max_length: int = None):
        input_ids = self.tf_tokenizer(x)
        attention_mask = tf.ones_like(input_ids)

        if self.pad_token_id is not None:
            # pad the tokens up to max length
            max_length = max_length if max_length is not None else self.max_length

            if max_length is not None:
                input_ids, attention_mask = pad_model_inputs(
                    input_ids, max_seq_length=max_length, pad_value=self.pad_token_id
                )

        return {"attention_mask": attention_mask, "input_ids": input_ids}
