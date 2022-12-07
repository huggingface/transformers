import os
from typing import Dict, List, Union

import tensorflow as tf

from keras_nlp.tokenizers import BytePairTokenizer
from keras_nlp.tokenizers.byte_pair_tokenizer import remove_strings_from_inputs, split_strings_for_bpe
from tensorflow_text import pad_model_inputs
from transformers import BasicTokenizer, CLIPTokenizer


class CLIPKerasNLPTokenizer(BytePairTokenizer):
    def __init__(self, vocab, merges, bos_token_id=None, eos_token_id=None, sequence_length=None):
        self.append_token = tf.convert_to_tensor("</w>")
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.nlp = BasicTokenizer(do_lower_case=True)

        super().__init__(vocab, merges, sequence_length)

    def tokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = [" ".join(self.nlp.tokenize(inputs))]
            inputs = tf.convert_to_tensor(inputs)

        inputs = tf.strings.lower(inputs, encoding="")

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)
        raw_tokens = split_strings_for_bpe(
            inputs
        )  # This is an english sentence -> ["This", " is", " an", " english", "sentence"]
        token_row_splits = raw_tokens.row_splits
        flat_tokens = tf.strings.regex_replace(raw_tokens.flat_values, "^ ", "")
        # flat_tokens = tf.strings.regex_replace(raw_tokens.flat_values, "^ ", "") # -> ["This", "is", "an", "english", "sentence"]

        # first = tf.reshape(flat_tokens, (1,-1)) # (10, ) -> (1,10) [["This", " is", " an", " english", "sentence"]]
        # second = tf.ragged.constant([["</w>"] * flat_tokens.shape[-1]]) # [["</w>", "</w>, ...]]
        # flat_tokens_ = tf.reshape(tf.strings.join(tf.concat([first, second], axis=0)), (-1)) # [["This</w>", "is</w>", " an</w>", " english</w>", "sentence"]]
        # tokenized = self.token_to_id_map.lookup(flat_tokens_) # [1, 2, 3, -1, 0]
        # self.cache.insert(tf.boolean_mask(flat_tokens, tokenized != -1), tf.boolean_mask(flat_tokens, tokenized != -1))
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
        tokens = tf.strings.split(tokenized_words, sep=" ")  # ["This</w>", "is</w>", "an", "english", "sent ence"]
        tokens = tf.concat([tokens[:, :-1], tokens[:, -1:] + "</w>"], axis=-1)
        # return tokens
        # return tokens
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

    @tf.function
    def _bpe_merge_one_step(self, words, mask):
        """Perform one step of byte-pair merge."""
        # Get all word pairs.
        first, second = words[:, :-1], words[:, 1:]  # sentence</w> -> ([s, e, n, t, e, n, c], [e, n, t, e, n, c, e])
        second = tf.concat([second[:, :-1], second[:, -1:] + "</w>"], axis=-1)  # [e, n, t, e, n, c, e</w>]

        # Mask empty.
        non_empty_mask = second.nested_row_lengths()[0] != 0
        mask = mask & non_empty_mask
        if not tf.reduce_any(mask):
            return [words, mask]
        non_empty_indices = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        filterd_first = tf.ragged.boolean_mask(first, mask)
        filtered_second = tf.ragged.boolean_mask(second, mask)

        # Get byte pair ranking in merge rules.
        pairs = tf.strings.join([filterd_first, filtered_second], separator=" ")
        pair_rank = self.merge_ranks.lookup(pairs)

        # Get BPE pair ranks.
        min_pair_rank = tf.reduce_min(pair_rank, axis=1)
        pair_found_mask = min_pair_rank != self.merge_ranks_lookup_default

        # Tokens that cannot be further merged are marked as finished.
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(non_empty_indices, axis=1), pair_found_mask)
        if not tf.math.reduce_any(mask):
            return [words, mask]

        masked_pair_rank = tf.ragged.boolean_mask(pair_rank, pair_found_mask)
        min_pair_rank_indices = tf.math.argmin(masked_pair_rank.to_tensor(self.merge_ranks_lookup_default), axis=1)

        # Get words and pairs to process.
        unfinished_words = tf.ragged.boolean_mask(words, mask)

        pair_left = tf.gather(unfinished_words, min_pair_rank_indices, batch_dims=1)
        pair_right = tf.gather(unfinished_words, min_pair_rank_indices + 1, batch_dims=1)

        merged_pairs = tf.strings.join([pair_left, pair_right])
        empty_strs = tf.fill(tf.shape(merged_pairs), "")

        unfinished_word_indices = tf.cast(tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask), dtype=tf.int64)
        merged_pair_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis],
            ],
            axis=1,
        )
        empty_string_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis] + 1,
            ],
            axis=1,
        )

        tensor_words = words.to_tensor(default_value="")
        tensor_words = tf.tensor_scatter_nd_update(
            tensor_words,
            merged_pair_indices,
            merged_pairs,
        )

        words = tf.tensor_scatter_nd_update(
            tensor_words,
            empty_string_indices,
            empty_strs,
        )
        # Remove empty strings.
        words = remove_strings_from_inputs(words, "")
        return [words, mask]


class TFCLIPTokenizer(tf.keras.layers.Layer):
    """
    Args:
    This is an in-graph tokenizer for CLIP. It should be initialized similarly to other tokenizers, using the
    `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
    from an existing standard tokenizer object. In-graph tokenizers, unlike other Hugging Face tokenizers, are actually:
    Keras layers and are designed to be run when the model is called, rather than during preprocessing. As a result,
    they have somewhat more limited options than standard tokenizer classes. They are most useful when you want to
    create an end-to-end model that goes straight from `tf.string` inputs to outputs.
        vocab (Dict[str, int]): Vocabulary dict for Byte Pair Tokenizer merges (List[str]): Merges list for Byte Pair
        Tokenizer
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

        if tokenizer._eos_token is not None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer._eos_token.content)

        if tokenizer._bos_token is not None:
            bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer._bos_token.content)

        return cls(vocab, merges, eos_token_id=eos_token_id, bos_token_id=bos_token_id, *args, **kwargs)

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

    def tokenize(self, x):
        input_ids = self.tf_tokenizer(x)

        oned_tokens = tf.ones(1)
        bos_tokens = tf.cast((oned_tokens * self.bos_token_id), tf.int32)
        eos_tokens = tf.cast((oned_tokens * self.eos_token_id), tf.int32)

        input_ids = tf.concat([bos_tokens, input_ids, eos_tokens], axis=0)
        return input_ids

    def call(self, x, max_length: int = None):
        input_ids = tf.map_fn(
            self.tokenize,
            x,
            dtype=tf.int32,
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=0,
                dtype=tf.int32,
            ),
        )

        attention_mask = tf.ones_like(input_ids)

        if self.pad_token_id is not None:
            # pad the tokens up to max length
            max_length = max_length if max_length is not None else self.max_length

            if max_length is not None:
                input_ids, attention_mask = pad_model_inputs(
                    input_ids, max_seq_length=max_length, pad_value=self.pad_token_id
                )

        return {"attention_mask": attention_mask, "input_ids": input_ids}
