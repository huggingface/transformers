import os
from typing import List, Union

import tensorflow as tf

from tensorflow_text import FastBertTokenizer, ShrinkLongestTrimmer, case_fold_utf8, combine_segments, pad_model_inputs

from ...utils import requires_backends
from .tokenization_bert import BertTokenizer


class TFBertTokenizer(tf.keras.layers.Layer):
    # TODO Do we need to change the name? Most TF users will still want the normal tokenizers
    # TODO Add tests, particularly one with a full model and one with saving to savedmodel, as these are main use cases

    def __init__(
        self,
        vocab_list: List,
        do_lower_case: bool,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
        padding="longest",
        truncation=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_token_type_ids=True,
        return_attention_mask=True,
    ):
        super().__init__()

        requires_backends(self, ["tensorflow_text"])
        self.tf_tokenizer = FastBertTokenizer(vocab_list, token_out_type=tf.int64)
        self.vocab_list = vocab_list
        self.do_lower_case = do_lower_case
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.paired_trimmer = ShrinkLongestTrimmer(max_length - 3)
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_token_type_ids = return_token_type_ids
        self.return_attention_mask = return_attention_mask

    @classmethod
    def from_tokenizer(cls, tokenizer: "PreTrainedTokenizerBase", **kwargs):  # noqa: F821
        vocab = tokenizer.get_vocab()
        vocab = sorted([(wordpiece, idx) for wordpiece, idx in vocab.items()], key=lambda x: x[1])
        vocab_list = [entry[0] for entry in vocab]
        return cls(
            vocab_list=vocab_list,
            do_lower_case=tokenizer.do_lower_case,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        tokenizer = BertTokenizer.from_pretrained(*init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer, **kwargs)

    def unpaired_tokenize(self, texts):
        if self.do_lower_case:
            texts = case_fold_utf8(texts)
        else:
            texts = tf.constant(texts)
        return self.tf_tokenizer.tokenize(texts)

    def trim_paired_texts(self, text_a, text_b):
        texts = tf.ragged.stack([text_a, text_b], axis=1)
        texts = self.paired_trimer.trim(texts)
        text_a, text_b = texts[:, 0, :], texts[:, 1, :]
        return text_a, text_b

    def call(
        self,
        text,
        text_pair=None,
        padding=None,
        truncation=None,
        max_length=None,
        pad_to_multiple_of=None,
        return_token_type_ids=None,
        return_attention_mask=None,
    ):
        if padding is None:
            padding = self.padding
        if padding not in ("longest", "max_length"):
            raise ValueError("Padding must be either 'longest' or 'max_length'!")
        if max_length is not None and text_pair is not None:
            # Because we have to instantiate a Trimmer to do it properly
            raise ValueError("max_length cannot be overridden at call time when truncating paired texts!")
        if max_length is None:
            max_length = self.max_length
        if truncation is None:
            truncation = self.truncation
        if pad_to_multiple_of is None:
            pad_to_multiple_of = self.pad_to_multiple_of
        if return_token_type_ids is None:
            return_token_type_ids = self.return_token_type_ids
        if return_attention_mask is None:
            return_attention_mask = self.return_attention_mask
        text = self.unpaired_tokenize(text)
        if text_pair is None:
            if truncation and text.bounding_shape(axis=-1) > max_length - 2:
                text = text[:, : max_length - 2]  # Allow room for special tokens
            input_ids, token_type_ids = combine_segments(
                (text,), start_of_sequence_id=self.cls_token_id, end_of_segment_id=self.sep_token_id
            )
        else:
            text_pair = self.unpaired_tokenize(text_pair)
            if truncation:
                text, text_pair = self.trim_paired_texts(text, text_pair)
            input_ids, token_type_ids = combine_segments(
                (text, text_pair), start_of_sequence_id=self.cls_token_id, end_of_segment_id=self.sep_token_id
            )
        if padding == "longest":
            pad_length = input_ids.bounding_shape(axis=-1)
            if pad_to_multiple_of is not None:
                # No ceiling division in tensorflow, so we negate floordiv instead
                pad_length = pad_to_multiple_of * (-tf.math.floordiv(-pad_length, pad_to_multiple_of))
        else:
            pad_length = max_length

        input_ids, attention_mask = pad_model_inputs(input_ids, max_seq_length=pad_length, pad_value=self.pad_token_id)
        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = attention_mask
        if return_token_type_ids:
            token_type_ids, _ = pad_model_inputs(
                token_type_ids, max_seq_length=pad_length, pad_value=self.pad_token_id
            )
            output["token_type_ids"] = token_type_ids
        return output

    def get_config(self):
        return {
            "vocab_list": self.vocab_list,
            "do_lower_case": self.do_lower_case,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "pad_token_id": self.pad_token_id,
        }
