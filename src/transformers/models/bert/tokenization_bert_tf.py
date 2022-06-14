from typing import List

import tensorflow as tf

from tensorflow_text import FastBertTokenizer, case_fold_utf8, combine_segments, pad_model_inputs

from ... import PreTrainedTokenizer


class TFBertTokenizer(tf.keras.layers.Layer):
    # TODO Input normalization?
    # TODO Is texts_a / texts_b the right way to handle inputs? Should it just be a multidimensional tensor?
    # TODO Should this be a more complete class rather than reading data from an existing tokenizer?
    # TODO Add imports and maybe some kind of AutoModel to make this findable by users
    # TODO Do we need to change the name? Most TF users will still want the normal tokenizers
    # TODO Add tests, particularly one with a full model and one with saving to savedmodel, as these are main use cases
    # TODO Properly variable padding

    def __init__(self, vocab_list: List, do_lower_case: bool, cls_token_id: int, sep_token_id: int, pad_token_id: int):
        super().__init__()
        self.tf_tokenizer = FastBertTokenizer(vocab_list, token_out_type=tf.int64)
        self.vocab_list = vocab_list
        self.do_lower_case = do_lower_case
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer):
        vocab = tokenizer.get_vocab()
        vocab = sorted([(wordpiece, idx) for wordpiece, idx in vocab.items()], key=lambda x: x[1])
        vocab_list = [entry[0] for entry in vocab]
        return cls(
            vocab_list=vocab_list,
            do_lower_case=tokenizer.do_lower_case,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    def unpaired_tokenize(self, texts):
        if self.do_lower_case:
            texts = case_fold_utf8(texts)
        else:
            texts = tf.constant(texts)
        return self.tf_tokenizer.tokenize(texts)

    def call(self, texts_a, texts_b=None):
        texts_a = self.unpaired_tokenize(texts_a)
        if texts_b is None:
            input_ids, token_type_ids = combine_segments(
                (texts_a,), start_of_sequence_id=self.cls_token_id, end_of_segment_id=self.sep_token_id
            )
        else:
            texts_b = self.unpaired_tokenize(texts_b)
            input_ids, token_type_ids = combine_segments(
                (texts_a, texts_b), start_of_sequence_id=self.cls_token_id, end_of_segment_id=self.sep_token_id
            )
        input_ids, attention_mask = pad_model_inputs(input_ids, max_seq_length=512, pad_value=self.pad_token_id)
        token_type_ids, _ = pad_model_inputs(token_type_ids, max_seq_length=512, pad_value=self.pad_token_id)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    def get_config(self):
        return {
            "vocab_list": self.vocab_list,
            "do_lower_case": self.do_lower_case,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "pad_token_id": self.pad_token_id,
        }
