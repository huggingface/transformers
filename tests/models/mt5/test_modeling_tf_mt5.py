# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available():
    import tensorflow as tf

    from transformers import AutoTokenizer, T5Tokenizer, TFAutoModelForSeq2SeqLM, TFMT5ForConditionalGeneration


@require_tf
class TFMT5ModelTest(unittest.TestCase):  # no mixin with common tests -> most cases are already covered in the TF T5
    @slow
    def test_resize_embeddings(self):
        model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        original_vocab_size = model.get_input_embeddings().weight.shape[0]
        # the vocab size is defined in the model config
        self.assertEqual(original_vocab_size, model.config.vocab_size)

        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        tokenizer.add_special_tokens({"bos_token": "", "eos_token": ""})
        model._resize_token_embeddings(len(tokenizer))
        # the vocab size is now resized to the length of the tokenizer, which is different from the original size
        self.assertEqual(model.get_input_embeddings().weight.shape[0], len(tokenizer))
        self.assertNotEqual(model.get_input_embeddings().weight.shape[0], original_vocab_size)


@require_tf
@require_sentencepiece
@require_tokenizers
class TFMT5ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_mt5_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_mt5_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_mt5_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_mt5_spm_model_path, extra_ids=100)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = TFAutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        input_ids = tokenizer("Hello there", return_tensors="tf").input_ids
        labels = tokenizer("Hi I am", return_tensors="tf").input_ids

        loss = model(input_ids, labels=labels).loss
        mtf_score = -tf.math.reduce_mean(loss).numpy()

        EXPECTED_SCORE = -21.228168
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 2e-4)
