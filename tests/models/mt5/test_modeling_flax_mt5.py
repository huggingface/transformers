# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import is_flax_available
from transformers.testing_utils import require_flax, require_sentencepiece, require_tokenizers, require_torch, slow


if is_flax_available():
    import optax
    from flax.training.common_utils import onehot

    from transformers import AutoTokenizer, FlaxMT5ForConditionalGeneration
    from transformers.models.t5.modeling_flax_t5 import shift_tokens_right


@require_torch
@require_sentencepiece
@require_tokenizers
@require_flax
class MT5IntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_mt5_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_mt5_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_mt5_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_mt5_spm_model_path)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = FlaxMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        input_ids = tokenizer("Hello there", return_tensors="np").input_ids
        labels = tokenizer("Hi I am", return_tensors="np").input_ids

        decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)

        logits = model(input_ids, decoder_input_ids=decoder_input_ids).logits
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()

        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -84.9127
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
