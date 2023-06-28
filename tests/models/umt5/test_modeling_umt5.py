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

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device


if is_torch_available():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# https://github.com/google-research/t5x/blob/main/docs/models.md#umt5-checkpoints

@require_torch
@require_sentencepiece
@require_tokenizers
class Umt5IntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparison run the kaggle notbook available here : TODO share view link only
        """

        model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small", return_dict=True).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
        input_text = [
            'Bonjour monsieur <extra_id_0> bien <extra_id_1>.',
            'No se como puedo <extra_id_0>.',
            'This is the reason why we <extra_id_0> them.',
            'Sie <extra_id_0> schönen Film erstellen.'
        ]
        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -84.9127
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
