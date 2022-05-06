# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch OFA model. """

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow


if is_torch_available():
    from transformers import OFAForConditionalGeneration, OFATokenizer


@require_torch
@require_sentencepiece
@require_tokenizers
class OFAModelIntegrationTests(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        import torch

        model = OFAForConditionalGeneration.from_pretrained("OFA-Sys/OFA-base")
        tokenizer = OFATokenizer.from_pretrained("OFA-Sys/OFA-base")
        txt = " what is the description of the image?"
        inputs = tokenizer([txt], max_length=1024, return_tensors="pt")["input_ids"]
        patch_img = torch.ones((1, 3, 256, 256), dtype=torch.float32)
        gen = model.generate(inputs, patch_images=patch_img, num_beams=4)
        result = tokenizer.batch_decode(gen, skip_special_tokens=True).strip()
        self.assertTrue(result == "the image is out of focus")
