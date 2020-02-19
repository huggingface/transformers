# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

import ipdb

from transformers import is_torch_available

from .utils import CACHE_DIR, require_torch, slow, torch_device
from .utils_generate import GPT2_PADDING_TOKENS, GPT2_PRETRAINED_MODEL_GENERATION_TEST_CASES


if is_torch_available():
    import torch
    from transformers import (
        GPT2LMHeadModel,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    )


@require_torch
class LMGenerateTest(unittest.TestCase):

    all_model_classes_gpt2 = (GPT2LMHeadModel) if is_torch_available() else ()

    class LMGenerateTester(object):
        def __init__(
            self,
            parent,
            max_length=20,
            do_sample=True,
            num_return_sequences=1,
            num_beams=1,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_id=None,
        ):
            self.parent = parent
            self.max_length = max_length
            self.do_sample = do_sample
            self.num_return_sequences = num_return_sequences
            self.num_beams = num_beams
            self.bos_token_id = bos_token_id
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id

        def set_tokens(self, tokens_archieve):
            self.bos_token_id = tokens_archieve["bos_token_id"] if "bos_token_id" in tokens_archieve else None
            self.pad_token_id = tokens_archieve["pad_token_id"] if "pad_token_id" in tokens_archieve else None
            self.eos_token_id = tokens_archieve["eos_token_id"] if "eos_token_id" in tokens_archieve else None

        def list_to_torch_tensor(self, tokens_list):
            return torch.tensor(tokens_list, dtype=torch.long, device=torch_device)

        def check_language_generate(self, model_generate_test_cases, model_archive_map, lm_model):
            for model_name in list(model_archive_map.keys()):
                if model_name in model_generate_test_cases:
                    model = lm_model.from_pretrained(model_name, cache_dir=CACHE_DIR)
                    self.parent.assertIsNotNone(model)
                    model_test_config = model_generate_test_cases[model_name]
                    torch_seed = model_test_config["seed"]
                    input_ids = self.list_to_torch_tensor(model_test_config["input"])
                    expected_output_ids = model_test_config["exp_output"]

                    torch.manual_seed(torch_seed)
                    output_ids = model.generate(
                        input_ids,
                        bos_token_id=self.bos_token_id,
                        pad_token_id=self.pad_token_id,
                        eos_token_ids=self.eos_token_id,
                        max_length=self.max_length,
                        do_sample=self.do_sample,
                        num_return_sequences=self.num_return_sequences,
                        num_beams=self.num_beams,
                    )
                    ipdb.set_trace()
                    self.parent.assertListEqual(list(output_ids[0]), expected_output_ids)

    def setUp(self):
        self.model_tester = LMGenerateTest.LMGenerateTester(self)

    @slow
    def test_gpt2_generate(self):
        self.model_tester.set_tokens(GPT2_PADDING_TOKENS)
        self.model_tester.check_language_generate(
            GPT2_PRETRAINED_MODEL_GENERATION_TEST_CASES, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2LMHeadModel,
        )
