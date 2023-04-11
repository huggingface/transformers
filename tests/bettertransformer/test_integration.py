# coding=utf-8
# Copyright 2023 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest

from optimum.bettertransformer import BetterTransformer

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import (
    require_optimum,
    require_torch,
    slow,
)


@require_torch
@require_optimum
@slow
class BetterTransformerIntegrationTest(unittest.TestCase):
    # refer to the full test suite in Optimum library:
    # https://github.com/huggingface/optimum/tree/main/tests/bettertransformer

    def test_transform_and_reverse(self):
        model_id = "hf-internal-testing/tiny-random-t5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        inp = tokenizer("This is me", return_tensors="pt")

        model = BetterTransformer.transform(model)

        model.generate(**inp)

        model = BetterTransformer.reverse(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            model_reloaded = AutoModelForSeq2SeqLM.from_pretrained(tmpdirname)

            model_reloaded.generate(**inp)
