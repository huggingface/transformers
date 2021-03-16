# coding=utf-8
# Copyright 2020 Huggingface
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

import io
import json
import unittest

from parameterized import parameterized
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers.testing_utils import get_tests_dir, require_torch, slow, torch_device
from utils import calculate_bleu


filename = get_tests_dir() + "/test_data/fsmt/fsmt_val_data.json"
with io.open(filename, "r", encoding="utf-8") as f:
    bleu_data = json.load(f)


@require_torch
class ModelEvalTester(unittest.TestCase):
    def get_tokenizer(self, mname):
        return FSMTTokenizer.from_pretrained(mname)

    def get_model(self, mname):
        model = FSMTForConditionalGeneration.from_pretrained(mname).to(torch_device)
        if torch_device == "cuda":
            model.half()
        return model

    @parameterized.expand(
        [
            ["en-ru", 26.0],
            ["ru-en", 22.0],
            ["en-de", 22.0],
            ["de-en", 29.0],
        ]
    )
    @slow
    def test_bleu_scores(self, pair, min_bleu_score):
        # note: this test is not testing the best performance since it only evals a small batch
        # but it should be enough to detect a regression in the output quality
        mname = f"facebook/wmt19-{pair}"
        tokenizer = self.get_tokenizer(mname)
        model = self.get_model(mname)

        src_sentences = bleu_data[pair]["src"]
        tgt_sentences = bleu_data[pair]["tgt"]

        batch = tokenizer(src_sentences, return_tensors="pt", truncation=True, padding="longest").to(torch_device)
        outputs = model.generate(
            input_ids=batch.input_ids,
            num_beams=8,
        )
        decoded_sentences = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        scores = calculate_bleu(decoded_sentences, tgt_sentences)
        print(scores)
        self.assertGreaterEqual(scores["bleu"], min_bleu_score)
