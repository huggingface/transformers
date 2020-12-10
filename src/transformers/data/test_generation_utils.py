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

import random
import unittest

import timeout_decorator

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import MarianConfig, MarianMTModel


@require_torch
class GenerationUtilsTest(unittest.TestCase):
    @cached_property
    def config(self):
        config = MarianConfig.from_pretrained("sshleifer/tiny-marian-en-de")
        return config

    @cached_property
    def model(self):
        return MarianMTModel(self.config)

    def test_postprocess_next_token_scores(self):
        config = self.config
        model = self.model
        # Initialize an input id tensor with batch size 8 and sequence length 12
        input_ids = torch.arange(0, 96, 1).view((8, 12))
        eos = config.eos_token_id
        bad_words_ids_test_cases = [[[299]], [[23, 24], [54]], [[config.eos_token_id]], []]
        masked_scores = [
            [(0, 299), (1, 299), (2, 299), (3, 299), (4, 299), (5, 299), (6, 299), (7, 299)],
            [(1, 24), (0, 54), (1, 54), (2, 54), (3, 54), (4, 54), (5, 54), (6, 54), (7, 54)],
            [(0, eos), (1, eos), (2, eos), (3, eos), (4, eos), (5, eos), (6, eos), (7, eos)],
            [],
        ]

        for test_case_index, bad_words_ids in enumerate(bad_words_ids_test_cases):
            # Initialize a scores tensor with batch size 8 and vocabulary size 300
            scores = torch.rand((8, 300))
            output = model.postprocess_next_token_scores(
                scores,
                input_ids,
                0,
                bad_words_ids,
                13,
                15,
                config.max_length,
                config.eos_token_id,
                config.repetition_penalty,
                32,
                5,
            )
            for masked_score in masked_scores[test_case_index]:
                self.assertTrue(output[masked_score[0], masked_score[1]] == -float("inf"))

    @timeout_decorator.timeout(10)
    def test_postprocess_next_token_scores_large_bad_words_list(self):

        config = self.config
        model = self.model
        # Initialize an input id tensor with batch size 8 and sequence length 12
        input_ids = torch.arange(0, 96, 1).view((8, 12))

        bad_words_ids = []
        for _ in range(100):
            length_bad_word = random.randint(1, 4)
            bad_words_ids.append(random.sample(range(1, 300), length_bad_word))

        scores = torch.rand((8, 300))
        _ = model.postprocess_next_token_scores(
            scores,
            input_ids,
            0,
            bad_words_ids,
            13,
            15,
            config.max_length,
            config.eos_token_id,
            config.repetition_penalty,
            32,
            5,
        )
