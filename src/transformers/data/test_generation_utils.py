import random
import unittest

import timeout_decorator

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import (
        MarianConfig,
        MarianMTModel,
    )


@require_torch
class GenerationUtilsTest(unittest.TestCase):
    @cached_property
    def config_and_model(self):
        config = MarianConfig.from_pretrained("sshleifer/tiny-marian-en-de")
        return config, MarianMTModel(config)

    @require_torch
    def test_postprocess_next_token_scores(self):
        config, model = self.config_and_model
        # Initialize an input id tensor with batch size 8 and sequence length 12
        input_ids = torch.arange(0, 96, 1).view((8, 12))

        bad_words_ids_test_cases = [[[299]], [[23, 24], [54]], [[config.eos_token_id]], []]
        masked_scores = [
            [(0, 299), (1, 299), (2, 299), (3, 299), (4, 299), (5, 299), (6, 299), (7, 299)],
            [(1, 24), (0, 54), (1, 54), (2, 54), (3, 54), (4, 54), (5, 54), (6, 54), (7, 54)],
            [
                (0, config.eos_token_id),
                (1, config.eos_token_id),
                (2, config.eos_token_id),
                (3, config.eos_token_id),
                (4, config.eos_token_id),
                (5, config.eos_token_id),
                (6, config.eos_token_id),
                (7, config.eos_token_id),
            ],
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

    @require_torch
    @timeout_decorator.timeout(10)
    def test_postprocess_next_token_scores_large_bad_words_list(self):

        config, model = self.config_and_model
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
