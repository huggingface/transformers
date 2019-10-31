from collections import namedtuple
import unittest

import numpy as np
import torch

from transformers.generate import BeamSearch
from transformers import PreTrainedEncoderDecoder


StubTokenizer = namedtuple("Tokenizer", ["bos_token_id", "eos_token_id", "pad_token_id"])
StubTransformer = namedtuple("Transformer", ["encoder", "decoder"])


class BeamSearchtest(unittest.TestCase):
    def test_beam_search_encoder_decoder_integration(self):
        """ We make sure that no internal change in the PreTrainedEncoderDecoder
        class will break the integration with the beam search.
        """

        model = PreTrainedEncoderDecoder("encoder", "decoder")
        tokenizer = StubTokenizer(0, 1, 2)
        try:
            _ = BeamSearch(
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                beam_size=1,
                min_length=1,
                max_length=1,
                alpha=0,
                block_repeating_trigrams=False,
            )
        except:
            self.fail("Instantiating BeamSearch with a PreTrainedEncoderDecoder failed.")

    def test_beam_search_min_length(self):
        """ We keep predicting the end_token for the first beam and check that
        it is not marked as finished until the beam has reached the minimum
        length. """
        eos_idx = 3
        vocab_size = 10

        batch_size = 3
        beam_size = 2
        min_length = 5

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(bos_token_id=0, eos_token_id=eos_idx, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=5,
            max_length=10,
            alpha=0,
            block_repeating_trigrams=False,
        )

        # To test that the minimum length is correctly enforced we constantly
        # assign the highest probability to the [EOS] token (and assign lower
        # probabilities to some other tokens).
        # Since BeamSearch will reset its probability to 1e-20 as long as
        # min_length has not been reached, we need to reset the value between
        # steps.
        non_eos_idxs = [4, 5, 1, 8, 9]
        score_distribution = torch.log_softmax(
            torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0
        )

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))
        log_probabilities[0, eos_idx] = score_distribution[0]
        for idx, score in zip(non_eos_idxs, score_distribution[1:]):
            log_probabilities[0, idx] = score

        for step in range(1, min_length + 2):
            log_probabilities[0, eos_idx] = score_distribution[0]

            # Beam #3 and #4 teminate at the first step since the probability
            # of the [EOS] token is -1e20 > -\infty so there are only two beams left.
            surviving_beams_rows = beam.grow(log_probabilities)
            if step < min_length:
                np.testing.assert_array_equal(
                    beam.growing_beams.numpy(),
                    np.repeat(np.array([[0] + [4] * step]), 2, axis=0),
                )
            elif step == min_length:
                np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([]))
                self.assertTrue(beam.is_done)
                break

            log_probabilities = log_probabilities.index_select(0, surviving_beams_rows)

    def test_beam_search_max_length(self):
        """ We keep predicting the same non-EOS token until we reach the
        maximum permitted length """
        batch_size = 3
        beam_size = 2
        max_length = 5
        vocab_size = 10

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(bos_token_id=0, eos_token_id=1, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=2,
            max_length=max_length,
            alpha=0,
            block_repeating_trigrams=False,
        )

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))

        # To test that beam search enforces the max length constraint we
        # keep giving the highest probability to a token that is not the
        # [EOS] token.
        # The beam search will stop at max_length-1, assuming that one would
        # add the [EOS] token at the end of the returned sequence.
        token_idxs = [3, 4, 5]
        score_distribution = torch.log_softmax(torch.tensor([10.0, 6.0, 4.0]), dim=0)
        for idx, score in zip(token_idxs, score_distribution):
            log_probabilities[:, idx] = score

        for step in range(1, max_length + 2):
            surviving_beams_rows = beam.grow(log_probabilities)
            if step + 1 < max_length:
                self.assertFalse(beam.is_done)
            elif step + 1 == max_length:  # Now [EOS] is the most probable token
                np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([]))
                self.assertTrue(beam.is_done)
                break

            log_probabilities = log_probabilities.index_select(0, surviving_beams_rows)

    def test_beam_search_block_repeating_trigrams(self):
        """ We make sure that the beams that contain repeating trigrams are removed. """
        batch_size = 3
        beam_size = 2
        max_length = 10
        vocab_size = 10

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(bos_token_id=0, eos_token_id=1, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=2,
            max_length=max_length,
            alpha=0,
            block_repeating_trigrams=True,
        )

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))

        # To test that BeamSearch enforces the 3-gram constraint we give the
        # highest probably to the same tokens in a cyclic fashion and make sure
        # they disappear once the cycle has completed.
        token_idxs = [3, 4, 5]
        score_distribution = torch.log_softmax(torch.tensor([10.0, 6.0, 4.0]), dim=0)
        for idx, score in zip(token_idxs, score_distribution):
            log_probabilities[:, idx] = score

        for step in range(1, max_length + 2):
            # Rotate the probabilities at each step
            for idx in token_idxs:
                score = score_distribution[(idx + step) % 3]
                log_probabilities[::beam_size, idx] = score

            surviving_beams_rows = beam.grow(log_probabilities)
            log_probabilities = log_probabilities.index_select(0, surviving_beams_rows)

            if step < 7:
                self.assertFalse(
                    np.array_equal(
                        log_probabilities.numpy()[0, :],
                        np.array([-1e20] * vocab_size, dtype="float32"),
                    )
                )
            if step == 7:
                np.testing.assert_array_equal(
                    log_probabilities.numpy()[0, :],
                    np.array([-1e20] * vocab_size, dtype="float32"),
                )

    def test_beam_search_example_for_one_step(self):
        """ We test that the predictions for one step of growth are correct. """
        batch_size = 2
        beam_size = 2
        max_length = 10
        vocab_size = 5

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(bos_token_id=0, eos_token_id=1, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=2,
            max_length=max_length,
            alpha=0,
            block_repeating_trigrams=False,
        )

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))
        log_probabilities[0, 3:] = torch.log_softmax(torch.tensor([2.0, 1.0]), dim=0)
        log_probabilities[2, 3:] = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)

        # First pass
        surviving_beams_rows = beam.grow(log_probabilities)
        np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([0, 0, 2, 2]))
        np.testing.assert_array_equal(
            beam.growing_beams.numpy(), np.array([[0, 3], [0, 4], [0, 4], [0, 3]])
        )
        self.assertFalse(beam.is_done)

        # Second pass
        surviving_beams_rows = beam.grow(log_probabilities)
        np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([0, 0, 2, 2]))
        np.testing.assert_array_equal(
            beam.growing_beams.numpy(),
            np.array([[0, 3, 3], [0, 3, 4], [0, 4, 4], [0, 4, 3]]),
        )
        self.assertFalse(beam.is_done)


if __name__ == "__name__":
    unittest.main()
