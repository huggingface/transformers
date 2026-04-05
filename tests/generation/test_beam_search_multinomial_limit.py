# Copyright 2025 The HuggingFace Team Inc.
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

from transformers import GPT2Config, GPT2LMHeadModel, is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    import transformers.generation.utils as gen_utils


@require_torch
class BeamSearchMultinomialLimitTest(unittest.TestCase):
    def test_get_top_k_continuations_gumbel_when_flat_dim_over_limit(self):
        """Above _BEAM_SEARCH_MULTINOMIAL_DIM_LIMIT, avoid torch.multinomial on the full flat dim (#45245)."""
        old_limit = gen_utils._BEAM_SEARCH_MULTINOMIAL_DIM_LIMIT
        try:
            gen_utils._BEAM_SEARCH_MULTINOMIAL_DIM_LIMIT = 16
            config = GPT2Config(
                vocab_size=100,
                n_positions=64,
                n_embd=32,
                n_layer=1,
                n_head=1,
                bos_token_id=0,
                eos_token_id=0,
            )
            model = GPT2LMHeadModel(config)
            model.to(torch_device)
            model.eval()

            batch_size, num_beams, vocab_size = 1, 2, 10
            flat = num_beams * vocab_size
            self.assertGreaterEqual(flat, gen_utils._BEAM_SEARCH_MULTINOMIAL_DIM_LIMIT)

            accumulated = torch.randn(batch_size, flat, device=torch_device, dtype=torch.float32)
            max_length = 8
            cur_len, decoder_prompt_len = 2, 2
            running_sequences = torch.zeros(batch_size, num_beams, max_length, dtype=torch.long, device=torch_device)
            running_beam_indices = torch.zeros(
                batch_size, num_beams, max_length - decoder_prompt_len, dtype=torch.int32, device=torch_device
            )
            beams_to_keep = 4

            topk_log_probs, topk_running_sequences, topk_running_beam_indices = model._get_top_k_continuations(
                accumulated_log_probs=accumulated,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=True,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )
            self.assertEqual(topk_log_probs.shape, (batch_size, beams_to_keep))
            self.assertEqual(topk_running_sequences.shape[1], beams_to_keep)
            self.assertEqual(topk_running_beam_indices.shape[1], beams_to_keep)
        finally:
            gen_utils._BEAM_SEARCH_MULTINOMIAL_DIM_LIMIT = old_limit
