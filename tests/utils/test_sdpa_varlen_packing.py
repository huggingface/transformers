# Copyright 2025 HuggingFace Inc.
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
"""
Tests for the `sdpa` packed-sequence `varlen_attn` fast path and its block-diagonal-mask fallback
(`integrations/sdpa_attention.py` and the `allow_torch_varlen_skip` signal in `masking_utils.py`).
"""

import unittest
from unittest import mock

from transformers.testing_utils import require_torch_gpu, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import LlamaConfig, LlamaModel
    from transformers.integrations import sdpa_attention


def _tiny_llama_config() -> "LlamaConfig":
    # Tiny random-weight config; GQA (heads != kv_heads) also exercises the `enable_gqa` passthrough.
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attn_implementation="sdpa",
    )


def _build_packed_inputs(doc_lengths: list[int], vocab_size: int, device: str):
    position_ids = torch.cat([torch.arange(length, device=device) for length in doc_lengths]).unsqueeze(0)
    total_len = position_ids.shape[1]
    input_ids = torch.randint(0, vocab_size, (1, total_len), device=device)
    return input_ids, position_ids


@require_torch_gpu
class SdpaVarlenPackingTest(unittest.TestCase):
    def setUp(self):
        if not sdpa_attention._is_torch_varlen_attn_available:
            self.skipTest("torch.nn.attention.varlen.varlen_attn is unavailable (needs torch>=2.10)")
        torch.manual_seed(0)
        self.doc_lengths = [6, 5, 7]
        self.config = _tiny_llama_config()
        self.model = LlamaModel(self.config).to(device=torch_device, dtype=torch.bfloat16).eval()
        self.input_ids, self.position_ids = _build_packed_inputs(
            self.doc_lengths, self.config.vocab_size, torch_device
        )

    def _per_document_reference(self) -> "torch.Tensor":
        """Run each document on its own and concatenate the results -- the ground truth that the packed forward
        must match, since packing must be numerically transparent."""
        outputs = []
        offset = 0
        for length in self.doc_lengths:
            doc_input_ids = self.input_ids[:, offset : offset + length]
            doc_position_ids = torch.arange(length, device=torch_device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(
                    input_ids=doc_input_ids,
                    position_ids=doc_position_ids,
                    attention_mask=None,
                    use_cache=False,
                ).last_hidden_state
            outputs.append(out)
            offset += length
        return torch.cat(outputs, dim=1)

    def _packed_forward(self):
        with torch.no_grad():
            return self.model(
                input_ids=self.input_ids,
                position_ids=self.position_ids,
                attention_mask=None,
                use_cache=False,
            ).last_hidden_state

    def test_varlen_fast_path_matches_per_document_reference(self):
        """The eligible packed row must route to `varlen_attn` and match the per-document reference."""
        with mock.patch.object(sdpa_attention, "varlen_attn", wraps=sdpa_attention.varlen_attn) as varlen_attn_spy:
            packed_output = self._packed_forward()
            self.assertGreater(
                varlen_attn_spy.call_count,
                0,
                "varlen_attn was never invoked for an eligible packed row; the fast path did not engage.",
            )

        reference_output = self._per_document_reference()
        torch.testing.assert_close(packed_output, reference_output, atol=2e-2, rtol=2e-2)

    def test_fallback_block_diagonal_mask_matches_per_document_reference(self):
        """When the packed row is ineligible for the fast path (forced here), `sdpa_attention_forward` must
        rebuild an equivalent block-diagonal mask from `position_ids` and still isolate documents correctly."""
        with mock.patch.object(sdpa_attention, "_is_torch_varlen_attn_available", False):
            with mock.patch.object(sdpa_attention, "varlen_attn", wraps=sdpa_attention.varlen_attn) as varlen_attn_spy:
                packed_output = self._packed_forward()
                self.assertEqual(
                    varlen_attn_spy.call_count,
                    0,
                    "varlen_attn should not be invoked once the fast path is made ineligible.",
                )

        reference_output = self._per_document_reference()
        torch.testing.assert_close(packed_output, reference_output, atol=2e-2, rtol=2e-2)

    def test_masking_utils_skips_mask_for_eligible_packed_row(self):
        """`create_causal_mask` must return `None` for an eligible packed `sdpa` row, letting
        `sdpa_attention_forward` own document isolation."""
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=torch.empty((1, self.position_ids.shape[1]), dtype=torch.bfloat16, device=torch_device),
            attention_mask=None,
            past_key_values=None,
            position_ids=self.position_ids,
        )
        self.assertIsNone(causal_mask)


if __name__ == "__main__":
    unittest.main()
