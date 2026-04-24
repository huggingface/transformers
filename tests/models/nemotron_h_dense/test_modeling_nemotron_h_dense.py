# Copyright 2024-2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NemotronHDense model."""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import NemotronHDenseConfig, is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import NemotronHDenseModel


class NemotronHDenseModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = NemotronHDenseModel

    def __init__(
        self,
        parent,
        hybrid_override_pattern="M-M-*-",
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=40,
        mamba_num_heads=8,
        mamba_head_dim=16,
        n_groups=1,
        ssm_state_size=16,
        chunk_size=4,
        **kwargs,
    ):
        super().__init__(
            parent=parent,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            **kwargs,
        )
        self.hybrid_override_pattern = hybrid_override_pattern
        # `-` is absorbed as an FFN tail; each `M` / `*` is one logical decoder layer.
        self.num_hidden_layers = sum(1 for c in hybrid_override_pattern if c in ("M", "*"))
        self.head_dim = mamba_head_dim
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.n_groups = n_groups
        self.ssm_state_size = ssm_state_size
        self.chunk_size = chunk_size
        self.use_mamba_kernels = False


@require_torch
class NemotronHDenseModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_tester_class = NemotronHDenseModelTester

    @unittest.skip(reason="NemotronHDense has a hybrid mamba/attention cache.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="Hybrid mamba/attention cache continuation is handled elsewhere.")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(reason="Mamba2 layers don't support padding via position_ids.")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Hybrid cache has non-standard layer shapes checked elsewhere.")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(reason="Hybrid cache generate-with-cache validated via separate tests.")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="Hybrid cache generate-with-cache validated via separate tests.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="NemotronHDense needs at least 3 layers (mamba/mamba/attention).")
    def test_num_layers_is_small(self):
        pass

    def test_attention_outputs(self):
        """Only hybrid layers produce attention weights — count must match `*` chars."""
        import torch

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        expected_num_attentions = config.hybrid_override_pattern.count("*")

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            model.to(torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)

    def test_hybrid_override_pattern_validation(self):
        """Dense pattern accepts M / * / - and rejects E (MoE)."""
        config = NemotronHDenseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="M-M-*-")
        self.assertEqual(config.layer_types, ["mamba", "mamba", "attention"])
        self.assertEqual(config.num_hidden_layers, 3)

        with self.assertRaises((ValueError, StrictDataclassClassValidationError)):
            NemotronHDenseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="M*E")

    def test_config_roundtrip_save_load(self):
        config1 = NemotronHDenseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="M-*-M-")

        with tempfile.TemporaryDirectory() as tmpdir:
            config1.save_pretrained(tmpdir)
            config2 = NemotronHDenseConfig.from_pretrained(tmpdir)

            self.assertEqual(config2.hybrid_override_pattern, "M-*-M-")
            self.assertEqual(config2.num_hidden_layers, 3)
            self.assertEqual(config2.vocab_size, 100)
            self.assertEqual(config2.hidden_size, 32)
