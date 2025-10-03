# Copyright 2025 The HRM Team and HuggingFace Inc. team. All rights reserved.
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
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for HRM checkpoint conversion script."""

import unittest

from transformers import is_torch_available
from transformers.models.hrm.convert_hrm_checkpoint_to_hf import convert_state_dict
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch


@require_torch
class HrmConversionTest(unittest.TestCase):
    """Test HRM checkpoint conversion functionality."""

    def test_convert_embedding_keys(self):
        """Test conversion of embedding layer keys."""
        # Test both possible key formats map to the same output
        state_dict_1 = {"embedding.weight": torch.randn(11, 512)}
        converted_1 = convert_state_dict(state_dict_1.copy())
        self.assertIn("model.inner.embeddings.weight", converted_1)

        state_dict_2 = {"embeddings.weight": torch.randn(11, 512)}
        converted_2 = convert_state_dict(state_dict_2.copy())
        self.assertIn("model.inner.embeddings.weight", converted_2)

    def test_convert_puzzle_embedding_keys(self):
        """Test conversion of puzzle embedding keys."""
        original_state_dict = {
            "puzzle_emb.weight": torch.randn(10, 128),
            "puzzle_embeddings.weight": torch.randn(10, 128),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.puzzle_emb.weight
        self.assertIn("model.puzzle_emb.weight", converted)

    def test_convert_h_level_keys(self):
        """Test conversion of H-level (high-level) module keys."""
        original_state_dict = {
            "H_level.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "h_level.layers.1.mlp.gate_up_proj.weight": torch.randn(2048, 512),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.inner.H_level.*
        self.assertIn("model.inner.H_level.layers.0.self_attn.q_proj.weight", converted)
        self.assertIn("model.inner.H_level.layers.1.mlp.gate_up_proj.weight", converted)

    def test_convert_l_level_keys(self):
        """Test conversion of L-level (low-level) module keys."""
        original_state_dict = {
            "L_level.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
            "l_level.layers.1.norm_eps": torch.tensor(1e-5),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.inner.L_level.*
        self.assertIn("model.inner.L_level.layers.0.self_attn.k_proj.weight", converted)
        self.assertIn("model.inner.L_level.layers.1.norm_eps", converted)

    def test_convert_act_mechanism_keys(self):
        """Test conversion of ACT (Adaptive Computation Time) mechanism keys."""
        original_state_dict = {
            "q_halt.weight": torch.randn(1, 512),
            "q_halt.bias": torch.randn(1),
            "q_continue.weight": torch.randn(1, 512),
            "q_continue.bias": torch.randn(1),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.q_halt and model.q_continue
        self.assertIn("model.q_halt.weight", converted)
        self.assertIn("model.q_halt.bias", converted)
        self.assertIn("model.q_continue.weight", converted)
        self.assertIn("model.q_continue.bias", converted)

    def test_convert_lm_head_keys(self):
        """Test conversion of language modeling head keys."""
        original_state_dict = {
            "lm_head.weight": torch.randn(11, 512),
            "output.weight": torch.randn(11, 512),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.inner.lm_head.weight
        self.assertIn("model.inner.lm_head.weight", converted)

    def test_convert_init_states(self):
        """Test conversion of initial carry state keys."""
        original_state_dict = {
            "H_init": torch.randn(1, 1, 512),
            "L_init": torch.randn(1, 1, 512),
        }

        converted = convert_state_dict(original_state_dict.copy())

        # Should be converted to model.H_init and model.L_init
        self.assertIn("model.H_init", converted)
        self.assertIn("model.L_init", converted)

    def test_preserve_tensor_values(self):
        """Test that tensor values are preserved during conversion."""
        original_weight = torch.randn(512, 512)
        original_state_dict = {
            "H_level.layers.0.weight": original_weight.clone(),
        }

        converted = convert_state_dict(original_state_dict)

        # Check that the tensor values are identical
        converted_weight = converted["model.inner.H_level.layers.0.weight"]
        self.assertTrue(torch.allclose(original_weight, converted_weight))

    def test_no_duplicate_model_prefix(self):
        """Test that keys already prefixed with 'model.' are not duplicated."""
        original_state_dict = {
            "model.inner.H_level.layers.0.weight": torch.randn(512, 512),
        }

        converted = convert_state_dict(original_state_dict)

        # Should not add another 'model.' prefix
        self.assertIn("model.inner.H_level.layers.0.weight", converted)
        self.assertNotIn("model.model.inner.H_level.layers.0.weight", converted)

    def test_full_checkpoint_structure(self):
        """Test conversion of a complete checkpoint structure."""
        original_state_dict = {
            # Embeddings
            "embedding.weight": torch.randn(11, 512),
            "puzzle_emb.weight": torch.randn(10, 128),
            # H-level module
            "H_level.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "H_level.layers.0.mlp.gate_up_proj.weight": torch.randn(2048, 512),
            # L-level module
            "L_level.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
            "L_level.layers.0.mlp.down_proj.weight": torch.randn(512, 1024),
            # ACT mechanism
            "q_halt.weight": torch.randn(1, 512),
            "q_halt.bias": torch.randn(1),
            "q_continue.weight": torch.randn(1, 512),
            "q_continue.bias": torch.randn(1),
            # LM head
            "lm_head.weight": torch.randn(11, 512),
            # Init states
            "H_init": torch.randn(1, 1, 512),
            "L_init": torch.randn(1, 1, 512),
        }

        converted = convert_state_dict(original_state_dict)

        # Verify all keys were converted
        self.assertEqual(len(converted), len(original_state_dict))

        # Verify no original keys remain (all should be converted)
        for key in original_state_dict.keys():
            self.assertNotIn(key, converted)

        # Verify all converted keys have model prefix
        for key in converted.keys():
            self.assertTrue(key.startswith("model."))
