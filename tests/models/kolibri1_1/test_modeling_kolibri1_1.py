# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Kolibri1_1 model."""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from safetensors.torch import load_file, save_file

from transformers import is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import Kolibri1_1Config, Kolibri1_1ForCausalLM, Kolibri1_1Model


class Kolibri1_1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1_1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Covers all K/V sharing modes: a layer whose K/V other layers read, layers reading them, and layers with
        # their own K/V that no other layer reads
        self.num_hidden_layers = 5
        self.layer_types = ["sliding_attention"] + ["full_attention"] * 4
        self.kv_source_layers = [0, 1, 1, 1, 4]
        # A dense MLP layer in front, like the checkpoints of Kolibri 1.1
        self.mlp_only_layers = [0]
        self.mlp_only_intermediate_size = 40


@require_torch
class Kolibri1_1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Kolibri1_1ModelTester

    def test_invalid_layouts(self):
        kwargs = {"num_hidden_layers": 4, "layer_types": ["sliding_attention", "full_attention"] * 2}
        Kolibri1_1Config(**kwargs)  # valid without the overrides below
        invalid = [
            ({"layer_types": ["sliding_attention", "full_attention", "linear_attention", "full_attention"]}, "Unsupported layer types"),
            ({"kv_source_layers": [0, 1, 2]}, "`kv_source_layers` needs one entry per layer"),
            ({"kv_source_layers": [0, 1, 1, 1]}, "Sliding-attention layer 2 cannot read the K/V of another layer"),
            ({"kv_source_layers": [0, 3, 2, 3]}, "Layer 1 must read the K/V of an earlier layer"),
            ({"kv_source_layers": [0, 0, 2, 3]}, "Layer 1 must read the K/V of a layer with the same attention type"),
            ({"mlp_only_layers": [0]}, "`mlp_only_layers` need a positive `mlp_only_intermediate_size`"),
            ({"mlp_only_layers": [4], "mlp_only_intermediate_size": 8}, "`mlp_only_layers` must be layer indices"),
            ({"mlp_only_layers": [0], "mlp_only_intermediate_size": -8}, "`mlp_only_layers` need a positive"),
        ]  # fmt: skip
        for overrides, message in invalid:
            with self.subTest(**overrides), self.assertRaisesRegex(StrictDataclassClassValidationError, message):
                Kolibri1_1Config(**{**kwargs, **overrides})

    def test_kv_shared_layers_have_no_kv_weights(self):
        config = self.model_tester.get_config()
        model = Kolibri1_1Model(config)
        for layer_idx, layer in enumerate(model.layers):
            shared = config.kv_source_layers[layer_idx] != layer_idx
            for name in ("k_proj", "v_proj", "k_norm"):
                self.assertEqual(hasattr(layer.self_attn, name), not shared)
        self.assertEqual(config.kv_sharing_modes, [None, "store", "reuse", "reuse", None])

    def test_shared_kv_checkpoint_weights_are_ignored(self):
        # Checkpoints exported from training keep the unused key/value projections of K/V-sharing layers
        config = self.model_tester.get_config()
        for model_class, prefix in ((Kolibri1_1Model, ""), (Kolibri1_1ForCausalLM, "model.")):
            with self.subTest(model_class.__name__), tempfile.TemporaryDirectory() as tmp_dir:
                model_class(config).save_pretrained(tmp_dir)
                state_dict = load_file(f"{tmp_dir}/model.safetensors")
                for name in ("k_proj.weight", "v_proj.weight"):
                    state_dict[f"{prefix}layers.2.self_attn.{name}"] = state_dict[
                        f"{prefix}layers.1.self_attn.{name}"
                    ].clone()
                save_file(state_dict, f"{tmp_dir}/model.safetensors", metadata={"format": "pt"})
                _, loading_info = model_class.from_pretrained(tmp_dir, output_loading_info=True)
                self.assertEqual(loading_info["unexpected_keys"], set())
                self.assertEqual(loading_info["missing_keys"], set())
