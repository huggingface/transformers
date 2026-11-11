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

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import Kolibri1_1Config, Kolibri1_1Model


class Kolibri1_1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1_1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["sliding_attention", "full_attention"]
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
            ({"mlp_only_layers": [0]}, "`mlp_only_layers` need a positive `mlp_only_intermediate_size`"),
            ({"mlp_only_layers": [4], "mlp_only_intermediate_size": 8}, "`mlp_only_layers` must be layer indices"),
            ({"mlp_only_layers": [0], "mlp_only_intermediate_size": -8}, "`mlp_only_layers` need a positive"),
        ]  # fmt: skip
        for overrides, message in invalid:
            with self.subTest(**overrides), self.assertRaisesRegex(StrictDataclassClassValidationError, message):
                Kolibri1_1Config(**{**kwargs, **overrides})
