# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path

from transformers import is_torch_available
from transformers.model_debugging_utils import model_addition_debugger_context


if is_torch_available():
    import torch
    from torch import nn

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 4)
            self.linear_1 = nn.Linear(4, 8)
            self.linear_2 = nn.Linear(8, 2)
            self.act = nn.ReLU()

        def forward(self, input_ids: str):
            hidden_states = self.embed(input_ids).mean(dim=1)
            hidden_states = self.act(self.linear_1(hidden_states))
            return self.linear_2(hidden_states)

    class TestModelAdditionDebugger(unittest.TestCase):
        def setUp(self):
            self.model = ToyModel()
            self.inputs = {"input_ids": torch.randint(0, 10, (1, 3))}

        def tearDown(self):
            gc.collect()

        def test_debugger_outputs(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_addition_debugger_context(self.model, debug_path=str(tmpdir)):
                    _ = self.model.forward(**self.inputs)

                base = f"{self.model.__class__.__name__}_debug_tree"
                summary = Path(os.path.join(tmpdir, f"{base}_SUMMARY.json"))
                full = Path(os.path.join(tmpdir, f"{base}_FULL_TENSORS.json"))
                self.assertTrue(os.path.isfile(summary) and os.path.isfile(full))
                data = json.loads(summary.read_text())
                self.assertTrue({"module_path", "inputs", "children"} <= data.keys())
                self.assertTrue(data["children"])

    class ToyLayer(nn.Module):
        def __init__(self, layer_index):
            super().__init__()
            self.layer_index = layer_index
            self.layer_operation = nn.Linear(4, 4)

        def forward(self, hidden_states):
            return self.layer_operation(hidden_states)

    class ToyModelWithLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(4, 4)
            self.layers = nn.ModuleList([ToyLayer(layer_index) for layer_index in range(6)])
            self.output_proj = nn.Linear(4, 2)

        def forward(self, x):
            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)
            return self.output_proj(x)

    class TestModelWithLayers(unittest.TestCase):
        def setUp(self):
            self.inputs = {"input_ids": torch.randint(0, 10, (1, 3))}
            self.model_with_layers = ToyModelWithLayers()
            self.dense_input = {"x": torch.randn(1, 4)}

        def tearDown(self):
            gc.collect()

        def test_layer_pruning_behavior(self):
            # No pruning: expect all 6 layers
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_addition_debugger_context(self.model_with_layers, debug_path=tmpdir, do_prune_layers=False):
                    _ = self.model_with_layers(**self.dense_input)

                summary_path = os.path.join(tmpdir, "ToyModelWithLayers_debug_tree_SUMMARY.json")
                with open(summary_path) as f:
                    data = json.load(f)
                self.assertEqual(set(data.keys()), {"module_path", "inputs", "children"})
                for layer_index in range(6):
                    self.assertEqual(
                        data["children"][layer_index + 1]["module_path"],
                        f"ToyModelWithLayers.layers.{int(layer_index)}",
                    )

            # Pruning: expect only 2 layers (0 and 5)
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_addition_debugger_context(self.model_with_layers, debug_path=tmpdir, do_prune_layers=True):
                    _ = self.model_with_layers(**self.dense_input)

                summary_path = os.path.join(tmpdir, "ToyModelWithLayers_debug_tree_SUMMARY.json")
                with open(summary_path) as f:
                    data = json.load(f)
                self.assertEqual(set(data.keys()), {"module_path", "inputs", "children"})
                self.assertEqual(data["children"][1]["module_path"], "ToyModelWithLayers.layers.0")
                self.assertEqual(data["children"][2]["module_path"], "ToyModelWithLayers.layers.5")
