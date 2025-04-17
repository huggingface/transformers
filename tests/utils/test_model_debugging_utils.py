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

from transformers import is_torch_available, is_vision_available
from transformers.model_debugging_utils import model_addition_debugger_context


if is_vision_available():
    pass

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
