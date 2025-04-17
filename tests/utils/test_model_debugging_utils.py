
import unittest

import numpy as np

from transformers import is_torch_available, is_vision_available
from transformers.processing_utils import _validate_images_text_input_order
from transformers.testing_utils import require_torch, require_vision
from pathlib import Path
from transformers.model_debugging_utils import model_addition_debugger_context
import unittest
import tempfile
import shutil
import gc
import json

if is_vision_available():
    import PIL

if is_torch_available():
    import torch
    from torch import nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
        self.act = nn.ReLU()

    def forward(self, input_ids: str):
        x = self.embed(input_ids).mean(dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class TestModelAdditionDebugger(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmpdirname)
        self.tmp_path.mkdir(exist_ok=True)
        self.model = ToyModel()
        self.inputs = {"input_ids": torch.randint(0, 10, (1, 3))}

    def tearDown(self):
        for f in self.tmp_path.glob("*_debug_tree_*"):
            f.unlink()
        self.tmp_path.rmdir()
        gc.collect()

    def test_debugger_outputs(self):
        with model_addition_debugger_context(self.model, debug_path=str(self.tmp_path)):
            _ = self.model.forward(**self.inputs)

        base = f"{self.model.__class__.__name__}_debug_tree"
        summary = self.tmp_path / f"{base}_SUMMARY.json"
        full = self.tmp_path / f"{base}_FULL_TENSORS.json"

        self.assertTrue(summary.exists() and full.exists())
        data = json.loads(summary.read_text())
        self.assertTrue({"module_path", "inputs", "outputs", "children"} <= data.keys())
        self.assertTrue(data["children"])