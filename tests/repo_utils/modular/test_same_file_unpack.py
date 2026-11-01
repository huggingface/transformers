import os
import sys
import unittest

import libcst as cst


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from modular_model_converter import ModularFileMapper, create_modules  # noqa: E402


MODULAR_CODE = """
from ..llama.modeling_llama import LlamaAttention


class DummyAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.foo = 1


class DummyKVSharedAttention(DummyAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.k_proj
"""


class SameFileUnpackTest(unittest.TestCase):
    def test_same_file_parent_is_unpacked(self):
        module = cst.parse_module(MODULAR_CODE)
        mapper = ModularFileMapper(module, "dummy", "transformers")
        module.visit(mapper)
        code = create_modules(mapper)["modeling"].code
        child = code[code.index("class DummyKVSharedAttention") :]

        self.assertTrue(child.startswith("class DummyKVSharedAttention(nn.Module):"))
        self.assertIn("self.foo = 1", child)
        self.assertIn("def forward(", child)
        self.assertNotIn("self.k_proj =", child)
        self.assertNotIn("del self.k_proj", child)
