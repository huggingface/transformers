# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from transformers import is_torch_available
from transformers.testing_utils import require_torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


if is_torch_available():
    import torch

    from transformers.models.apertus1p5 import convert_apertus1p5_weights_to_hf as conversion


@require_torch
class Apertus1p5ConversionTest(unittest.TestCase):
    def test_convert_removes_stale_canonical_weight_files(self):
        config = Mock(tie_word_embeddings=False)
        converted_weights = {"lm_head.weight": torch.ones(2, 2)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for filename in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME):
                (output_dir / filename).write_bytes(b"stale")
            old_shard = output_dir / "model-old-00001-of-00001.safetensors"
            old_shard.write_bytes(b"unreferenced")

            with (
                patch.object(conversion, "build_config", return_value=config),
                patch.object(
                    conversion,
                    "remapped_sources",
                    return_value=[("apertus", "model.safetensors", converted_weights)],
                ),
                patch.object(conversion, "write_processor"),
            ):
                conversion.convert("apertus", "vision", "audio", str(output_dir))

            self.assertFalse((output_dir / SAFE_WEIGHTS_NAME).exists())
            self.assertFalse((output_dir / WEIGHTS_NAME).exists())
            self.assertFalse((output_dir / WEIGHTS_INDEX_NAME).exists())
            self.assertTrue(old_shard.exists())

            with (output_dir / SAFE_WEIGHTS_INDEX_NAME).open() as f:
                index = json.load(f)
            self.assertEqual(index["weight_map"], {"lm_head.weight": "model-apertus-model.safetensors"})
            self.assertTrue((output_dir / "model-apertus-model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
