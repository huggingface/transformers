# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import os
import unittest
from pathlib import Path


_GOLDEN = Path("/tmp/gliner2-golden")
_SMALL = _GOLDEN / "fastino--gliner2.5-small-v1" / "inference.json"
_TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-small-v1/snapshots/df5910e44bc4ffdb0d95399a83b0ca4516349aa5"
)
_TEXT = "Ada Lovelace wrote notes about the analytical engine in London."
_SCHEMA = {"entities": ["person", "work", "location"]}


class Gliner2GoldenTest(unittest.TestCase):
    """Local golden check. Skips when ``/tmp/gliner2-golden`` is absent."""

    def test_ada_lovelace_entity_input_ids(self):
        """Match golden input ids for the Ada Lovelace entity schema."""
        if not _GOLDEN.is_dir():
            self.skipTest("/tmp/gliner2-golden is absent")
        self.assertTrue(_SMALL.is_file(), "small-v1 inference.json is missing")
        tokenizer = _TOKENIZER if (_TOKENIZER / "tokenizer.json").is_file() else None
        if tokenizer is None:
            matches = list(_TOKENIZER.rglob("tokenizer.json")) if _TOKENIZER.is_dir() else []
            tokenizer = matches[0].parent if matches else None
        if tokenizer is None:
            self.skipTest("small-v1 tokenizer snapshot is absent")
        os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import Gliner2Processor

        processor = Gliner2Processor.from_pretrained(tokenizer, local_files_only=True)
        encoded = processor(_TEXT, schema=_SCHEMA, return_tensors="pt")
        golden = json.loads(_SMALL.read_text())
        self.assertEqual(encoded["input_ids"][0].tolist(), golden["processor"]["input_ids"])
