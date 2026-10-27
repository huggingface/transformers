# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        BridgeTowerImageProcessor,
        BridgeTowerProcessor,
        RobertaTokenizer,
    )


@require_vision
class BridgeTowerProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = BridgeTowerProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

    def test_load_legacy_checkpoint_without_image_processor_type(self):
        # The `BridgeTower/*` Hub checkpoints ship a `preprocessor_config.json` with no `image_processor_type` and a
        # `config.json` with no `model_type`, and no `tokenizer_config.json`: the processor class must resolve its own
        # image processor and tokenizer instead of failing or falling back to a generic tokenizer.
        processor = self.get_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            for file_name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
                if os.path.isfile(os.path.join(tmpdir, file_name)):
                    os.remove(os.path.join(tmpdir, file_name))
            with open(os.path.join(tmpdir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
                json.dump({"size": 294}, f)
            with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"hidden_size": 768}, f)

            loaded = BridgeTowerProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(loaded.image_processor, BridgeTowerImageProcessor)
        self.assertEqual(loaded.image_processor.size["shortest_edge"], 294)
        self.assertIsInstance(loaded.tokenizer, RobertaTokenizer)
        self.assertEqual(loaded.tokenizer.pad_token, "<pad>")
