# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import os
import unittest

from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES, BertTokenizer
from transformers.testing_utils import require_tokenizers, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import VisionTextDualEncoderProcessor, ViTImageProcessorFast


@require_tokenizers
@require_vision
class VisionTextDualEncoderProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VisionTextDualEncoderProcessor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_map = {
            "do_resize": True,
            "size": {"height": 18, "width": 18},
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        return ViTImageProcessorFast(**image_processor_map)

    @classmethod
    def _setup_tokenizer(cls):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ",", "low", "lowest"]  # fmt: skip
        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        return BertTokenizer.from_pretrained(cls.tmpdirname)
