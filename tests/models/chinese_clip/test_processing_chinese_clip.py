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

from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import ChineseCLIPProcessor


@require_vision
class ChineseCLIPProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ChineseCLIPProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "的",
            "价",
            "格",
            "是",
            "15",
            "便",
            "alex",
            "##andra",
            "，",
            "。",
            "-",
            "t",
            "shirt",
        ]
        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        return tokenizer_class(vocab_file=vocab_file)

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_map = {
            "do_resize": True,
            "size": {"height": 224, "width": 224},
            "do_center_crop": True,
            "crop_size": {"height": 18, "width": 18},
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "do_convert_rgb": True,
        }
        return image_processor_class(**image_processor_map)
