# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
    from transformers import FlavaProcessor
    from transformers.models.flava.image_processing_flava import (
        FLAVA_CODEBOOK_MEAN,
        FLAVA_CODEBOOK_STD,
        FLAVA_IMAGE_MEAN,
        FLAVA_IMAGE_STD,
    )


@require_vision
class FlavaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = FlavaProcessor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_map = {
            "image_mean": FLAVA_IMAGE_MEAN,
            "image_std": FLAVA_IMAGE_STD,
            "do_normalize": True,
            "do_resize": True,
            "size": 224,
            "do_center_crop": True,
            "crop_size": 224,
            "input_size_patches": 14,
            "total_mask_patches": 75,
            "mask_group_max_patches": None,
            "mask_group_min_patches": 16,
            "mask_group_min_aspect_ratio": 0.3,
            "mask_group_max_aspect_ratio": None,
            "codebook_do_resize": True,
            "codebook_size": 112,
            "codebook_do_center_crop": True,
            "codebook_crop_size": 112,
            "codebook_do_map_pixels": True,
            "codebook_do_normalize": True,
            "codebook_image_mean": FLAVA_CODEBOOK_MEAN,
            "codebook_image_std": FLAVA_CODEBOOK_STD,
        }

        image_processor = image_processor_class(**image_processor_map)
        return image_processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ",", "low", "lowest"]  # fmt: skip
        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as fp:
            fp.write("".join([x + "\n" for x in vocab_tokens]))

        return tokenizer_class.from_pretrained(cls.tmpdirname)
