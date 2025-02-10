# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

from transformers import AutoProcessor, AutoTokenizer, InternVLProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import GotOcr2ImageProcessor


@require_vision
class InternVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InternVLProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = GotOcr2ImageProcessor(
            do_resize=True,
            size={"height": 448, "width": 448},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            do_center_crop=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("../InternVLTest-1B")
        processor_kwargs = self.prepare_processor_dict()
        processor = InternVLProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
