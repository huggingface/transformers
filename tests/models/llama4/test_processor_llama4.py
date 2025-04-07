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
from typing import Optional

from transformers import AutoProcessor, Llama4Processor, PreTrainedTokenizerFast
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Llama4ImageProcessorFast


@require_vision
class Llama4ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Llama4Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = Llama4ImageProcessorFast(max_patches=1, size={"height": 20, "width": 20})
        tokenizer = PreTrainedTokenizerFast.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit")
        processor_kwargs = self.prepare_processor_dict()
        processor = Llama4Processor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # Override as Llama4ProcessorProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <image>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <image>"]
        return ["lower newer <image>", "<image> upper older longer string"] + ["<image> lower newer"] * (
            batch_size - 2
        )
