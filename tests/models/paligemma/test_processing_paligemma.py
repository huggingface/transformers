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

from transformers import AutoProcessor, GemmaTokenizerFast, PaliGemmaProcessor
from transformers.testing_utils import require_read_token, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import SiglipImageProcessor


@require_vision
@require_read_token
class PaliGemmaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PaliGemmaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SiglipImageProcessor(do_center_crop=False)
        tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-7b")
        image_processor.image_seq_length = 32

        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_text_with_image_tokens(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        text_multi_images = "<image><image><bos>Dummy text!"
        text_single_image = "<image><bos>Dummy text!"
        text_noimage = "Dummy text!"

        image = self.prepare_image_inputs()[0]

        out_noimage = processor(text=text_noimage, images=image, return_tensors="np")
        out_singlimage = processor(text=text_singlimage, images=image, return_tensors="np")
        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_singlimage[k].tolist())

        out_multiimages = processor(text=text_multiimages, images=[image, image], return_tensors="np")
        out_noimage = processor(text=text_noimage, images=[[image, image]], return_tensors="np")

        # We can't be sure what is users intention, whether user want "one text + two images" or user forgot to add the second text
        with self.assertRaises(ValueError):
            out_noimage = processor(text=text_noimage, images=[image, image], return_tensors="np")

        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_multiimages[k].tolist())

        text_batched = ["Dummy text!", "Dummy text!"]
        text_batched_with_image = ["<image><bos>Dummy text!", "<image><bos>Dummy text!"]
        out_images = processor(text=text_batched_with_image, images=[image, image], return_tensors="np")
        out_noimage_nested = processor(text=text_batched, images=[[image], [image]], return_tensors="np")
        out_noimage = processor(text=text_batched, images=[image, image], return_tensors="np")
        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_images[k].tolist() == out_noimage_nested[k].tolist())
