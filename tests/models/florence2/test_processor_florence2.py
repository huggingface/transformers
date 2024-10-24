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

from transformers import AutoProcessor, AutoTokenizer, BartTokenizerFast, Florence2Processor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPImageProcessor


@require_vision
class Florence2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Florence2Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor.from_pretrained("microsoft/Florence-2-base")
        tokenizer = BartTokenizerFast.from_pretrained("microsoft/Florence-2-base")
        processor_kwargs = self.prepare_processor_dict()
        processor = Florence2Processor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    @unittest.skip(
        "Skip because the model has no processor kwargs except for chat template and"
        "chat template is saved as a separate file. Stop skipping this test when the processor"
        "has new kwargs saved in config file."
    )
    def test_processor_to_json_string(self):
        pass

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["microsoft/Florence-2-base"]:
            processor = Florence2Processor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)
