# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import AutoProcessor, BartTokenizerFast, Florence2Processor
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

    def test_post_process_parse_description_with_bboxes_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text_without_phrase = "</s><s><loc_53><loc_334><loc_933><loc_775><loc_711><loc_203><loc_906><loc_546><loc_585><loc_309><loc_774><loc_709><loc_577></s><pad>"
        image_size = (1000, 1000)
        parsed_text_without_phrase = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(text_without_phrase, image_size=image_size, allow_empty_phrase=True)
        EXPECTED_PARSED_TEXT_WITHOUT_PHRASE = [
            {'bbox': [53.5, 334.5, 933.5, 775.5], 'cat_name': ''},
            {'bbox': [711.5, 203.5, 906.5, 546.5], 'cat_name': ''},
            {'bbox': [585.5, 309.5, 774.5, 709.5], 'cat_name': ''},
        ]
        self.assertEqual(parsed_text_without_phrase, EXPECTED_PARSED_TEXT_WITHOUT_PHRASE)

        text_with_phrase = "</s><s>car<loc_53><loc_334><loc_933><loc_775>door handle<loc_425><loc_504><loc_474><loc_516></s><pad>"
        image_size = (1000, 1000)
        parsed_text_with_phrase = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(text_with_phrase, image_size=image_size, allow_empty_phrase=False)
        EXPECTED_PARSED_TEXT_WITH_PHRASE = [{'bbox': [53.5, 334.5, 933.5, 775.5], 'cat_name': 'car'},
        {'bbox': [425.5, 504.5, 474.5, 516.5], 'cat_name': 'door handle'}]
        self.assertEqual(parsed_text_with_phrase, EXPECTED_PARSED_TEXT_WITH_PHRASE)

    def test_post_process_parse_description_with_polygons_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text_without_phrase = "<loc_279><loc_379><loc_282><loc_379><loc_290><loc_373><loc_293><loc_373><loc_298><loc_369><loc_301><loc_369>"
        image_size = (1000, 1000)
        parsed_text_without_phrase = processor.post_processor.parse_description_with_polygons_from_text_and_spans(text_without_phrase, image_size=image_size, allow_empty_phrase=True)
        EXPECTED_PARSED_TEXT_WITHOUT_PHRASE = [{'cat_name': '',
            'polygons': [[279.5,
                379.5,
                282.5,
                379.5,
                290.5,
                373.5,
                293.5,
                373.5,
                298.5,
                369.5,
                301.5,
                369.5]]}]
        self.assertEqual(parsed_text_without_phrase, EXPECTED_PARSED_TEXT_WITHOUT_PHRASE)

        text_with_phrase = "object<loc_769><loc_248><loc_771><loc_234><loc_773><loc_206><loc_773><loc_198><loc_771><loc_193>"
        image_size = (1000, 1000)
        parsed_text_with_phrase = processor.post_processor.parse_description_with_polygons_from_text_and_spans(text_with_phrase, image_size=image_size, allow_empty_phrase=False)
        EXPECTED_PARSED_TEXT_WITH_PHRASE = [{'cat_name': 'object',
            'polygons': [[769.5,
                248.5,
                771.5,
                234.5,
                773.5,
                206.5,
                773.5,
                198.5,
                771.5,
                193.5]]}]
        self.assertEqual(parsed_text_with_phrase, EXPECTED_PARSED_TEXT_WITH_PHRASE)
