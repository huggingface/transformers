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

import unittest

from transformers import (
    MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
    BertTokenizer,
    ViltFeatureExtractor,
    ViltForQuestionAnswering,
    PreTrainedTokenizer,
    is_vision_available,
)
from transformers.pipelines import VisualQuestionAnsweringPipeline, VisualQuestionAnsweringArgumentHandler, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_torch,
    require_tf,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass

@is_pipeline_test
class VisualQuestionAnsweringArgumentHandlerTests(unittest.TestCase):
    def test_argument_handler(self):
        qa = VisualQuestionAnsweringArgumentHandler()
        image1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        query = "How many cats are there?"

        for image in [image1, image2]:
            output = qa(images=image, queries=query)
            self.assertEqual(type(output), list)
            self.assertEqual(len(output), 1)
            for item in output:
                self.assertEqual(type(item["image"]), Image.Image)
                self.assertEqual(type(item["query"]), str)

        output = qa(images=[image1, image2], queries=[query, query])
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 2)
        for item in output:
            self.assertEqual(type(item["image"]), Image.Image)
            self.assertEqual(type(item["query"]), str)

    def test_argument_handler_error_handling(self):
        qa = VisualQuestionAnsweringArgumentHandler()
        image1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        query = "How many cats are there?"

        with self.assertRaises(ValueError):
            qa(images=[image1, image2], queries=query)
            qa(images=[image1, image2], queries=[query])
            qa(images=image1, queries=[query, query])
            qa(images=[image1], queries=[query, query])
            qa(images=[image1, image2], queries=[query, query, query])


@is_pipeline_test
@require_vision
class VisualQuestionAnsweringPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", top_k=2)
        images = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        queries = "How many cats are there?"
        return vqa_pipeline, (images, queries)

    def run_pipeline_test(self, vqa_pipeline, examples):
        images, queries = examples
        outputs = vqa_pipeline(images, queries)
        self.assertEqual(
            outputs,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", top_k=2)
        # model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        # tokenizer = BertTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        # feature_extractor = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        # vqa_pipeline = VisualQuestionAnsweringPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        query = "How many cats are there?"
        expected_output_top_2 = [{'score': 0.9444, 'label': '2'}, {'label': '1', 'score': 0.0542}]

        outputs = vqa_pipeline(images=image, queries=query, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output_top_2)

        outputs = vqa_pipeline(images=[image], queries=query, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output_top_2)

        outputs = vqa_pipeline(images=image, queries=[query], top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output_top_2)

        outputs = vqa_pipeline(images=[image], queries=[query], top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output_top_2)

        outputs = vqa_pipeline(images=[image, image], queries=[query, query], top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), [expected_output_top_2] * 2)
    
    @require_tf
    @unittest.skip("Image segmentation not implemented in TF")
    def test_small_model_tf(self):
        pass

