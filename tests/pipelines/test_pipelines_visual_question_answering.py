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
        question = "How many cats are there?"

        for image in [image1, image2]:
            output = qa(image=image, question=question)
            self.assert_helper(output, expected_output_size=1)

        output = qa(image=[image1, image2], question=[question, question])
        self.assert_helper(output, expected_output_size=2)
        
        output = qa(image=image1, question=[question, question])
        self.assert_helper(output, expected_output_size=2)
        
        output = qa(image=[image1, image2], question=question)
        self.assert_helper(output, expected_output_size=2)
        
        output = qa([{"image": image1, "question": question}, {"image": image2, "question": question}])
        self.assert_helper(output, expected_output_size=2)
        
        output = qa({"image": image1, "question": question})
        self.assert_helper(output, expected_output_size=1)
        
    
    def assert_helper(self, output, expected_output_size):
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), expected_output_size)
        for item in output:
            self.assertEqual(type(item), dict)
            self.assertIn("image", item)
            self.assertIn("question", item)

    def test_argument_handler_error_handling(self):
        qa = VisualQuestionAnsweringArgumentHandler()
        image1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        question = "How many cats are there?"

        with self.assertRaises(ValueError):
            qa(image=[image1, image2], question=[question, question, question])
            qa(image=[image1], question=[question, question])
            qa(image=[image1], question=[question, question])
            qa({"random_key1": image1, "random_key2": question})


@is_pipeline_test
@require_vision
class VisualQuestionAnsweringPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        vqa_pipeline = pipeline("visual-question-answering", model="sijunhe/tiny-vilt-random-vqa")
        examples = [
            {
                "image": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "question": "How many cats are there?"
            },
            {
                "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                "question": "How many cats are there?"
            }
        ]
        return vqa_pipeline, examples

    def run_pipeline_test(self, vqa_pipeline, examples):
        outputs = vqa_pipeline(examples, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "label": ANY(str)}],
                [{"score": ANY(float), "label": ANY(str)}],
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="sijunhe/tiny-vilt-random-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        query = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=query, top_k=2)
        self.assertEqual(outputs, [[{"score": ANY(float), "label": ANY(str)}, {"score": ANY(float), "label": ANY(str)}]])

        outputs = vqa_pipeline(image=[image, image], question=[query, query], top_k=2)
        self.assertEqual(outputs, [[{"score": ANY(float), "label": ANY(str)}, {"score": ANY(float), "label": ANY(str)}]] * 2)
    
    @slow
    @require_torch
    def test_large_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        query = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=query, top_k=2)
        self.assertEqual(outputs, [[{"score": ANY(float), "label": ANY(str)}, {"score": ANY(float), "label": ANY(str)}]])

        outputs = vqa_pipeline(image=[image, image], question=[query, query], top_k=2)
        self.assertEqual(outputs, [[{"score": ANY(float), "label": ANY(str)}, {"score": ANY(float), "label": ANY(str)}]] * 2)

    @require_tf
    @unittest.skip("Visual question answering not implemented in TF")
    def test_small_model_tf(self):
        pass

