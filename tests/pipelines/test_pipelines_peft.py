# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers.pipelines import pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    require_peft,
    require_torch,
    slow,
)
from transformers.utils import is_peft_available

from .test_pipelines_common import ANY


if is_peft_available():
    from peft import AutoPeftModelForCausalLM, PeftModel


@is_pipeline_test
@slow
@require_peft
@require_torch
class PeftIntegrationPipelineTests(unittest.TestCase):
    r"""
    Few tests to check if pipeline is compatible with PEFT models.
    """

    def test_pipeline_text_generation_with_kwargs(self):
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        peft_model_kwargs = {"adapter_name": "default"}

        pipe = pipeline("text-generation", model_id, peft_model_kwargs=peft_model_kwargs)
        output = pipe("Hello, my name is ")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {"generated_text": ANY(str)},
            ],
        )

    def test_pipeline_text_generation(self):
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        tok_id = "hf-internal-testing/tiny-random-OPTForCausalLM"

        pipe = pipeline("text-generation", model_id)
        output = pipe("Hello, my name is ")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {"generated_text": ANY(str)},
            ],
        )

        # with an external model
        peft_model = AutoPeftModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", peft_model, tokenizer=tok_id)
        output = pipe("Hello, my name is ")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {"generated_text": ANY(str)},
            ],
        )

    def test_pipeline_question_answering(self):
        model_id = "peft-internal-testing/tiny_OPTForQuestionAnswering-lora"

        pipe = pipeline("question-answering", model_id)
        output = pipe(
            question=["What field is HuggingFace working ?", "In what field is HuggingFace ?"],
            context=[
                "HuggingFace is a startup based in New-York",
                "HuggingFace is a startup founded in Paris",
            ],
        )

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
            ],
        )

    def test_pipeline_token_classification(self):
        model_id = "peft-internal-testing/tiny_GPT2ForTokenClassification-lora"

        pipe = pipeline("token-classification", model_id)
        output = pipe("Hello")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {
                    "entity": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "index": ANY(int),
                    "word": ANY(str),
                }
                for i in range(len(output))
            ],
        )

    def test_pipeline_text2text_generation(self):
        model_id = "peft-internal-testing/tiny_T5ForSeq2SeqLM-lora"

        pipe = pipeline("text2text-generation", model_id)
        output = pipe("Hello")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertEqual(
            output,
            [
                {"generated_text": ANY(str)},
            ],
        )

    def test_pipeline_feature_extraction(self):
        model_id = "peft-internal-testing/tiny_OPTForFeatureExtraction-lora"

        pipe = pipeline("feature-extraction", model_id)
        output = pipe("Hello")

        self.assertTrue(isinstance(pipe.model, PeftModel))
        self.assertTrue(
            isinstance(output, list)
            and isinstance(output[0], list)
            and isinstance(output[0][0], list)
            and isinstance(output[0][0][0], float)
        )
