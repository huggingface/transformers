# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_CAUSAL_LM_MAPPING, TextGenerationPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, require_tf, require_torch

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
class TextGenerationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING

    @require_torch
    def test_small_model_pt(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="pt")
        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope. oscope. FiliFili@@"
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope. oscope. FiliFili@@"
                    }
                ],
                [
                    {
                        "generated_text": "This is a second test ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope. oscope. FiliFili@@"
                    }
                ],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="tf")

        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵 please,"
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"], do_sample=False)
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵 please,"
                    }
                ],
                [
                    {
                        "generated_text": "This is a second test Chieftain Chieftain prefecture prefecture prefecture Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵 please,"
                    }
                ],
            ],
        )

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        return text_generator, ["This is a test", "Another test"]

    def run_pipeline_test(self, text_generator, _):
        model = text_generator.model
        tokenizer = text_generator.tokenizer

        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator("This is a test", return_full_text=False)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        outputs = text_generator("This is a test", return_full_text=True)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        # Empty prompt is slighly special
        # it requires BOS token to exist.
        # Special case for Pegasus which will always append EOS so will
        # work even without BOS.
        if text_generator.tokenizer.bos_token_id is not None or "Pegasus" in tokenizer.__class__.__name__:
            outputs = text_generator("")
            self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        else:
            with self.assertRaises((ValueError, AssertionError)):
                outputs = text_generator("")

        if text_generator.framework == "tf":
            # TF generation does not support max_new_tokens, and it's impossible
            # to control long generation with only max_length without
            # fancy calculation, dismissing tests for now.
            return
        # We don't care about infinite range models.
        # They already work.
        if tokenizer.model_max_length < 10000:
            # Handling of large generations
            with self.assertRaises((RuntimeError, IndexError, ValueError, AssertionError)):
                text_generator("This is a test" * 500, max_new_tokens=20)

            outputs = text_generator("This is a test" * 500, handle_long_generation="hole", max_new_tokens=20)
            # Hole strategy cannot work
            with self.assertRaises(ValueError):
                text_generator(
                    "This is a test" * 500,
                    handle_long_generation="hole",
                    max_new_tokens=tokenizer.model_max_length + 10,
                )
