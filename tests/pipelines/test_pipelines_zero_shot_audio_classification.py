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

from datasets import load_dataset

from transformers.pipelines import pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_torch, slow


@is_pipeline_test
@require_torch
class ZeroShotAudioClassificationPipelineTests(unittest.TestCase):
    # Deactivating auto tests since we don't have a good MODEL_FOR_XX mapping,
    # and only CLAP would be there for now.
    # model_mapping = {CLAPConfig: CLAPModel}

    @require_torch
    def test_small_model_pt(self, dtype="float32"):
        audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model="hf-internal-testing/tiny-clap-htsat-unfused",
            dtype=dtype,
        )
        dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        audio = dataset["train"]["audio"][-1]["array"]
        output = audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
        self.assertEqual(
            nested_simplify(output),
            [{"score": 0.501, "label": "Sound of a dog"}, {"score": 0.499, "label": "Sound of vaccum cleaner"}],
        )

    @require_torch
    def test_small_model_pt_fp16(self):
        self.test_small_model_pt(dtype="float16")

    @unittest.skip(reason="No models are available in TF")
    def test_small_model_tf(self):
        pass

    @slow
    @require_torch
    def test_large_model_pt(self):
        audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model="laion/clap-htsat-unfused",
        )
        # This is an audio of a dog
        dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        audio = dataset["train"]["audio"][-1]["array"]
        output = audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])

        self.assertEqual(
            nested_simplify(output),
            [
                {"score": 1.0, "label": "Sound of a dog"},
                {"score": 0.0, "label": "Sound of vaccum cleaner"},
            ],
        )

        output = audio_classifier([audio] * 5, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
        self.assertEqual(
            nested_simplify(output),
            [
                [
                    {"score": 1.0, "label": "Sound of a dog"},
                    {"score": 0.0, "label": "Sound of vaccum cleaner"},
                ],
            ]
            * 5,
        )
        output = audio_classifier(
            [audio] * 5, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"], batch_size=5
        )
        self.assertEqual(
            nested_simplify(output),
            [
                [
                    {"score": 1.0, "label": "Sound of a dog"},
                    {"score": 0.0, "label": "Sound of vaccum cleaner"},
                ],
            ]
            * 5,
        )

    @unittest.skip(reason="No models are available in TF")
    def test_large_model_tf(self):
        pass
