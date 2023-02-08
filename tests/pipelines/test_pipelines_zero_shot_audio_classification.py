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

from datasets import load_dataset

from transformers.pipelines import pipeline
from transformers.testing_utils import require_torch

from .test_pipelines_common import PipelineTestCaseMeta


@require_torch
class ZeroShotAudioClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    # Deactivating auto tests since we don't have a good MODEL_FOR_XX mapping,
    # and only CLAP would be there for now.
    # model_mapping = {CLAPConfig: CLAPModel}

    # def get_test_pipeline(self, model, tokenizer, processor):
    #     if tokenizer is None:
    #         # Side effect of no Fast Tokenizer class for these model, so skipping
    #         # But the slow tokenizer test should still run as they're quite small
    #         self.skipTest("No tokenizer available")
    #         return
    #         # return None, None

    #     audio_classifier = ZeroShotAudioClassificationPipeline(
    #         model=model, tokenizer=tokenizer, feature_extractor=processor
    #     )

    #     # test with a raw waveform
    #     audio = Audio.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     audio2 = Audio.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     return audio_classifier, [audio, audio2]

    # def run_pipeline_test(self, pipe, examples):
    #     audio = Audio.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     outputs = pipe(audio, candidate_labels=["A", "B"])
    #     self.assertEqual(outputs, {"text": ANY(str)})

    #     # Batching
    #     outputs = pipe([audio] * 3, batch_size=2, candidate_labels=["A", "B"])
    @require_torch
    def test_small_model_pt(self):
        pass

    # @require_torch
    # def test_small_model_pt(self):
    #     audio_classifier = pipeline(
    #         model="hf-internal-testing/tiny-random-clap-zero-shot-audio-classification",
    #     )
    #     audio = Audio.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     output = audio_classifier(audio, candidate_labels=["a", "b", "c"])

    #     self.assertEqual(
    #         nested_simplify(output),
    #         [{"score": 0.333, "label": "a"}, {"score": 0.333, "label": "b"}, {"score": 0.333, "label": "c"}],
    #     )

    #     output = audio_classifier([audio] * 5, candidate_labels=["A", "B", "C"], batch_size=2)
    #     self.assertEqual(
    #         nested_simplify(output),
    #         # Pipeline outputs are supposed to be deterministic and
    #         # So we could in theory have real values "A", "B", "C" instead
    #         # of ANY(str).
    #         # However it seems that in this particular case, the floating
    #         # scores are so close, we enter floating error approximation
    #         # and the order is not guaranteed anymore with batching.
    #         [
    #             [
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #             ],
    #             [
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #             ],
    #             [
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #             ],
    #             [
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #             ],
    #             [
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #                 {"score": 0.333, "label": ANY(str)},
    #             ],
    #         ],
    #     )

    # @slow
    @require_torch
    def test_large_model_pt(self):
        audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model="ybelkada/clap-htsat-unfused",
        )
        # This is an audio of 2 cats with remotes and no planes
        dataset = load_dataset("ashraq/esc50")
        audio = dataset["train"]["audio"][-1]["array"]
        output = audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])

        self.assertEqual(
            output,
            [
                {"score": 0.9990969896316528, "label": "Sound of a dog"},
                {"score": 0.0009030875517055392, "label": "Sound of vaccum cleaner"},
            ],
        )

        output = audio_classifier([audio] * 5, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
        self.assertEqual(
            output,
            [
                [
                    {"score": 0.9990969896316528, "label": "Sound of a dog"},
                    {"score": 0.0009030875517055392, "label": "Sound of vaccum cleaner"},
                ]
            ]
            * 5,
        )
        # TODO batching will be supported in next PR, the base pipeline needs to be modified
        # output = audio_classifier([audio] * 5, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"], batch_size=5)
