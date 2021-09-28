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

import numpy as np

from transformers import MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
from transformers.pipelines import AudioClassificationPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_datasets,
    require_tf,
    require_torch,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
@require_torch
class AudioClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING

    @require_datasets
    @slow
    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        import datasets

        audio_classifier = AudioClassificationPipeline(model=model, feature_extractor=feature_extractor)

        # test with a raw waveform
        audio = np.zeros((34000,))
        output = audio_classifier(audio)
        # by default a model is initialized with num_labels=2
        self.assertEqual(
            output,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )
        output = audio_classifier(audio, top_k=1)
        self.assertEqual(
            output,
            [
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

        # test with a local file
        dataset = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        filename = dataset[0]["file"]
        output = audio_classifier(filename)
        self.assertEqual(
            output,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        model = "anton-l/wav2vec2-random-tiny-classifier"

        audio_classifier = pipeline("audio-classification", model=model)

        audio = np.ones((8000,))
        output = audio_classifier(audio, top_k=4)
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [
                {"score": 0.0843, "label": "on"},
                {"score": 0.0840, "label": "left"},
                {"score": 0.0837, "label": "off"},
                {"score": 0.0835, "label": "yes"},
            ],
        )

    @require_torch
    @require_datasets
    @slow
    def test_large_model_pt(self):
        import datasets

        model = "superb/wav2vec2-base-superb-ks"

        audio_classifier = pipeline("audio-classification", model=model)
        dataset = datasets.load_dataset("anton-l/superb_dummy", "ks", split="test")

        audio = np.array(dataset[3]["speech"], dtype=np.float32)
        output = audio_classifier(audio, top_k=4)
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [
                {"score": 0.9809, "label": "go"},
                {"score": 0.0073, "label": "up"},
                {"score": 0.0064, "label": "_unknown_"},
                {"score": 0.0015, "label": "down"},
            ],
        )

    @require_tf
    @unittest.skip("Audio classification is not implemented for TF")
    def test_small_model_tf(self):
        pass
