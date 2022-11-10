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

from huggingface_hub import hf_hub_download
from transformers import (
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
    VideoMAEFeatureExtractor,
    is_decord_available,
    is_vision_available,
)
from transformers.pipelines import VideoClassificationPipeline, pipeline
from transformers.testing_utils import (
    nested_simplify,
    require_decord,
    require_tf,
    require_torch,
    require_torch_or_tf,
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


@require_torch_or_tf
@require_vision
@require_decord
class VideoClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        example_video_filepath = hf_hub_download(
            repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset"
        )
        video_classifier = VideoClassificationPipeline(model=model, feature_extractor=feature_extractor, top_k=2)
        examples = [
            example_video_filepath,
        ]
        return video_classifier, examples

    def run_pipeline_test(self, video_classifier, examples):

        outputs = video_classifier(examples[0])

        self.assertEqual(
            outputs,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        small_model = "hf-internal-testing/tiny-random-VideoMAEForVideoClassification"
        small_feature_extractor = VideoMAEFeatureExtractor(size=10, crop_size=dict(height=10, width=10))
        video_classifier = pipeline(
            "video-classification", model=small_model, feature_extractor=small_feature_extractor
        )

        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
        outputs = video_classifier(video_file_path, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [{"score": 0.521, "label": "LABEL_0"}, {"score": 0.479, "label": "LABEL_1"}],
        )

        outputs = video_classifier(
            [
                video_file_path,
                video_file_path,
            ],
            top_k=2,
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [
                [{"score": 0.521, "label": "LABEL_0"}, {"score": 0.479, "label": "LABEL_1"}],
                [{"score": 0.521, "label": "LABEL_0"}, {"score": 0.479, "label": "LABEL_1"}],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        pass
