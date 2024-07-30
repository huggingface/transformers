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

from transformers import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING, VideoMAEFeatureExtractor
from transformers.pipelines import VideoClassificationPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_av,
    require_tf,
    require_torch,
    require_torch_or_tf,
    require_vision,
)

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch_or_tf
@require_vision
@require_av
class VideoClassificationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor, torch_dtype="float32"):
        example_video_filepath = hf_hub_download(
            repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset"
        )
        video_classifier = VideoClassificationPipeline(
            model=model, image_processor=processor, top_k=2, torch_dtype=torch_dtype
        )
        examples = [
            example_video_filepath,
            "https://huggingface.co/datasets/nateraw/video-demo/resolve/main/archery.mp4",
        ]
        return video_classifier, examples

    def run_pipeline_test(self, video_classifier, examples):
        for example in examples:
            outputs = video_classifier(example)

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
        small_feature_extractor = VideoMAEFeatureExtractor(
            size={"shortest_edge": 10}, crop_size={"height": 10, "width": 10}
        )
        video_classifier = pipeline(
            "video-classification", model=small_model, feature_extractor=small_feature_extractor, frame_sampling_rate=4
        )

        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
        outputs = video_classifier(video_file_path, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [{"score": 0.5199, "label": "LABEL_0"}, {"score": 0.4801, "label": "LABEL_1"}],
        )

        outputs = video_classifier(
            [
                video_file_path,
                video_file_path,
            ],
            top_k=2,
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [{"score": 0.5199, "label": "LABEL_0"}, {"score": 0.4801, "label": "LABEL_1"}],
                [{"score": 0.5199, "label": "LABEL_0"}, {"score": 0.4801, "label": "LABEL_1"}],
            ],
        )

    @require_tf
    @unittest.skip
    def test_small_model_tf(self):
        pass
