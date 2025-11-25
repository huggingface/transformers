# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, VideoMAEImageProcessor
from transformers.pipelines import VideoToTextPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_av,
    require_torch,
    require_vision,
)

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch
@require_vision
@require_av
class VideoToTextPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    example_video_filepath = None

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls.example_video_filepath is None:
            cls.example_video_filepath = hf_hub_download(
                repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset"
            )

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        dtype="float32",
    ):
        self._load_dataset()
        video_to_text = VideoToTextPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            dtype=dtype,
            max_new_tokens=20,
        )
        examples = [
            self.example_video_filepath,
            # TODO: re-enable this once we have a stable hub solution for CI
            # "https://huggingface.co/datasets/nateraw/video-demo/resolve/main/archery.mp4",
        ]
        return video_to_text, examples

    def run_pipeline_test(self, video_to_text, examples):
        for example in examples:
            outputs = video_to_text(example)

            self.assertEqual(
                outputs,
                [
                    {"generated_text": ANY(str)},
                ],
            )

    @require_torch
    def test_small_model_pt(self):
        small_model = "hf-internal-testing/tiny-random-vit-gpt2"
        small_image_processor = VideoMAEImageProcessor(
            size={"shortest_edge": 10}, crop_size={"height": 10, "width": 10}
        )
        video_to_text = pipeline(
            "video-to-text", model=small_model, image_processor=small_image_processor, frame_sampling_rate=4, max_new_tokens=19
        )

        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
        output = video_to_text(video_file_path)
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [
                {
                    "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                },
            ],
        )

        outputs = video_to_text(
            [
                video_file_path,
                video_file_path,
            ],
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {
                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                    }
                ],
                [
                    {
                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                    }
                ],
            ],
        )

