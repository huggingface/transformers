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
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from huggingface_hub import VideoClassificationOutputElement, hf_hub_download

from transformers import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING, VideoMAEImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.pipelines import VideoClassificationPipeline, pipeline
from transformers.testing_utils import (
    compare_pipeline_output_to_hub_spec,
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
class VideoClassificationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING
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
        video_classifier = VideoClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            dtype=dtype,
            top_k=2,
        )
        examples = [
            self.example_video_filepath,
            # TODO: re-enable this once we have a stable hub solution for CI
            # "https://huggingface.co/datasets/nateraw/video-demo/resolve/main/archery.mp4",
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
            for element in outputs:
                compare_pipeline_output_to_hub_spec(element, VideoClassificationOutputElement)

    @require_torch
    def test_small_model_pt(self):
        small_model = "hf-internal-testing/tiny-random-VideoMAEForVideoClassification"
        small_feature_extractor = VideoMAEImageProcessor(
            size={"shortest_edge": 10}, crop_size={"height": 10, "width": 10}
        )
        video_classifier = pipeline(
            "video-classification", model=small_model, feature_extractor=small_feature_extractor, frame_sampling_rate=4
        )

        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
        output = video_classifier(video_file_path, top_k=2)
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [{"score": 0.5199, "label": "LABEL_0"}, {"score": 0.4801, "label": "LABEL_1"}],
        )
        for element in output:
            compare_pipeline_output_to_hub_spec(element, VideoClassificationOutputElement)

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
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VideoClassificationOutputElement)

    @require_torch
    def test_video_processor_path(self):
        """VideoClassificationPipeline uses video_processor when present, falling back to image_processor."""
        small_model = "hf-internal-testing/tiny-random-VideoMAEForVideoClassification"
        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")

        video_classifier = pipeline(
            "video-classification",
            model=small_model,
            feature_extractor=VideoMAEImageProcessor(
                size={"shortest_edge": 10}, crop_size={"height": 10, "width": 10}
            ),
            frame_sampling_rate=4,
        )

        # When no video_processor is set, image_processor or feature_extractor should be used
        self.assertIsNone(video_classifier.video_processor)
        self.assertIsNotNone(video_classifier.image_processor or video_classifier.feature_extractor)

        # Swap in a fake video_processor and verify it takes priority
        fake_video_processor = MagicMock()
        fake_video_processor.return_value = BatchFeature({"pixel_values": torch.zeros(1, 16, 3, 10, 10)})
        video_classifier.video_processor = fake_video_processor
        video_classifier.image_processor = None

        fake_model_output = SequenceClassifierOutput(logits=torch.tensor([[0.6, 0.4]]))
        frames = np.stack([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(8)])
        with (
            patch("transformers.pipelines.video_classification.read_video_pyav", return_value=frames),
            patch.object(video_classifier, "_forward", return_value=fake_model_output),
        ):
            output = video_classifier(video_file_path, top_k=2)

        fake_video_processor.assert_called_once()
        self.assertEqual(len(output), 2)
        self.assertIn("score", output[0])
        self.assertIn("label", output[0])
