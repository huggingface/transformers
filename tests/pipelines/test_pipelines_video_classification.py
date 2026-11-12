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

from huggingface_hub import VideoClassificationOutputElement, hf_hub_download

from transformers import (
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
    AutoModelForVideoClassification,
    VideoMAEImageProcessor,
    is_torch_available,
)
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


if is_torch_available():
    import torch


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
            # "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/archery.mp4",
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
    def test_problem_type_defaults_and_overrides(self):
        small_model = AutoModelForVideoClassification.from_pretrained(
            "hf-internal-testing/tiny-random-VideoMAEForVideoClassification"
        )
        small_image_processor = VideoMAEImageProcessor(
            size={"shortest_edge": 10}, crop_size={"height": 10, "width": 10}
        )
        classifier = pipeline(
            "video-classification",
            model=small_model,
            image_processor=small_image_processor,
            frame_sampling_rate=4,
        )
        video_file_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
        num_labels = classifier.model.config.num_labels
        labels = [classifier.model.config.id2label[i] for i in range(num_labels)]

        raw = classifier(video_file_path, top_k=num_labels, function_to_apply="none")
        raw_scores = {item["label"]: item["score"] for item in raw}
        logits = torch.tensor([raw_scores[label] for label in labels])

        for problem_type, activation in (
            ("regression", "none"),
            ("multi_label_classification", "sigmoid"),
            ("single_label_classification", "softmax"),
            (None, "softmax"),
        ):
            with self.subTest(problem_type=problem_type):
                classifier.model.config.problem_type = problem_type
                result = classifier(video_file_path, top_k=num_labels)
                scores = {item["label"]: item["score"] for item in result}
                expected = {
                    "none": logits,
                    "sigmoid": logits.sigmoid(),
                    "softmax": logits.softmax(-1),
                }[activation]
                self.assertEqual(len(scores), num_labels)
                for label, value in zip(labels, expected.tolist()):
                    self.assertAlmostEqual(scores[label], value, places=4)

        classifier.model.config.problem_type = "regression"
        initialized = pipeline(
            "video-classification",
            model=classifier.model,
            image_processor=small_image_processor,
            frame_sampling_rate=4,
            function_to_apply="softmax",
        )

        for override, expected in (
            (None, logits.softmax(-1)),
            ("none", logits),
            ("sigmoid", logits.sigmoid()),
        ):
            with self.subTest(call_override=override):
                result = initialized(video_file_path, top_k=num_labels, function_to_apply=override)
                scores = {item["label"]: item["score"] for item in result}
                for label, value in zip(labels, expected.tolist()):
                    self.assertAlmostEqual(scores[label], value, places=4)
