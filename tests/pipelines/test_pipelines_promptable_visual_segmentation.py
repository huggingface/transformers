# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from transformers import (
    MODEL_FOR_PROMPTABLE_VISUAL_SEGMENTATION_MAPPING,
    PromptableVisualSegmentationPipeline,
    Sam2Model,
    Sam2Processor,
    SamModel,
    SamProcessor,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, require_torch, require_vision, slow


if is_vision_available():
    import requests
    from PIL import Image


@is_pipeline_test
@require_torch
@require_vision
class PromptableVisualSegmentationPipelineTests(unittest.TestCase):
    model_mapping = (
        dict(list(MODEL_FOR_PROMPTABLE_VISUAL_SEGMENTATION_MAPPING.items()))
        if MODEL_FOR_PROMPTABLE_VISUAL_SEGMENTATION_MAPPING
        else []
    )

    # Test image URLs
    test_image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        dtype="float32",
    ):
        segmenter = PromptableVisualSegmentationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            dtype=dtype,
        )
        examples = [
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "input_points": [[[[450, 600]]]],
                "input_labels": [[[1]]],
            },
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "input_boxes": [[[100, 200, 350, 550]]],
            },
        ]
        return segmenter, examples

    def run_pipeline_test(self, segmenter, examples):
        for example in examples:
            result = segmenter(**example)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            # Each result should be a list of objects (for multiple images)
            for obj_list in result:
                self.assertIsInstance(obj_list, list)
                for obj in obj_list:
                    self.assertIn("mask", obj)
                    self.assertIn("score", obj)

    def get_test_image(self):
        """Helper to load test image."""
        return Image.open(requests.get(self.test_image_url, stream=True).raw).convert("RGB")

    def test_sam2_single_point(self):
        """Test SAM2 with single point prompt."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]  # Single point
        input_labels = [[[1]]]  # Positive

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1, "Should return results for 1 image")
        self.assertGreater(len(results[0]), 0, "Should return at least 1 mask")
        self.assertIn("score", results[0][0])
        self.assertIn("mask", results[0][0])
        self.assertIsInstance(results[0][0]["score"], float)
        self.assertTrue(0 <= results[0][0]["score"] <= 1, "Score should be between 0 and 1")

    def test_sam2_box_prompt(self):
        """Test SAM2 with box prompt."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_boxes = [[[75, 275, 1725, 850]]]  # Box around truck

        results = segmenter(image, input_boxes=input_boxes, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)
        self.assertIn("score", results[0][0])
        self.assertIn("mask", results[0][0])

    def test_sam2_multiple_points(self):
        """Test SAM2 with multiple points per object."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points
        input_labels = [[[1, 1]]]  # Both positive

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    def test_sam2_multiple_objects(self):
        """Test SAM2 with multiple objects in same image."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        # Points for two different objects
        input_points = [[[[500, 375]], [[650, 750]]]]
        input_labels = [[[1], [1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(len(results[0]), 2, "Should return at least 2 masks for 2 objects")

    def test_sam2_multimask_output(self):
        """Test SAM2 with multimask_output=True."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=True)

        self.assertEqual(len(results), 1)
        # With multimask_output=True, should return 3 masks per object
        self.assertGreaterEqual(len(results[0]), 3, "Should return at least 3 masks with multimask_output=True")

    def test_sam2_mask_threshold(self):
        """Test SAM2 with mask_threshold parameter."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(
            image, input_points=input_points, input_labels=input_labels, mask_threshold=0.5, multimask_output=False
        )

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    def test_sam2_top_k(self):
        """Test SAM2 with top_k parameter."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(
            image, input_points=input_points, input_labels=input_labels, multimask_output=True, top_k=2
        )

        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]), 2, "Should return at most 2 masks with top_k=2")

    def test_sam_single_point(self):
        """Test SAM with single point prompt."""
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)
        self.assertIn("score", results[0][0])
        self.assertIn("mask", results[0][0])

    def test_results_sorted_by_score(self):
        """Test that results are sorted by score in descending order."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=True)

        scores = [r["score"] for r in results[0]]
        sorted_scores = sorted(scores, reverse=True)
        self.assertEqual(scores, sorted_scores, "Results should be sorted by score in descending order")

    def test_error_no_prompts(self):
        """Test that error is raised when no prompts are provided."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()

        with self.assertRaises(ValueError) as context:
            segmenter(image)

        self.assertIn("at least one prompt type", str(context.exception))

    def test_error_points_without_labels(self):
        """Test that error is raised when points are provided without labels."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]

        with self.assertRaises(ValueError) as context:
            segmenter(image, input_points=input_points)

        self.assertIn("input_labels", str(context.exception))

    @slow
    def test_sam2_automatic_loading(self):
        """Test that SAM2 can be loaded automatically with checkpoint name."""
        segmenter = pipeline("promptable-visual-segmentation", model="facebook/sam2.1-hiera-large")

        self.assertIsInstance(segmenter.model, Sam2Model)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    @slow
    def test_sam_automatic_loading(self):
        """Test that SAM can be loaded automatically with checkpoint name."""
        segmenter = pipeline("promptable-visual-segmentation", model="facebook/sam-vit-base")

        self.assertIsInstance(segmenter.model, SamModel)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    def test_mask_shape(self):
        """Test that mask shape matches original image size."""
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")

        segmenter = pipeline("promptable-visual-segmentation", model=model, processor=processor)

        image = self.get_test_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)

        mask = results[0][0]["mask"]
        expected_shape = (image.height, image.width)
        self.assertEqual(
            mask.shape, expected_shape, f"Mask shape {mask.shape} should match image size {expected_shape}"
        )
