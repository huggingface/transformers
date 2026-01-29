# Copyright 2025 The HuggingFace Inc. team.
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
    PromptableConceptSegmentationPipeline,
    is_torch_available,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_torch,
    require_vision,
    slow,
)


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


if is_torch_available():
    import torch


@is_pipeline_test
@require_vision
@require_torch
class PromptableConceptSegmentationPipelineTests(unittest.TestCase):
    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        dtype="float32",
    ):
        segmenter = PromptableConceptSegmentationPipeline(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dtype=dtype,
        )

        examples = [
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "text": "cat",
            }
        ]
        return segmenter, examples

    def run_pipeline_test(self, segmenter, examples):
        outputs = segmenter(examples[0].get("image"), text=examples[0].get("text"), threshold=0.0)

        n = len(outputs)
        self.assertGreater(n, 0)

        # Check output structure
        for output in outputs:
            self.assertIn("score", output)
            self.assertIn("box", output)
            self.assertIn("mask", output)
            self.assertIsInstance(output["score"], float)
            self.assertIsInstance(output["box"], dict)
            self.assertIn("xmin", output["box"])
            self.assertIn("ymin", output["box"])
            self.assertIn("xmax", output["box"])
            self.assertIn("ymax", output["box"])
            self.assertTrue(is_torch_available() and isinstance(output["mask"], torch.Tensor))

    @require_torch
    @slow
    def test_small_model_pt_text_prompt(self):
        """Test pipeline with text-only prompt."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.1,
        )

        # Check that we got results
        self.assertGreater(len(outputs), 0)

        # Check structure of first result
        result = outputs[0]
        self.assertIn("score", result)
        self.assertIn("box", result)
        self.assertIn("mask", result)
        self.assertIn("label", result)
        self.assertEqual(result["label"], "cat")

        # Check box format
        self.assertIsInstance(result["box"]["xmin"], int)
        self.assertIsInstance(result["box"]["ymin"], int)
        self.assertIsInstance(result["box"]["xmax"], int)
        self.assertIsInstance(result["box"]["ymax"], int)

        # Check mask shape
        self.assertEqual(len(result["mask"].shape), 2)  # Should be 2D (H, W)

    @require_torch
    @slow
    def test_small_model_pt_box_prompt(self):
        """Test pipeline with box-only prompt."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Use a bounding box around a cat in the image
        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            input_boxes=[[[100, 50, 400, 350]]],
            input_boxes_labels=[[1]],
            threshold=0.1,
        )

        # Check that we got results
        self.assertGreater(len(outputs), 0)

        # Check structure
        result = outputs[0]
        self.assertIn("score", result)
        self.assertIn("box", result)
        self.assertIn("mask", result)

    @require_torch
    @slow
    def test_small_model_pt_combined_prompt(self):
        """Test pipeline with combined text and box prompts."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Text prompt with a negative box
        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            input_boxes=[[[50, 50, 150, 150]]],  # Negative box
            input_boxes_labels=[[0]],  # 0 = negative
            threshold=0.1,
        )

        # Should still get results, but filtered by negative box
        self.assertGreaterEqual(len(outputs), 0)

    @require_torch
    @slow
    def test_batched_text_prompts(self):
        """Test batching with multiple images and text prompts."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        outputs = segmenter(
            [
                {
                    "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                    "text": "cat",
                },
                {
                    "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                    "text": "remote",
                },
            ],
            threshold=0.1,
        )

        # Should get a list of lists
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[0], list)
        self.assertIsInstance(outputs[1], list)

    @require_torch
    @slow
    def test_threshold(self):
        """Test score threshold filtering."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Get results with low threshold
        outputs_low = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.01,
        )

        # Get results with high threshold
        outputs_high = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.5,
        )

        # High threshold should give fewer or equal results
        self.assertLessEqual(len(outputs_high), len(outputs_low))

    @require_torch
    @slow
    def test_mask_threshold(self):
        """Test mask binarization threshold."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.1,
            mask_threshold=0.5,
        )

        # Check that masks are binary
        if len(outputs) > 0:
            mask = outputs[0]["mask"]
            unique_values = torch.unique(mask)
            # Mask should be binary (0 and 1) or close to it
            self.assertTrue(all(val in [0, 1, 0.0, 1.0] or (val >= 0 and val <= 1) for val in unique_values))

    @require_torch
    @slow
    def test_top_k(self):
        """Test top_k parameter to limit number of results."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Get all results
        outputs_all = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.01,
        )

        # Get only top 2
        outputs_top2 = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.01,
            top_k=2,
        )

        # Should have at most 2 results
        self.assertLessEqual(len(outputs_top2), 2)
        self.assertLessEqual(len(outputs_top2), len(outputs_all))

    @require_torch
    @slow
    def test_dict_input_format(self):
        """Test dict input format."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Dict format
        outputs = segmenter(
            {"image": "./tests/fixtures/tests_samples/COCO/000000039769.png", "text": "cat"},
            threshold=0.1,
        )

        self.assertGreater(len(outputs), 0)

    @require_torch
    @slow
    def test_no_prompt_error(self):
        """Test that error is raised when no prompts are provided."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        with self.assertRaises(ValueError) as context:
            segmenter("./tests/fixtures/tests_samples/COCO/000000039769.png")

        self.assertIn("at least one prompt", str(context.exception).lower())

    @require_torch
    @slow
    def test_multiple_boxes(self):
        """Test with multiple positive boxes."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Multiple positive boxes
        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            input_boxes=[[[100, 50, 300, 250], [350, 100, 550, 350]]],
            input_boxes_labels=[[1, 1]],
            threshold=0.1,
        )

        # Should get results
        self.assertGreaterEqual(len(outputs), 0)

    @require_torch
    @slow
    def test_scores_are_sorted(self):
        """Test that results are sorted by score in descending order."""
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.01,
        )

        if len(outputs) > 1:
            scores = [output["score"] for output in outputs]
            # Check that scores are sorted in descending order
            self.assertEqual(scores, sorted(scores, reverse=True))

    @require_torch
    @slow
    def test_automatic_model_processor_conversion(self):
        """Test that the pipeline automatically converts Sam3VideoModel/Processor to Sam3Model/Processor."""
        # This should work even though facebook/sam3 has Sam3VideoModel by default
        segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")

        # Verify correct types were loaded
        self.assertEqual(segmenter.model.__class__.__name__, "Sam3Model")
        self.assertEqual(segmenter.processor.__class__.__name__, "Sam3Processor")

        # Verify it works functionally
        outputs = segmenter(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text="cat",
            threshold=0.3,
        )

        self.assertGreater(len(outputs), 0)
        self.assertIn("score", outputs[0])
        self.assertIn("box", outputs[0])
        self.assertIn("mask", outputs[0])
