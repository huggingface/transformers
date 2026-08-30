# Copyright 2026 HuggingFace Inc.
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
from types import SimpleNamespace

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


class PPDocLayoutV4ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    ):
        size = size if size is not None else {"height": 40, "width": 40}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


# `[center_x, center_y, dx1, dy1, ..., dx4, dy4]`, offsets are shifted by +0.5
_DUMMY_QUAD = [0.5, 0.5, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6]


def _dummy_logits(scores_per_query, num_labels=25):
    """
    Builds class logits that make the flattened `query x class` top-k select one distinct query per entry.

    `post_process_object_detection` runs `topk` over `num_queries * num_labels` scores and keeps `num_queries` of
    them, so a query is only represented in the output if one of its class logits makes it into that top-k. Giving
    every query exactly one dominant class, with all the others far below, is what keeps the selected queries
    distinct -- uniform logits would return the same query under `num_queries` different labels instead.
    """
    num_queries = len(scores_per_query)
    logits = torch.full((1, num_queries, num_labels), -10.0)
    for query, score in enumerate(scores_per_query):
        logits[:, query, query % num_labels] = score
    return logits


def _dummy_outputs(num_queries=8, num_labels=25):
    """
    Builds model-shaped outputs whose reading order is the single chain `0 -> 1 -> ... -> num_queries - 1`.

    The confidences run the other way, so the top-k returns the queries in reverse reading order and the decode has
    to permute them back. A post-processor that ignored the order heads would come out backwards.
    """
    successor = torch.full((1, num_queries, num_queries), -10.0)
    for i in range(num_queries - 1):
        successor[:, i, i + 1] = 5.0
    relative = torch.triu(torch.full((num_queries, num_queries), 5.0), diagonal=1)
    relative = (relative - relative.T).unsqueeze(0).contiguous()
    quad = torch.tensor(_DUMMY_QUAD)
    return SimpleNamespace(
        logits=_dummy_logits([1.0 + query for query in range(num_queries)], num_labels=num_labels),
        pred_boxes=quad.expand(1, num_queries, 10).contiguous(),
        relative_order_logits=relative,
        successor_order_logits=successor,
    )


@require_torch
@require_vision
class PPDocLayoutV4ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = PPDocLayoutV4ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(
        reason="PPDocLayoutV4 uses antialias=False which is not supported for 4-channel images consistently"
    )
    def test_call_numpy_4_channels(self):
        pass

    def test_post_process_quad_and_boxes(self):
        outputs = _dummy_outputs()
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])[
                0
            ]

            self.assertEqual(result["polygon_points"].shape[-2:], torch.Size((4, 2)))
            # The quad is a centered rectangle spanning 20% of each side, scaled by (width, height).
            torch.testing.assert_close(
                result["polygon_points"][0],
                torch.tensor([[40.0, 80.0], [60.0, 80.0], [60.0, 120.0], [40.0, 120.0]]),
            )
            # `boxes` is the axis aligned rect enclosing the quad, in (x1, y1, x2, y2) order.
            torch.testing.assert_close(result["boxes"][0], torch.tensor([40.0, 80.0, 60.0, 120.0]))

    def test_post_process_num_coords_4_falls_back_to_rect(self):
        outputs = _dummy_outputs()
        # `config.num_coords=4` predicts plain `(cx, cy, w, h)`; the same centered rect as the 10-coord quad.
        outputs.pred_boxes = torch.tensor([0.5, 0.5, 0.2, 0.2]).expand(1, 8, 4).contiguous()
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])[
                0
            ]

            torch.testing.assert_close(
                result["polygon_points"][0],
                torch.tensor([[40.0, 80.0], [60.0, 80.0], [60.0, 120.0], [40.0, 120.0]]),
            )
            torch.testing.assert_close(result["boxes"][0], torch.tensor([40.0, 80.0, 60.0, 120.0]))

    def test_post_process_rejects_unsupported_num_coords(self):
        outputs = _dummy_outputs()
        outputs.pred_boxes = torch.rand(1, 8, 6)
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            with self.assertRaisesRegex(ValueError, "Unsupported num_coords: 6"):
                image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])

    def test_post_process_reading_order_is_sorted(self):
        num_queries = 8
        outputs = _dummy_outputs(num_queries=num_queries)
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])[
                0
            ]

            self.assertEqual(len(result["order_seq"]), num_queries)
            self.assertEqual(result["order_seq"].tolist(), list(range(num_queries)))
            # Query `q` carries label `q` and is read at rank `q`, so the labels come back in ascending order even
            # though the top-k handed them over sorted by descending confidence.
            self.assertEqual(result["labels"].tolist(), list(range(num_queries)))
            scores = result["scores"]
            self.assertTrue(bool((scores[1:] > scores[:-1]).all()))

    def test_post_process_reading_order_breaks_cycles(self):
        """A successor graph with a cycle still has to decode into a total order, dropping the weakest edge."""
        num_queries = 4
        successor = torch.full((1, num_queries, num_queries), -10.0)
        # `0 -> 1 -> 2 -> 0` is a cycle whose weakest edge is the one closing it, plus an isolated query 3.
        successor[:, 0, 1] = 5.0
        successor[:, 1, 2] = 4.0
        successor[:, 2, 0] = 0.5
        relative = torch.triu(torch.full((num_queries, num_queries), 5.0), diagonal=1)
        relative = (relative - relative.T).unsqueeze(0).contiguous()
        outputs = SimpleNamespace(
            logits=_dummy_logits([4.0, 3.0, 2.0, 1.0]),
            pred_boxes=torch.tensor(_DUMMY_QUAD).expand(1, num_queries, 10).contiguous(),
            relative_order_logits=relative,
            successor_order_logits=successor,
        )

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])[
                0
            ]

            self.assertEqual(result["order_seq"].tolist(), list(range(num_queries)))
            # Dropping `2 -> 0` leaves the chain `0 -> 1 -> 2`, and the relative order head puts the isolated
            # query 3 last.
            self.assertEqual(result["labels"].tolist(), [0, 1, 2, 3])

    def test_post_process_repeated_query_keeps_single_rank(self):
        """The top-k is over `query x class`, so one query can be kept under several labels with one shared rank."""
        num_queries = 4
        logits = torch.full((1, num_queries, 25), -10.0)
        # Query 1 is confident under two labels, so it takes two of the four top-k slots and query 3 takes none.
        logits[:, 0, 0], logits[:, 1, 1], logits[:, 1, 2], logits[:, 2, 3] = 6.0, 5.0, 4.9, 3.0
        successor = torch.full((1, num_queries, num_queries), -10.0)
        successor[:, 0, 1] = 5.0
        successor[:, 1, 2] = 5.0
        relative = torch.triu(torch.full((num_queries, num_queries), 5.0), diagonal=1)
        relative = (relative - relative.T).unsqueeze(0).contiguous()
        outputs = SimpleNamespace(
            logits=logits,
            pred_boxes=torch.tensor(_DUMMY_QUAD).expand(1, num_queries, 10).contiguous(),
            relative_order_logits=relative,
            successor_order_logits=successor,
        )

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(200, 100)])[
                0
            ]

            self.assertEqual(len(result["scores"]), num_queries)
            # Ranks are per query, not per kept detection, so query 1's two labels share rank 1.
            self.assertEqual(result["order_seq"].tolist(), [0, 1, 1, 2])
            self.assertEqual(result["labels"].tolist(), [0, 1, 2, 3])

    def test_post_process_without_detections(self):
        outputs = _dummy_outputs(num_queries=8)
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.post_process_object_detection(outputs, threshold=1.0, target_sizes=[(200, 100)])[
                0
            ]

            self.assertEqual(result["scores"].shape, torch.Size((0,)))
            self.assertEqual(result["boxes"].shape, torch.Size((0, 4)))
            self.assertEqual(result["polygon_points"].shape, torch.Size((0, 4, 2)))
            self.assertEqual(result["order_seq"].shape, torch.Size((0,)))

    def test_resize_runs_in_float_and_clips_overshoot(self):
        """
        The reference preprocessing resizes with `cv2.resize`, which rounds to `uint8` exactly once and saturates.
        Rescaling before the resize and clipping the bicubic overshoot keeps every pixel within one 8-bit step of
        that reference; resizing in `uint8` instead drifts far enough to permute the predicted reading order.
        """
        # A one pixel wide white bar on black maximizes bicubic ringing.
        image = torch.zeros(3, 64, 64, dtype=torch.uint8)
        image[:, :, 30:34] = 255

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]

            self.assertEqual(pixel_values.dtype, torch.float32)
            # Without the clip the bicubic undershoot/overshoot leaves this range by ~0.1.
            self.assertGreaterEqual(pixel_values.min().item(), 0.0)
            self.assertLessEqual(pixel_values.max().item(), 1.0)
            # Resizing in uint8 would quantize to multiples of 1/255, the float path does not.
            off_grid = (pixel_values * 255 - (pixel_values * 255).round()).abs().max().item()
            self.assertGreater(off_grid, 1e-3)

    def test_resize_clips_overshoot_without_rescale(self):
        """
        `do_rescale=False` is documented as "the caller already passes pixel values in `[0, 1]`", so the overshoot
        has to be clipped against 1 rather than against 255 on that path too. Clipping against 255 is a no-op for
        unit-interval floats and lets the bicubic ringing reach the model.
        """
        # A one pixel wide white bar on black maximizes bicubic ringing.
        image = torch.zeros(3, 64, 64, dtype=torch.float32)
        image[:, :, 30:34] = 1.0

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            pixel_values = image_processor(images=image, do_rescale=False, return_tensors="pt")["pixel_values"]

            self.assertEqual(pixel_values.dtype, torch.float32)
            self.assertGreaterEqual(pixel_values.min().item(), 0.0)
            self.assertLessEqual(pixel_values.max().item(), 1.0)

    def test_resize_does_not_clip_float_images_outside_the_unit_interval(self):
        """
        A float image that is not in `[0, 1]` lives in the same `[0, 255]` range as an integer one, so clipping it
        against 1 would saturate almost every pixel to white instead of only trimming the bicubic ringing.
        """
        image = torch.zeros(3, 64, 64, dtype=torch.float32)
        image[:, :, 30:34] = 255.0

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            no_rescale = image_processor(images=image, do_rescale=False, return_tensors="pt")["pixel_values"]
            # Same pixel content as an integer tensor, which is bounded by 255 and rescaled to the unit interval.
            rescaled = image_processor(images=image.to(torch.uint8), return_tensors="pt")["pixel_values"]

            self.assertGreaterEqual(no_rescale.min().item(), 0.0)
            self.assertLessEqual(no_rescale.max().item(), 255.0)
            self.assertGreater(no_rescale.max().item(), 1.0)
            torch.testing.assert_close(no_rescale / 255, rescaled, rtol=0, atol=1e-6)

    def test_post_process_requires_target_sizes(self):
        outputs = _dummy_outputs()
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            with self.assertRaises(ValueError):
                image_processor.post_process_object_detection(outputs, threshold=0.0)
