# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision
from transformers.utils import is_torch_available, is_torchvision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers.models.cellpose.image_processing_cellpose import (
        CellposeImageProcessor,
        _fill_holes,
        _get_padding,
        _tile_origins,
    )


def _disk_logits(height, width, centers, radius):
    """Cellpose-like (flow_y, flow_x, cellprob) with flows of magnitude 5 pointing to each disk center."""
    yy, xx = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float(), indexing="ij")
    logits = torch.zeros(3, height, width)
    logits[2] = -5.0
    disks = []
    for cy, cx in centers:
        offset = torch.stack([cy - yy, cx - xx])
        norm = offset.norm(dim=0) + 1e-6
        inside = norm <= radius
        logits[:2, inside] = (5 * offset / norm)[:, inside]
        logits[2, inside] = 5.0
        disks.append(inside)
    return logits, disks


class CellposeImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # images up to 240 pixels fit in a single 256 tile once padded, so there is one output per image
        kwargs.setdefault("max_resolution", 240)
        kwargs.setdefault("tile_size", 256)
        kwargs.setdefault("tile_overlap", 0.1)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        return 3, self.tile_size, self.tile_size  # only the first 3 channels are kept


@require_torch
@require_torchvision
class CellposeImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = CellposeImageProcessingTester

    def setUp(self):
        super().setUp()
        self.processor = CellposeImageProcessor()

    def _tile_like_preprocess(self, logits):
        """Pad and cut full-image logits into tiles the way `preprocess` does (without normalization)."""
        size, overlap = self.processor.tile_size, self.processor.tile_overlap
        top, bottom, left, right = _get_padding(*logits.shape[-2:], size)
        padded = torch.nn.functional.pad(logits, (left, right, top, bottom))
        tiles = [padded[:, y : y + size, x : x + size] for y, x in _tile_origins(*padded.shape[-2:], size, overlap)]
        return torch.stack(tiles), tuple(logits.shape[-2:])

    def test_preprocess_shapes_and_normalization(self):
        image = (np.random.default_rng(0).random((300, 530, 2)) * 4000).astype(np.uint16)
        inputs = self.processor(image, return_tensors="pt")
        # padded to 320 x 560 -> 2 x 3 tiles of 256
        self.assertEqual(tuple(inputs.pixel_values.shape), (6, 3, 256, 256))
        # third channel is missing in the input -> zeros
        self.assertEqual(inputs.pixel_values[:, 2].abs().max().item(), 0.0)

        batch = self.processor([np.zeros((100, 120)), torch.rand(3, 256, 256)], return_tensors="pt")
        self.assertEqual(tuple(batch.pixel_values.shape), (5, 3, 256, 256))
        self.assertEqual(batch.pixel_values[0].abs().max().item(), 0.0)  # flat image -> zeros

    def test_tiling_stitching_round_trip(self):
        rng = np.random.default_rng(0)
        for height, width in [(100, 120), (256, 256), (300, 530)]:
            image = rng.random((height, width, 3)).astype(np.float32)
            inputs = self.processor(image, return_tensors="pt")
            # stitching the tiles themselves gives back the normalized image, padding cropped away
            stitched = self.processor.stitch_tiles(inputs.pixel_values, [(height, width)])[0]
            self.assertEqual(tuple(stitched.shape), (3, height, width))
            image_t = torch.from_numpy(image).permute(2, 0, 1)
            low = torch.quantile(image_t.flatten(1), 0.01, dim=1)[:, None, None]
            high = torch.quantile(image_t.flatten(1), 0.99, dim=1)[:, None, None]
            torch.testing.assert_close(stitched, (image_t - low) / (high - low), atol=1e-5, rtol=1e-5)

    def test_post_process_disks(self):
        centers = [(40, 50), (120, 200), (220, 420)]
        logits, disks = _disk_logits(300, 530, centers, radius=15)
        tiles, size = self._tile_like_preprocess(logits)
        results = self.processor.post_process_instance_segmentation(SimpleNamespace(logits=tiles), [size])
        masks = results[0]["segmentation"]
        self.assertEqual(masks.dtype, torch.long)
        self.assertEqual(tuple(masks.shape), (300, 530))
        self.assertEqual(tuple(results[0]["flows"].shape), (2, 300, 530))
        self.assertNotIn("segmentation_scores", results[0])
        self.assertEqual(masks.max().item(), len(centers))
        self.assertEqual([info["id"] for info in results[0]["segments_info"]], [1, 2, 3])
        self.assertTrue(all(info["score"] > 0.5 for info in results[0]["segments_info"]))
        self.assertEqual(sorted(masks.unique().tolist()), list(range(len(centers) + 1)))
        for disk in disks:
            label = masks[disk].mode().values
            self.assertGreater(label.item(), 0)
            overlap = ((masks == label) & disk).sum() / (masks == label).sum()
            self.assertGreater(overlap.item(), 0.9)
            self.assertGreater(((masks == label) & disk).sum() / disk.sum(), 0.9)

    def test_post_process_min_size(self):
        logits, _ = _disk_logits(120, 150, [(40, 40), (90, 110)], radius=12)
        small, _ = _disk_logits(120, 150, [(20, 120)], radius=3)  # 29 pixels, much smaller than the two others
        inside = small[2] > 0
        logits[:, inside] = small[:, inside]
        tiles, size = self._tile_like_preprocess(logits)
        outputs = SimpleNamespace(logits=tiles)
        all_masks = self.processor.post_process_instance_segmentation(outputs, [size], min_size=0)[0]["segmentation"]
        filtered = self.processor.post_process_instance_segmentation(outputs, [size], min_size=100)[0]["segmentation"]
        self.assertEqual(filtered.max().item(), 2)
        self.assertEqual(all_masks.max().item(), 3)

    def test_post_process_flow_threshold(self):
        logits, disks = _disk_logits(120, 150, [(40, 40), (80, 110)], radius=15)
        # rotate the second cell's flows by 60 degrees: pixels still converge, but flows no longer match the shape
        cos, sin = 0.5, 3**0.5 / 2
        flow_y, flow_x = logits[0, disks[1]].clone(), logits[1, disks[1]].clone()
        logits[0, disks[1]], logits[1, disks[1]] = cos * flow_y - sin * flow_x, sin * flow_y + cos * flow_x
        tiles, size = self._tile_like_preprocess(logits)
        outputs = SimpleNamespace(logits=tiles)
        kept = self.processor.post_process_instance_segmentation(outputs, [size], flow_threshold=0)[0]["segmentation"]
        filtered = self.processor.post_process_instance_segmentation(outputs, [size], flow_threshold=0.4)[0][
            "segmentation"
        ]
        self.assertEqual(kept.max().item(), 2)
        self.assertEqual(filtered.max().item(), 1)
        self.assertGreater(((filtered == 1) & disks[0]).sum() / disks[0].sum(), 0.9)
        self.assertEqual(filtered[disks[1]].max().item(), 0)

    def test_post_process_threshold_and_scores(self):
        logits, disks = _disk_logits(120, 150, [(40, 40), (80, 110)], radius=15)
        logits[2, disks[0]] = 0.5  # first cell: probability ~0.62, second: ~0.99
        tiles, size = self._tile_like_preprocess(logits)
        outputs = SimpleNamespace(logits=tiles)
        result = self.processor.post_process_instance_segmentation(
            outputs, [size], threshold=0.9, return_segmentation_scores=True
        )[0]
        self.assertEqual(result["segmentation"].max().item(), 1)  # renumbered
        self.assertGreater(((result["segmentation"] == 1) & disks[1]).sum() / disks[1].sum(), 0.9)
        self.assertEqual([info["id"] for info in result["segments_info"]], [1])
        self.assertGreater(result["segments_info"][0]["score"], 0.9)
        torch.testing.assert_close(result["segmentation_scores"], logits[2].sigmoid(), atol=1e-5, rtol=1e-5)

    def test_tile_overlap(self):
        image = np.random.default_rng(0).random((300, 530)).astype(np.float32)
        default = self.processor(image, return_tensors="pt").pixel_values
        inputs = self.processor(image, tile_overlap=0.5, return_tensors="pt")
        self.assertGreater(len(inputs.pixel_values), len(default))
        # post-processing must stitch with the overlap used at preprocessing time
        logits, _ = _disk_logits(300, 530, [(40, 50)], radius=15)
        top, bottom, left, right = _get_padding(300, 530, 256)
        padded = torch.nn.functional.pad(logits, (left, right, top, bottom))
        tiles = [padded[:, y : y + 256, x : x + 256] for y, x in _tile_origins(*padded.shape[-2:], 256, 0.5)]
        self.assertEqual(len(tiles), len(inputs.pixel_values))
        result = self.processor.post_process_instance_segmentation(
            SimpleNamespace(logits=torch.stack(tiles)), [(300, 530)], tile_overlap=0.5
        )[0]
        self.assertEqual(result["segmentation"].max().item(), 1)

    def test_post_process_empty(self):
        logits = torch.zeros(3, 64, 64)
        logits[2] = -5.0
        tiles, size = self._tile_like_preprocess(logits)
        result = self.processor.post_process_instance_segmentation(SimpleNamespace(logits=tiles), [size])[0]
        self.assertEqual(result["segmentation"].abs().max().item(), 0)

    def test_fill_holes(self):
        masks = torch.zeros(12, 14, dtype=torch.long)
        masks[1:8, 1:8] = 1
        masks[3:6, 3:6] = 0  # hole
        masks[4, 4] = 2  # cell inside the hole is absorbed, like in the original
        masks[2:10, 9:13] = 3
        masks[4:6, 12] = 0  # notch open to the bounding box border is not a hole
        filled = _fill_holes(masks)
        self.assertTrue((filled[1:8, 1:8] == 1).all())
        self.assertTrue((filled[4:6, 12] == 0).all())
        self.assertTrue((filled[2:10, 9:12] == 3).all())
