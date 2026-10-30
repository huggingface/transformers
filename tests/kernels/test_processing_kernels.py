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
# Run the slow tests: CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 pytest -sv tests/kernels/test_processing_kernels.py

import json
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from tokenizers import Tokenizer, models

from transformers import (
    CLIPImageProcessor,
    CLIPProcessor,
    CohereCompassImageProcessor,
    CohereCompassVideoProcessor,
    Ernie4_5_VLMoeImageProcessor,
    GlmImageImageProcessor,
    HunYuanVLImageProcessor,
    LevitImageProcessor,
    MiniMaxM3VLImageProcessor,
    PaddleOCRVLImageProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    Qwen3VLVideoProcessor,
    Sam2ImageProcessor,
    SegformerImageProcessor,
    SegformerImageProcessorPil,
    TokenizersBackend,
    ViTImageProcessor,
)
from transformers.image_utils import PILImageResampling, SizeDict, load_image
from transformers.integrations.hub_kernels import lazy_load_kernel
from transformers.integrations.hub_processing_kernels import kernel_can_resize, run_processing_kernel
from transformers.testing_utils import TestCasePlus, require_kernels, require_torch_gpu, slow, torch_device
from transformers.video_processing_utils import BaseVideoProcessor


PATCHIFY_KERNEL_PROCESSORS = [
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    Qwen3VLVideoProcessor,
    CohereCompassImageProcessor,
    CohereCompassVideoProcessor,
    Ernie4_5_VLMoeImageProcessor,
    GlmImageImageProcessor,
    HunYuanVLImageProcessor,
    MiniMaxM3VLImageProcessor,
    PaddleOCRVLImageProcessor,
]


def connected_component_areas_reference(regions):
    """Area of the 8-connected component of every pixel, by label propagation. Slow, for tests only."""
    flat_regions = regions.flatten(1)
    labels = torch.arange(1, flat_regions.shape[1] + 1, dtype=torch.float32, device=regions.device)
    labels = (labels * flat_regions).view(regions.shape)
    while True:
        propagated = torch.nn.functional.max_pool2d(labels, kernel_size=3, stride=1, padding=1) * regions
        if torch.equal(propagated, labels):
            break
        labels = propagated
    areas = torch.zeros_like(labels, dtype=torch.int32)
    for index in range(labels.shape[0]):
        _, inverse, counts = labels[index].unique(return_inverse=True, return_counts=True)
        areas[index] = counts[inverse].to(torch.int32) * regions[index]
    return areas


def patchify_kernel_layout_reference(
    inputs,
    target_sizes,
    image_mean,
    image_std,
    rescale_factor,
    interpolation,
    patch_size,
    block_size,
    temporal_patch_size,
):
    """What `resize_normalize_patchify` writes for `(T, C, H, W)` inputs whose frames are already at their target size."""
    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)
    item_patches = []
    for input_frames in inputs:
        input_frames = torch.cat(
            [input_frames, input_frames[-1:].expand(-input_frames.shape[0] % temporal_patch_size, -1, -1, -1)]
        )
        input_frames = (input_frames.float() * rescale_factor - mean) / std
        frame_count, channels, height, width = input_frames.shape
        patches = input_frames.reshape(
            frame_count // temporal_patch_size,
            temporal_patch_size,
            channels,
            height // (patch_size * block_size),
            block_size,
            patch_size,
            width // (patch_size * block_size),
            block_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        item_patches.append(patches.reshape(-1, channels * temporal_patch_size * patch_size * patch_size))
    return torch.cat(item_patches)


HUB_PROCESSING_KERNELS = "transformers.integrations.hub_processing_kernels"


@contextmanager
def loaded_kernel(kernel):
    """Serve `kernel` as the loaded Hub kernel, and accept CPU tensors as the kernel device."""
    with patch(f"{HUB_PROCESSING_KERNELS}.lazy_load_kernel", return_value=kernel):
        with patch(f"{HUB_PROCESSING_KERNELS}._KERNEL_DEVICE_TYPE", "cpu"):
            yield


class ReferenceConnectedComponentsKernel:
    """Stand-in for `kernels-community/cv-utils`: records the input of `connected_component_areas`, answers the reference."""

    def __init__(self):
        self.inputs = []

    def connected_component_areas(self, mask):
        self.inputs.append(mask)
        return connected_component_areas_reference(mask)


class TestProcessingKernels(TestCasePlus):
    """Dispatch and gating of processing kernels. Runs on CPU, `kernels` is not needed."""

    def setUp(self):
        super().setUp()
        # One 8x8 object with a 2x2 hole, and a 1 pixel island away from it.
        self.mask_logits = torch.full((1, 1, 16, 16), -1.0)
        self.mask_logits[..., 2:10, 2:10] = 1.0
        self.mask_logits[..., 5:7, 5:7] = -1.0
        self.mask_logits[..., 13, 13] = 1.0

    def post_process(self, use_kernels, **kwargs):
        processor = Sam2ImageProcessor(use_kernels=use_kernels)
        return processor.post_process_masks([self.mask_logits.clone()], [(16, 16)], **kwargs)[0]

    def test_run_processing_kernel_falls_back(self):
        regions = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        kernel = MagicMock()
        with patch(f"{HUB_PROCESSING_KERNELS}.lazy_load_kernel", return_value=kernel):
            self.assertIsNone(run_processing_kernel("connected_component_areas", regions))
        kernel.connected_component_areas.assert_not_called()
        with loaded_kernel(None):
            self.assertIsNone(run_processing_kernel("connected_component_areas", regions))
        self.assertIsNone(run_processing_kernel("not_a_registered_op", regions))

    def test_sam2_fills_holes_and_removes_sprinkles(self):
        kernel = ReferenceConnectedComponentsKernel()
        with loaded_kernel(kernel):
            masks = self.post_process(use_kernels=True, max_hole_area=4, max_sprinkle_area=1)
            too_small_thresholds = self.post_process(use_kernels=True, max_hole_area=3, max_sprinkle_area=0)
        self.assertEqual(kernel.inputs[0].shape, (1, 1, 16, 16))
        expected = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        expected[..., 2:10, 2:10] = True
        self.assertTrue(torch.equal(masks, expected))
        self.assertTrue(torch.equal(too_small_thresholds, self.post_process(use_kernels=False)))

    def test_sam2_small_regions_need_the_kernel(self):
        for use_kernels in (False, True):
            with self.assertRaisesRegex(ValueError, "connected components kernel"):
                self.post_process(use_kernels, max_hole_area=4, max_sprinkle_area=1)

    def test_kernel_can_resize(self):
        image = torch.zeros(3, 4, 4, dtype=torch.uint8)
        unsupported = {
            "lanczos": ([image], PILImageResampling.LANCZOS),
            "no_images": ([], PILImageResampling.BICUBIC),
            "float_images": ([torch.rand(3, 4, 4)], PILImageResampling.BICUBIC),
            "mixed_channels": ([image, torch.zeros(1, 4, 4, dtype=torch.uint8)], PILImageResampling.BICUBIC),
        }
        self.assertFalse(kernel_can_resize([image, image], PILImageResampling.BICUBIC))
        with patch(f"{HUB_PROCESSING_KERNELS}._KERNEL_DEVICE_TYPE", "cpu"):
            for name, (inputs, resample) in unsupported.items():
                with self.subTest(name):
                    self.assertFalse(kernel_can_resize(inputs, resample))
            self.assertTrue(kernel_can_resize([image, image], PILImageResampling.BICUBIC))

    def test_semantic_segmentation_calls_the_upsample_argmax_kernel(self):
        kernel = MagicMock()
        outputs = SimpleNamespace(logits=torch.randn(2, 5, 8, 8))
        processor = SegformerImageProcessor(use_kernels=True)
        with loaded_kernel(kernel):
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[(16, 16), (20, 12)])
            with_scores = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(16, 16), (20, 12)], return_segmentation_scores=True
            )
            native_size = processor.post_process_semantic_segmentation(outputs)
        self.assertIs(segmentation, kernel.upsample_argmax.return_value)
        kernel.upsample_argmax.assert_called_once_with(outputs.logits, [(16, 16), (20, 12)])
        self.assertEqual(with_scores[1].segmentation_scores.shape, (5, 20, 12))
        self.assertEqual(native_size[0].shape, (8, 8))

    def run_vit_with_kernel(self, **kwargs):
        kernel = MagicMock()
        images = [torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8) for _ in range(2)]
        with loaded_kernel(kernel):
            output = ViTImageProcessor(use_kernels=True)(images, return_tensors=None, **kwargs)
        return kernel, output

    def test_resize_normalize_calls_the_kernel(self):
        kernel, output = self.run_vit_with_kernel(do_center_crop=True, crop_size={"height": 6, "width": 6})
        self.assertIs(output["pixel_values"], kernel.resize_normalize.return_value)
        kernel.resize_normalize.assert_called_once()
        self.assertEqual(kernel.resize_normalize.call_args.kwargs, {"crop_size": (6, 6)})
        self.assertEqual(kernel.resize_normalize.call_args.args[5], "bilinear")

    def test_resize_normalize_falls_back_on_a_crop_larger_than_the_resize(self):
        kernel, output = self.run_vit_with_kernel(
            size={"height": 8, "width": 8}, do_center_crop=True, crop_size={"height": 8, "width": 10}
        )
        kernel.resize_normalize.assert_not_called()
        self.assertEqual(tuple(output["pixel_values"][0].shape), (3, 8, 10))

    def test_resize_normalize_receives_the_sizes_of_resize(self):
        images = [
            torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8),
            torch.randint(0, 255, (3, 50, 30), dtype=torch.uint8),
        ]
        size_dicts = {
            "shortest_edge": {"shortest_edge": 20},
            "shortest_and_longest_edge": {"shortest_edge": 20, "longest_edge": 25},
            "max_height_and_width": {"max_height": 20, "max_width": 20},
            "height_and_width": {"height": 16, "width": 24},
        }
        processor = ViTImageProcessor(use_kernels=True)
        for name, size in size_dicts.items():
            with self.subTest(name):
                kernel = MagicMock()
                with loaded_kernel(kernel):
                    processor(images, size=size, do_center_crop=False, return_tensors=None)
                self.assertEqual(
                    [tuple(size) for size in kernel.resize_normalize.call_args.args[1]],
                    [tuple(processor.resize(image, SizeDict(**size)).shape[-2:]) for image in images],
                )

    def test_qwen2_vl_uses_the_patchify_kernel_output(self):
        images = [torch.randint(0, 255, (3, 64, 96), dtype=torch.uint8)]
        pixel_values = torch.zeros(24, 3 * 2 * 14 * 14)
        kernel = MagicMock()
        kernel.resize_normalize_patchify.return_value = (pixel_values, torch.tensor([[1, 4, 6]]))
        with loaded_kernel(kernel):
            output = Qwen2VLImageProcessor(use_kernels=True)(images, return_tensors="pt")
            Qwen2VLImageProcessor(use_kernels=False)(images, return_tensors="pt")
        kernel.resize_normalize_patchify.assert_called_once()
        self.assertEqual(kernel.resize_normalize_patchify.call_args.args[6:], (14, 2, 2))
        self.assertIs(output["pixel_values"], pixel_values)
        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 4, 6]])

    def test_qwen2_vl_video_passes_one_target_size_per_video(self):
        videos = [
            torch.randint(0, 255, (3, 3, 64, 96), dtype=torch.uint8),
            torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8),
        ]
        kernel = MagicMock()
        kernel.resize_normalize_patchify.return_value = (torch.zeros(1, 1), torch.tensor([[2, 4, 6], [1, 2, 2]]))
        with loaded_kernel(kernel):
            output = Qwen2VLVideoProcessor(use_kernels=True)(
                videos, do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt"
            )
        kernel_inputs, target_sizes = kernel.resize_normalize_patchify.call_args.args[:2]
        self.assertEqual([input_frames.shape[0] for input_frames in kernel_inputs], [3, 2])
        self.assertEqual(len(target_sizes), 2)
        self.assertEqual(output["video_grid_thw"].tolist(), [[2, 4, 6], [1, 2, 2]])

    def test_processor_overriding_resize_does_not_use_the_resize_kernel(self):
        with patch("transformers.image_processing_backends.run_processing_kernel", self.fail):
            output = LevitImageProcessor(use_kernels=True)([torch.randint(0, 255, (3, 64, 96), dtype=torch.uint8)])
        self.assertIn("pixel_values", output)

    def test_processors_reaching_the_patchify_kernel_use_its_layout(self):
        qwen3_vl_image_settings = {"patch_size": 16, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
        cases = [(processor_class, {}) for processor_class in PATCHIFY_KERNEL_PROCESSORS]
        cases.append((Qwen2VLImageProcessor, qwen3_vl_image_settings))
        for processor_class, settings in cases:
            with self.subTest(processor_class=processor_class.__name__, **settings):
                is_video = issubclass(processor_class, BaseVideoProcessor)

                def run(height, width):
                    """The arguments the kernel receives, and the output of the processor without kernels."""
                    kernel = MagicMock()
                    kernel.resize_normalize_patchify.return_value = (torch.zeros(0), torch.zeros(1, 3))
                    shape = (3, 3, height, width) if is_video else (3, height, width)
                    inputs = [torch.randint(0, 256, shape, dtype=torch.uint8)]
                    call_kwargs = {"do_sample_frames": False} if is_video else {}
                    with loaded_kernel(kernel):
                        processor_class(use_kernels=True, **settings)(inputs, return_tensors="pt", **call_kwargs)
                    self.assertTrue(kernel.resize_normalize_patchify.called, "the processor did not call the kernel")
                    output = processor_class(use_kernels=False, **settings)(inputs, return_tensors="pt", **call_kwargs)
                    output_name = "pixel_values_videos" if is_video else "pixel_values"
                    return kernel.resize_normalize_patchify.call_args.args, output[output_name]

                height, width = 112, 168
                for _ in range(4):
                    kernel_arguments, pixel_values = run(height, width)
                    if tuple(kernel_arguments[1][0]) == (height, width):
                        break
                    height, width = kernel_arguments[1][0]
                kernel_inputs, target_sizes = kernel_arguments[:2]
                self.assertEqual(
                    [tuple(input_frames.shape[-2:]) for input_frames in kernel_inputs],
                    [tuple(size) for size in target_sizes],
                )
                torch.testing.assert_close(
                    pixel_values.reshape(-1), patchify_kernel_layout_reference(*kernel_arguments).reshape(-1)
                )

    def test_use_kernels_is_a_runtime_flag(self):
        processor = Sam2ImageProcessor(use_kernels=True)
        self.assertNotIn("use_kernels", processor.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            self.assertFalse(Sam2ImageProcessor.from_pretrained(tmp_dir).use_kernels)
            self.assertTrue(Sam2ImageProcessor.from_pretrained(tmp_dir, use_kernels=True).use_kernels)

    def test_pil_processor_rejects_use_kernels(self):
        with self.assertRaisesRegex(ValueError, "torchvision"):
            SegformerImageProcessorPil(use_kernels=True)

    def test_processor_passes_use_kernels_to_the_image_processor_only(self):
        tokenizer = TokenizersBackend(
            tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
        )
        with tempfile.TemporaryDirectory() as saved, tempfile.TemporaryDirectory() as resaved:
            CLIPProcessor(image_processor=CLIPImageProcessor(), tokenizer=tokenizer).save_pretrained(saved)
            processor = CLIPProcessor.from_pretrained(saved, use_kernels=True)
            self.assertTrue(processor.image_processor.use_kernels)
            self.assertNotIn("use_kernels", processor.tokenizer.init_kwargs)
            processor.save_pretrained(resaved)
            with open(os.path.join(resaved, "tokenizer_config.json"), encoding="utf-8") as tokenizer_config:
                self.assertNotIn("use_kernels", json.load(tokenizer_config))
            self.assertFalse(CLIPProcessor.from_pretrained(resaved).image_processor.use_kernels)


@require_kernels
@require_torch_gpu
@slow
class TestConnectedComponentsKernel(TestCasePlus):
    """The `kernels-community/cv-utils` connected components kernel against the reference, through the processor."""

    def test_matches_reference(self):
        generator = torch.Generator().manual_seed(0)
        masks = torch.randn(4, 3, 63, 65, generator=generator).to(torch_device)
        processor = Sam2ImageProcessor(use_kernels=True)
        kernel = lazy_load_kernel("cv-utils")
        self.assertIsNotNone(kernel, "the cv-utils kernel did not load")
        regions = (masks.flatten(0, 1) > 0).unsqueeze(1)
        areas = kernel.connected_component_areas(regions)
        torch.testing.assert_close(areas.to(torch.int32).cpu(), connected_component_areas_reference(regions.cpu()))
        filled = processor.post_process_masks([masks], [(63, 65)], max_hole_area=8, max_sprinkle_area=8)[0]
        with patch(f"{HUB_PROCESSING_KERNELS}.lazy_load_kernel", return_value=ReferenceConnectedComponentsKernel()):
            expected = processor.post_process_masks([masks], [(63, 65)], max_hole_area=8, max_sprinkle_area=8)[0]
        self.assertTrue(torch.equal(filled, expected))


@require_kernels
@require_torch_gpu
@slow
class TestUpsampleArgmaxKernel(TestCasePlus):
    def test_matches_default(self):
        kernel = lazy_load_kernel("cv-utils")
        self.assertIsNotNone(kernel, "the cv-utils kernel did not load")
        if not hasattr(kernel, "upsample_argmax"):
            self.skipTest("`upsample_argmax` is not available in this build of cv-utils")
        logits = torch.nn.functional.avg_pool2d(torch.randn(2, 150, 64, 64), 5, stride=1, padding=2) * 5
        outputs = SimpleNamespace(logits=logits.to(torch_device))
        target_sizes = [(256, 341), (480, 640)]
        with patch.object(kernel, "upsample_argmax", wraps=kernel.upsample_argmax) as upsample_argmax:
            with_kernels = SegformerImageProcessor(use_kernels=True).post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        upsample_argmax.assert_called_once()
        without_kernels = SegformerImageProcessor().post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        for with_kernel, without_kernel in zip(with_kernels, without_kernels):
            self.assertEqual(with_kernel.shape, without_kernel.shape)
            # The kernel upsamples on float16 tensor cores here, so near-ties between two classes can flip.
            self.assertLess((with_kernel != without_kernel).float().mean().item(), 1e-3)


@require_kernels
@require_torch_gpu
@slow
class TestResizeKernels(TestCasePlus):
    """The resize kernels of `kernels-community/cv-utils` against the default torchvision path, on photos."""

    def setUp(self):
        super().setUp()
        image = torch.from_numpy(np.array(load_image("http://images.cocodataset.org/val2017/000000039769.jpg")))
        image = image.permute(2, 0, 1).contiguous().to(torch_device)
        self.images = [image, image[:, :300, :500].contiguous(), image[:, 100:, 50:].contiguous()]

    def assert_kernel_matches_default(self, operation, processor_class, call):
        kernel = lazy_load_kernel("cv-utils")
        self.assertIsNotNone(kernel, "the cv-utils kernel did not load")
        if not hasattr(kernel, operation):
            self.skipTest(f"`{operation}` is not available in this build of cv-utils")
        with patch.object(kernel, operation, wraps=getattr(kernel, operation)) as kernel_operation:
            with_kernels = call(processor_class(use_kernels=True))
        kernel_operation.assert_called_once()
        without_kernels = call(processor_class(use_kernels=False))
        for key in without_kernels:
            self.assertEqual(with_kernels[key].shape, without_kernels[key].shape)
        # Both paths round to uint8 after each resize pass, in a different order of operations.
        one_level = 1 / 255 / min(processor_class().image_std)
        difference = (
            with_kernels.data[list(without_kernels)[0]] - without_kernels.data[list(without_kernels)[0]]
        ).abs()
        self.assertLess(difference.mean().item(), one_level / 4)
        self.assertLess((difference > one_level * 1.01).float().mean().item(), 1e-3)

    def test_resize_normalize(self):
        self.assert_kernel_matches_default(
            "resize_normalize", ViTImageProcessor, lambda processor: processor(self.images, return_tensors="pt")
        )

    def test_resize_normalize_patchify_images(self):
        self.assert_kernel_matches_default(
            "resize_normalize_patchify",
            Qwen2VLImageProcessor,
            lambda processor: processor(self.images, return_tensors="pt"),
        )

    def test_resize_normalize_patchify_video(self):
        video = torch.stack([self.images[0]] * 5)
        self.assert_kernel_matches_default(
            "resize_normalize_patchify",
            Qwen2VLVideoProcessor,
            lambda processor: processor(
                [video], do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt"
            ),
        )
