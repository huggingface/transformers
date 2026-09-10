# Copyright 2023 HuggingFace Inc.
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
import inspect
import json
import os
import pathlib
import sys
import tempfile
import warnings
from typing import Any

import numpy as np
import pytest

from transformers import AutoImageProcessor, BatchFeature
from transformers.image_utils import AnnotationFormat, ImageInput
from transformers.models.auto.image_processing_auto import (
    IMAGE_PROCESSOR_MAPPING_NAMES,
    get_image_processor_class_from_name,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from .test_preprocessing_common import PreprocessingTesterMixin


if is_torch_available():
    import torch

    from transformers.modeling_outputs import SemanticSegmenterOutput

if is_vision_available():
    from PIL import Image


_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(_parent_dir, "utils"))
from fetch_hub_objects_for_ci import url_to_local_path  # noqa: E402


COCO_CATS_IMAGE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
)

ADE20K_IMAGE_URLS = [
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000002.jpg",
]
ADE20K_SEGMENTATION_MAP_URLS = [
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.png",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000002.png",
]


def load_test_image(url: str):
    """
    Loads a test image from a given URL, properly routing through HuggingFace's
    local hub caching mechanism using url_to_local_path.
    """
    from transformers.image_utils import load_image

    return load_image(url_to_local_path(url))


def prepare_image_inputs(
    batch_size,
    min_resolution,
    max_resolution,
    num_channels,
    size_divisor=None,
    equal_resolution=False,
    numpify=False,
    torchify=False,
):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    image_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            # To avoid getting image width/height 0
            if size_divisor is not None:
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        image_inputs.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if torchify:
        image_inputs = [torch.from_numpy(image) for image in image_inputs]

    if numpify:
        # Numpy images are typically in channels last format
        image_inputs = [image.transpose(1, 2, 0) for image in image_inputs]

    return image_inputs


def prepare_video(num_frames, num_channels, width=10, height=10, numpify=False, torchify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for frame_idx in range(num_frames):
        video.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if torchify:
        video = [torch.from_numpy(frame) for frame in video]

    return video


def prepare_video_inputs(
    batch_size,
    num_frames,
    num_channels,
    min_resolution,
    max_resolution,
    equal_resolution=False,
    numpify=False,
    torchify=False,
):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for _ in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        video = prepare_video(
            num_frames=num_frames,
            num_channels=num_channels,
            width=width,
            height=height,
            numpify=numpify,
            torchify=torchify,
        )
        video_inputs.append(video)

    return video_inputs


class ImageProcessingTester:
    """Base class for the `<Model>ImageProcessingTester` classes used by `ImageProcessingTestMixin`."""

    def prepare_image_inputs(
        self,
        batch_size=None,
        min_resolution=None,
        max_resolution=None,
        num_channels=None,
        size_divisor=None,
        equal_resolution=False,
        numpify=False,
        torchify=False,
    ):
        return prepare_image_inputs(
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_channels=self.num_channels if num_channels is None else num_channels,
            min_resolution=self.min_resolution if min_resolution is None else min_resolution,
            max_resolution=self.max_resolution if max_resolution is None else max_resolution,
            size_divisor=size_divisor,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

    def prepare_semantic_segmentation_inputs_ade20k(self, batched: bool = False):
        """Loads image/segmentation map pairs from ADE20k as PIL images."""
        num_inputs = 2 if batched else 1
        images = [load_test_image(url) for url in ADE20K_IMAGE_URLS[:num_inputs]]
        # `load_test_image` converts the single channel label maps to RGB, duplicating the labels across channels
        segmentation_maps = [
            Image.fromarray(np.array(load_test_image(url))[..., 0])
            for url in ADE20K_SEGMENTATION_MAP_URLS[:num_inputs]
        ]
        if batched:
            return images, segmentation_maps
        return images[0], segmentation_maps[0]

    def expected_output_image_shape(self, images: list[ImageInput]) -> tuple[int, ...]:
        crop_size = getattr(self, "crop_size", None)
        if crop_size is not None:
            return self.num_channels, crop_size["height"], crop_size["width"]

        if "shortest_edge" in self.size:
            # Images are resized so that their shortest edge matches `size["shortest_edge"]` while keeping the aspect
            # ratio, then padded to the largest height and width in the batch.
            shortest_edge = self.size["shortest_edge"]
            expected_sizes = []
            for image in images:
                if isinstance(image, Image.Image):
                    width, height = image.size
                elif isinstance(image, np.ndarray):
                    height, width = image.shape[0], image.shape[1]
                else:
                    height, width = image.shape[1], image.shape[2]
                if width < height:
                    expected_sizes.append((int(shortest_edge * height / width), shortest_edge))
                elif width > height:
                    expected_sizes.append((shortest_edge, int(shortest_edge * width / height)))
                else:
                    expected_sizes.append((shortest_edge, shortest_edge))
            expected_height = max(expected_size[0] for expected_size in expected_sizes)
            expected_width = max(expected_size[1] for expected_size in expected_sizes)
            return self.num_channels, expected_height, expected_width

        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_post_process_semantic_segmentation_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = {
            "outputs": SemanticSegmenterOutput(
                logits=torch.randn(self.batch_size, self.num_labels, self.size["height"], self.size["width"])
            )
        }
        expected_shape = {
            "num_labels": self.num_labels,
            "height": self.size["height"],
            "width": self.size["width"],
        }
        return inputs, expected_shape


class ImageProcessingTestMixin(PreprocessingTesterMixin):
    test_cast_dtype = None

    def setUp(self):
        # Infer model_name from test folder (parent of this test file)

        test_file_path = pathlib.Path(sys.modules[self.__class__.__module__].__file__).resolve()
        model_name = test_file_path.parent.name
        try:
            image_processing_classes_names = IMAGE_PROCESSOR_MAPPING_NAMES[model_name]
        except KeyError:
            raise ValueError(f"Override `setUp` in your test class to provide custom setup for {model_name}.")
        self.image_processing_classes = {
            backend_name: get_image_processor_class_from_name(class_name)
            for backend_name, class_name in image_processing_classes_names.items()
        }

    # ── `PreprocessingTesterMixin` surface ────────────────────────────────

    #: `default_to_square` and `data_format` are backend-specific by design.
    backend_specific_keys = {"default_to_square", "data_format"}

    @property
    def processing_classes(self) -> dict:
        return self.image_processing_classes

    @property
    def processor_dict(self) -> dict:
        return self.image_processor_dict

    @property
    def auto_class(self):
        return AutoImageProcessor

    def _prepare_inputs(self):
        # Equal resolution: models that pad or resize to a multiple (glpn, swin2sr, vitpose)
        # cannot stack a ragged batch, and this fixture only needs to be *callable*.
        return self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

    def _assert_tensors_equivalence(self, tensor1, tensor2, atol=1e-1, rtol=1e-3, mean_atol=5e-3):
        """Assert that two tensors are equivalent within specified tolerances."""
        torch.testing.assert_close(tensor1, tensor2, atol=atol, rtol=rtol)
        self.assertLessEqual(torch.mean(torch.abs(tensor1 - tensor2)).item(), mean_atol)

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_test_image(COCO_CATS_IMAGE_URL)

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @require_vision
    @require_torch
    def test_backends_equivalence_batched(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processing_classes.values():
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = image_processing_class(**self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_image_processor_preprocess_arguments(self):
        is_tested = False

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(image_processor, "_valid_processor_keys") and hasattr(image_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(image_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = image_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(image_processor.preprocess, "_filter_out_non_signature_kwargs"):
                if hasattr(self.image_processor_tester, "prepare_image_inputs"):
                    inputs = self.image_processor_tester.prepare_image_inputs()
                elif hasattr(self.image_processor_tester, "prepare_video_inputs"):
                    inputs = self.image_processor_tester.prepare_video_inputs()
                else:
                    self.skipTest(reason="No valid input preparation method found")

                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    image_processor(inputs, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_torchvision_backend(self):
        # Test compilation with torchvision backend (equivalent to fast processor)
        if "torchvision" not in self.image_processing_classes:
            self.skipTest("Skipping compilation test as torchvision backend is not available")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.image_processing_classes["torchvision"](**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")
        # torch.compile can introduce 1-level rounding differences in uint8 resize; after normalization this can reach 2 / 255.
        self._assert_tensors_equivalence(
            output_eager.pixel_values, output_compiled.pixel_values, atol=1e-2, rtol=1e-4, mean_atol=1e-5
        )

    def test_new_models_require_torchvision_backend(self):
        """
        Test that new models support the torchvision backend.
        For more information on how to implement backend support, see this issue: https://github.com/huggingface/transformers/issues/36978,
        and ping @yonigozlan for help.
        """
        # Check if torchvision backend is available
        if "torchvision" in self.image_processing_classes:
            return
        if not self.image_processing_classes:
            self.skipTest("No image processing class defined")

        # Old models are those whose image processing file was first committed before 2025-09-01.
        # fmt: off
        _OLD_MODELS = {
            "aria", "beit", "bit", "blip", "bridgetower", "chameleon", "chinese_clip",
            "clip", "cohere2_vision", "conditional_detr", "convnext", "deepseek_vl",
            "deepseek_vl_hybrid", "deformable_detr", "deit", "depth_pro", "detr",
            "dinov3_vit", "donut", "dpt", "efficientloftr", "efficientnet", "eomt",
            "flava", "fuyu", "gemma3", "glm4v", "glpn", "got_ocr2", "grounding_dino",
            "idefics", "idefics2", "idefics3", "imagegpt", "janus", "kosmos2_5",
            "layoutlmv2", "layoutlmv3", "levit", "superglue", "lightglue", "llama4",
            "llava", "llava_next", "llava_onevision", "mask2former", "maskformer",
            "mllama", "mobilenet_v1", "mobilenet_v2", "mobilevit", "nougat",
            "oneformer", "ovis2", "owlv2", "owlvit", "perceiver", "perception_lm",
            "phi4_multimodal", "pix2struct", "pixtral", "poolformer",
            "prompt_depth_anything", "pvt", "qwen2_vl", "rt_detr", "sam", "sam2",
            "segformer", "seggpt", "siglip", "siglip2", "smolvlm", "superpoint",
            "swin2sr", "textnet", "tvp", "videomae", "vilt", "vit", "vitmatte",
            "vitpose", "vivit", "yolos", "zoedepth",
        }
        # fmt: on

        test_file_path = pathlib.Path(sys.modules[self.__class__.__module__].__file__).resolve()
        model_type = test_file_path.parent.name
        is_old_model = model_type in _OLD_MODELS
        # New models must support torchvision backend
        self.assertTrue(
            is_old_model,
            f"Model '{model_type}' was added after the cutoff date and must support "
            f"the torchvision backend. Please ensure torchvision backend is available.",
        )

    def test_post_process_test_mixin_inheritance(self):
        """
        Ensures that we have the post-process tester mixin if the processor implements the corresponding method.
        The test will fail otherwise, forcing the mixin to be added -- and ensuring proper test coverage.
        """
        METHOD_TO_MIXIN = {
            "post_process_semantic_segmentation": PostProcessSemanticSegmentationTestMixin,
        }
        for method_name, mixin_class in METHOD_TO_MIXIN.items():
            implements_method = any(
                hasattr(image_processing_class, method_name)
                for image_processing_class in self.image_processing_classes.values()
            )
            if implements_method:
                self.assertTrue(
                    issubclass(self.__class__, mixin_class),
                    msg=(
                        f"This processor implements `{method_name}`, so the tester must inherit from "
                        f"`{mixin_class.__name__}` to run the corresponding tests. Either add the inheritance "
                        f"or, if the processor only partially supports `{method_name}`, overwrite the test."
                    ),
                )
            else:
                self.assertFalse(
                    issubclass(self.__class__, mixin_class),
                    msg=(
                        f"This processor does not implement `{method_name}`, so the tester must not inherit from "
                        f"`{mixin_class.__name__}`. If the processor was recently updated to support "
                        f"`{method_name}`, add the `{mixin_class.__name__}` inheritance instead."
                    ),
                )


class PostProcessSemanticSegmentationTestMixin:
    @require_torch
    def test_post_process_semantic_segmentation(self):
        for image_processing_class in self.image_processing_classes.values():
            with self.subTest(image_processing_class):
                image_processor = image_processing_class(**self.image_processor_dict)
                inputs, expected_shape = (
                    self.image_processor_tester.prepare_post_process_semantic_segmentation_inputs()
                )

                segmentation = image_processor.post_process_semantic_segmentation(**inputs)

                self.assertEqual(len(segmentation), self.image_processor_tester.batch_size)
                self.assertEqual(segmentation[0].shape, (expected_shape["height"], expected_shape["width"]))

                # return_segmentation_scores=True: returns list of SemanticSegmentationPostProcessorOutput
                segmentation_output = image_processor.post_process_semantic_segmentation(
                    **inputs, return_segmentation_scores=True
                )
                self.assertEqual(len(segmentation_output), self.image_processor_tester.batch_size)
                self.assertTrue(torch.equal(segmentation_output[0].segmentation, segmentation[0]))
                self.assertEqual(
                    segmentation_output[0].segmentation_scores.shape,
                    (expected_shape["num_labels"], expected_shape["height"], expected_shape["width"]),
                )

    @require_torch
    def test_post_process_semantic_segmentation_target_sizes(self):
        inputs, expected_shape = self.image_processor_tester.prepare_post_process_semantic_segmentation_inputs()

        if "target_sizes" in inputs:
            self.skipTest(reason="target_sizes already in required inputs")

        for image_processing_class in self.image_processing_classes.values():
            with self.subTest(image_processing_class):
                image_processor = image_processing_class(**self.image_processor_dict)

                target_sizes = [(1, 4) for _ in range(self.image_processor_tester.batch_size)]
                segmentation_resized = image_processor.post_process_semantic_segmentation(
                    **inputs, target_sizes=target_sizes
                )
                self.assertEqual(segmentation_resized[0].shape, target_sizes[0])

                # return_segmentation_scores=True with target_sizes
                segmentation_output_resized = image_processor.post_process_semantic_segmentation(
                    **inputs, target_sizes=target_sizes, return_segmentation_scores=True
                )
                self.assertTrue(torch.equal(segmentation_output_resized[0].segmentation, segmentation_resized[0]))
                self.assertEqual(
                    segmentation_output_resized[0].segmentation_scores.shape,
                    (expected_shape["num_labels"],) + target_sizes[0],
                )

                # raise ValueError if target_sizes has wrong length
                with pytest.raises(ValueError):
                    image_processor.post_process_semantic_segmentation(**inputs, target_sizes=target_sizes + [(1, 4)])


class AnnotationFormatTestMixin:
    def test_processor_can_use_legacy_annotation_format(self):
        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        fixtures_path = pathlib.Path(__file__).parent / "fixtures" / "tests_samples" / "COCO"

        with open(fixtures_path / "coco_annotations.txt") as f:
            detection_target = json.loads(f.read())

        detection_annotations = {"image_id": 39769, "annotations": detection_target}

        detection_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": detection_annotations,
            "return_tensors": "pt",
        }

        with open(fixtures_path / "coco_panoptic_annotations.txt") as f:
            panoptic_target = json.loads(f.read())

        panoptic_annotations = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": panoptic_target}

        masks_path = pathlib.Path(fixtures_path / "coco_panoptic")

        panoptic_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": panoptic_annotations,
            "return_tensors": "pt",
            "masks_path": masks_path,
        }

        test_cases = [
            ("coco_detection", detection_params),
            ("coco_panoptic", panoptic_params),
            (AnnotationFormat.COCO_DETECTION, detection_params),
            (AnnotationFormat.COCO_PANOPTIC, panoptic_params),
        ]

        def _compare(a, b) -> None:
            if isinstance(a, (dict, BatchFeature)):
                self.assertEqual(a.keys(), b.keys())
                for k, v in a.items():
                    _compare(v, b[k])
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b))
                for idx in range(len(a)):
                    _compare(a[idx], b[idx])
            elif isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
            elif isinstance(a, str):
                self.assertEqual(a, b)

        for annotation_format, params in test_cases:
            with self.subTest(annotation_format):
                image_processor_params = {**image_processor_dict, **{"format": annotation_format}}
                image_processor_first = self.image_processing_classes["torchvision"](**image_processor_params)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    image_processor_first.save_pretrained(tmpdirname)
                    image_processor_second = self.image_processing_classes["torchvision"].from_pretrained(tmpdirname)

                # check the 'format' key exists and that the dicts of the
                # first and second processors are equal
                self.assertIn("format", image_processor_first.to_dict().keys())
                self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

                # perform encoding using both processors and compare
                # the resulting BatchFeatures
                first_encoding = image_processor_first(**params)
                second_encoding = image_processor_second(**params)
                _compare(first_encoding, second_encoding)


COCO_DATASET_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/"


def load_coco_image(image_name: str):
    """
    Helper to load a COCO fixture image by its filename.
    """
    return load_test_image(f"{COCO_DATASET_URL}{image_name}")
