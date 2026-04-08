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
"""Testing suite for the PyTorch SAM3 Video model."""

import gc
import unittest

from transformers.testing_utils import (
    backend_empty_cache,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available
from transformers.video_utils import load_video


if is_torch_available():
    import torch

    from transformers import Sam3VideoModel, Sam3VideoProcessor


def prepare_video():
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    raw_video, _ = load_video(video_url)
    return raw_video


@slow
class Sam3VideoModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        checkpoint_path = "facebook/sam3"
        self.video_model = Sam3VideoModel.from_pretrained(checkpoint_path).to(torch.float32)
        self.processor = Sam3VideoProcessor.from_pretrained(checkpoint_path)
        self.video_model.to(torch_device)
        self.video_model.eval()

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_video_propagate_with_text_prompt(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(
            video=raw_video,
            inference_device=torch_device,
            processing_device="cpu",
            video_storage_device="cpu",
        )

        # Add text prompt
        text = "person"
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=text,
        )

        # Propagate through video frames
        outputs_per_frame = {}
        model_outputs_per_frame = {}
        for model_outputs in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=3,
        ):
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs
            model_outputs_per_frame[model_outputs.frame_idx] = model_outputs

        # Check we processed the expected number of frames
        self.assertGreaterEqual(len(outputs_per_frame), 1)
        self.assertLessEqual(len(outputs_per_frame), 4)  # frame 0 + up to 3 more

        # Check output structure for each frame
        for processed_outputs in outputs_per_frame.values():
            self.assertIn("object_ids", processed_outputs)
            self.assertIn("scores", processed_outputs)
            self.assertIn("boxes", processed_outputs)
            self.assertIn("masks", processed_outputs)

            num_objects = len(processed_outputs["object_ids"])
            if num_objects > 0:
                self.assertEqual(processed_outputs["scores"].shape, (num_objects,))
                self.assertEqual(processed_outputs["boxes"].shape, (num_objects, 4))
                self.assertEqual(
                    processed_outputs["masks"].shape, (num_objects, raw_video.shape[-3], raw_video.shape[-2])
                )
                # Check boxes are in XYXY format (absolute coordinates)
                boxes = processed_outputs["boxes"]
                self.assertTrue(torch.all(boxes[:, 2] >= boxes[:, 0]))  # x2 >= x1
                self.assertTrue(torch.all(boxes[:, 3] >= boxes[:, 1]))  # y2 >= y1

        # Check numeric values for first frame
        if len(outputs_per_frame) > 0:
            first_frame_idx = min(outputs_per_frame.keys())
            first_outputs = outputs_per_frame[first_frame_idx]
            num_objects = len(first_outputs["object_ids"])
            if num_objects > 0:
                # Move outputs to CPU for comparison (postprocess_outputs may return CPU tensors)
                object_ids = (
                    first_outputs["object_ids"].cpu()
                    if isinstance(first_outputs["object_ids"], torch.Tensor)
                    else torch.tensor(first_outputs["object_ids"])
                )
                scores = (
                    first_outputs["scores"].cpu()
                    if isinstance(first_outputs["scores"], torch.Tensor)
                    else torch.tensor(first_outputs["scores"])
                )
                boxes = (
                    first_outputs["boxes"].cpu()
                    if isinstance(first_outputs["boxes"], torch.Tensor)
                    else torch.tensor(first_outputs["boxes"])
                )
                masks = (
                    first_outputs["masks"].cpu()
                    if isinstance(first_outputs["masks"], torch.Tensor)
                    else torch.tensor(first_outputs["masks"])
                )

                torch.testing.assert_close(
                    object_ids,
                    torch.tensor([0, 1], dtype=torch.int64),
                )
                torch.testing.assert_close(
                    scores,
                    torch.tensor([0.968647837638855, 0.9736108779907227], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    boxes[0],
                    torch.tensor([146.0, 135.0, 291.0, 404.0], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    masks[0, :3, :3].float(),
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )

        # Check raw model_outputs mask values for first frame
        if len(model_outputs_per_frame) > 0:
            first_frame_idx = min(model_outputs_per_frame.keys())
            first_model_outputs = model_outputs_per_frame[first_frame_idx]
            num_objects = len(first_model_outputs.object_ids)
            if num_objects > 0:
                # Check raw mask from model_outputs (low-resolution, before post-processing)
                first_obj_id = first_model_outputs.object_ids[0]
                raw_mask = first_model_outputs.obj_id_to_mask[first_obj_id].cpu()
                torch.testing.assert_close(
                    raw_mask[:1, :3, :3].float(),
                    torch.tensor(
                        [
                            [
                                [-2.952317476272583, -5.94632625579834, -7.991223335266113],
                                [-6.916913986206055, -10.058566093444824, -11.114638328552246],
                                [-8.195585250854492, -9.787644386291504, -10.39273452758789],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                    atol=5e-3,  # Higher tolerance for raw logits
                    rtol=5e-3,
                )

        # Check numeric values for last frame (to verify propagation consistency)
        if len(outputs_per_frame) > 1:
            last_frame_idx = max(outputs_per_frame.keys())
            last_outputs = outputs_per_frame[last_frame_idx]
            num_objects = len(last_outputs["object_ids"])
            if num_objects > 0:
                # Move outputs to CPU for comparison
                object_ids = (
                    last_outputs["object_ids"].cpu()
                    if isinstance(last_outputs["object_ids"], torch.Tensor)
                    else torch.tensor(last_outputs["object_ids"])
                )
                scores = (
                    last_outputs["scores"].cpu()
                    if isinstance(last_outputs["scores"], torch.Tensor)
                    else torch.tensor(last_outputs["scores"])
                )
                boxes = (
                    last_outputs["boxes"].cpu()
                    if isinstance(last_outputs["boxes"], torch.Tensor)
                    else torch.tensor(last_outputs["boxes"])
                )
                masks = (
                    last_outputs["masks"].cpu()
                    if isinstance(last_outputs["masks"], torch.Tensor)
                    else torch.tensor(last_outputs["masks"])
                )

                torch.testing.assert_close(
                    object_ids,
                    torch.tensor([0, 1], dtype=torch.int64),
                )
                torch.testing.assert_close(
                    scores,
                    torch.tensor([0.968647837638855, 0.9736108779907227], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    boxes[0],
                    torch.tensor([157.0, 116.0, 295.0, 382.0], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    masks[0, :3, :3].float(),
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )

        # Check raw model_outputs mask values for last frame
        if len(model_outputs_per_frame) > 1:
            last_frame_idx = max(model_outputs_per_frame.keys())
            last_model_outputs = model_outputs_per_frame[last_frame_idx]
            num_objects = len(last_model_outputs.object_ids)
            if num_objects > 0:
                # Check raw mask from model_outputs (low-resolution, before post-processing)
                first_obj_id = last_model_outputs.object_ids[0]
                raw_mask = last_model_outputs.obj_id_to_mask[first_obj_id].cpu()
                torch.testing.assert_close(
                    raw_mask[:1, :3, :3].float(),
                    torch.tensor(
                        [
                            [
                                [-23.023313522338867, -27.02887535095215, -22.29985237121582],
                                [-24.373233795166016, -31.428438186645508, -24.268810272216797],
                                [-24.550016403198242, -32.607383728027344, -26.500947952270508],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                    atol=5e-3,  # Higher tolerance for raw logits
                    rtol=5e-3,
                )

    def test_inference_video_streaming_with_text_prompt(self):
        raw_video = prepare_video()

        # Initialize session for streaming (no video provided)
        inference_session = self.processor.init_video_session(
            inference_device=torch_device,
            processing_device="cpu",
            video_storage_device="cpu",
        )

        # Add text prompt
        text = "person"
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=text,
        )

        # Process frames one by one (streaming mode)
        outputs_per_frame = {}
        model_outputs_per_frame = {}
        max_frame_num_to_track = 3
        for frame_idx, frame in enumerate(raw_video):
            if frame_idx >= max_frame_num_to_track:
                break

            # Process frame using processor
            inputs = self.processor(images=frame, device=torch_device, return_tensors="pt")

            # Process frame using streaming inference
            model_outputs = self.video_model(
                inference_session=inference_session,
                frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
                reverse=False,
            )

            # Post-process outputs with original_sizes for proper resolution handling
            processed_outputs = self.processor.postprocess_outputs(
                inference_session,
                model_outputs,
                original_sizes=inputs.original_sizes,  # Required for streaming inference
            )
            outputs_per_frame[frame_idx] = processed_outputs
            model_outputs_per_frame[frame_idx] = model_outputs

        # Check we processed the expected number of frames
        self.assertEqual(len(outputs_per_frame), max_frame_num_to_track)

        # Check output structure for each frame
        for frame_idx, processed_outputs in outputs_per_frame.items():
            self.assertIn("object_ids", processed_outputs)
            self.assertIn("scores", processed_outputs)
            self.assertIn("boxes", processed_outputs)
            self.assertIn("masks", processed_outputs)

            num_objects = len(processed_outputs["object_ids"])
            if num_objects > 0:
                self.assertEqual(processed_outputs["scores"].shape, (num_objects,))
                self.assertEqual(processed_outputs["boxes"].shape, (num_objects, 4))
                # For streaming, masks should be at original frame resolution
                H_orig, W_orig = raw_video[frame_idx].shape[0], raw_video[frame_idx].shape[1]
                self.assertEqual(processed_outputs["masks"].shape, (num_objects, H_orig, W_orig))
                # Check boxes are in XYXY format (absolute coordinates)
                boxes = processed_outputs["boxes"]
                self.assertTrue(torch.all(boxes[:, 2] >= boxes[:, 0]))  # x2 >= x1
                self.assertTrue(torch.all(boxes[:, 3] >= boxes[:, 1]))  # y2 >= y1

        # Check numeric values for first frame
        if len(outputs_per_frame) > 0:
            first_frame_idx = min(outputs_per_frame.keys())
            first_outputs = outputs_per_frame[first_frame_idx]
            num_objects = len(first_outputs["object_ids"])
            if num_objects > 0:
                # Move outputs to CPU for comparison (postprocess_outputs may return CPU tensors)
                object_ids = (
                    first_outputs["object_ids"].cpu()
                    if isinstance(first_outputs["object_ids"], torch.Tensor)
                    else torch.tensor(first_outputs["object_ids"])
                )
                scores = (
                    first_outputs["scores"].cpu()
                    if isinstance(first_outputs["scores"], torch.Tensor)
                    else torch.tensor(first_outputs["scores"])
                )
                boxes = (
                    first_outputs["boxes"].cpu()
                    if isinstance(first_outputs["boxes"], torch.Tensor)
                    else torch.tensor(first_outputs["boxes"])
                )
                masks = (
                    first_outputs["masks"].cpu()
                    if isinstance(first_outputs["masks"], torch.Tensor)
                    else torch.tensor(first_outputs["masks"])
                )

                torch.testing.assert_close(
                    object_ids,
                    torch.tensor([0, 1], dtype=torch.int64),
                )
                torch.testing.assert_close(
                    scores,
                    torch.tensor([0.9683944582939148, 0.9740181565284729], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    boxes[0],
                    torch.tensor([146.0, 135.0, 291.0, 404.0], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    masks[0, :3, :3].float(),
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )

        # Check raw model_outputs mask values for first frame
        if len(model_outputs_per_frame) > 0:
            first_frame_idx = min(model_outputs_per_frame.keys())
            first_model_outputs = model_outputs_per_frame[first_frame_idx]
            num_objects = len(first_model_outputs.object_ids)
            if num_objects > 0:
                # Check raw mask from model_outputs (low-resolution, before post-processing)
                first_obj_id = first_model_outputs.object_ids[0]
                raw_mask = first_model_outputs.obj_id_to_mask[first_obj_id].cpu()
                torch.testing.assert_close(
                    raw_mask[:1, :3, :3].float(),
                    torch.tensor(
                        [
                            [
                                [-2.987567901611328, -5.944897651672363, -7.973854064941406],
                                [-7.017378330230713, -10.088018417358398, -11.089308738708496],
                                [-8.274458885192871, -9.851463317871094, -10.428947448730469],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                    atol=5e-3,  # Higher tolerance for raw logits
                    rtol=5e-3,
                )

        # Check numeric values for last frame (to verify propagation consistency)
        if len(outputs_per_frame) > 1:
            last_frame_idx = max(outputs_per_frame.keys())
            last_outputs = outputs_per_frame[last_frame_idx]
            num_objects = len(last_outputs["object_ids"])
            if num_objects > 0:
                # Move outputs to CPU for comparison
                object_ids = (
                    last_outputs["object_ids"].cpu()
                    if isinstance(last_outputs["object_ids"], torch.Tensor)
                    else torch.tensor(last_outputs["object_ids"])
                )
                scores = (
                    last_outputs["scores"].cpu()
                    if isinstance(last_outputs["scores"], torch.Tensor)
                    else torch.tensor(last_outputs["scores"])
                )
                boxes = (
                    last_outputs["boxes"].cpu()
                    if isinstance(last_outputs["boxes"], torch.Tensor)
                    else torch.tensor(last_outputs["boxes"])
                )
                masks = (
                    last_outputs["masks"].cpu()
                    if isinstance(last_outputs["masks"], torch.Tensor)
                    else torch.tensor(last_outputs["masks"])
                )

                torch.testing.assert_close(
                    object_ids,
                    torch.tensor([0, 1], dtype=torch.int64),
                )
                torch.testing.assert_close(
                    scores,
                    torch.tensor([0.9683944582939148, 0.9740181565284729], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    boxes[0],
                    torch.tensor([154.0, 117.0, 294.0, 395.0], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    masks[0, :3, :3].float(),
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                )

        # Check raw model_outputs mask values for last frame
        if len(model_outputs_per_frame) > 1:
            last_frame_idx = max(model_outputs_per_frame.keys())
            last_model_outputs = model_outputs_per_frame[last_frame_idx]
            num_objects = len(last_model_outputs.object_ids)
            if num_objects > 0:
                # Check raw mask from model_outputs (low-resolution, before post-processing)
                first_obj_id = last_model_outputs.object_ids[0]
                raw_mask = last_model_outputs.obj_id_to_mask[first_obj_id].cpu()
                torch.testing.assert_close(
                    raw_mask[:1, :3, :3].float(),
                    torch.tensor(
                        [
                            [
                                [-23.935535430908203, -27.967025756835938, -23.519914627075195],
                                [-25.742399215698242, -32.65046310424805, -24.71213150024414],
                                [-25.263212203979492, -33.807132720947266, -27.463823318481445],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                    atol=5e-3,  # Higher tolerance for raw logits
                    rtol=5e-3,
                )

    def test_inference_video_multi_prompt(self):
        """Test multi-prompt tracking - detecting multiple object categories in one pass."""
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(
            video=raw_video,
            inference_device=torch_device,
            processing_device="cpu",
            video_storage_device="cpu",
        )

        # Add multiple text prompts
        prompts = ["person", "bed"]
        self.processor.add_text_prompt(
            inference_session=inference_session,
            text=prompts,
        )

        # Propagate through video frames
        outputs_per_frame = {}
        for model_outputs in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=3,
        ):
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs

        # Check we processed the expected number of frames
        self.assertGreaterEqual(len(outputs_per_frame), 1)
        self.assertLessEqual(len(outputs_per_frame), 4)

        # Check output structure for each frame
        for processed_outputs in outputs_per_frame.values():
            self.assertIn("object_ids", processed_outputs)
            self.assertIn("scores", processed_outputs)
            self.assertIn("boxes", processed_outputs)
            self.assertIn("masks", processed_outputs)
            self.assertIn("prompt_to_obj_ids", processed_outputs)  # Multi-prompt specific

            # Check prompt_to_obj_ids structure
            prompt_to_obj_ids = processed_outputs["prompt_to_obj_ids"]
            self.assertIsInstance(prompt_to_obj_ids, dict)
            for prompt, obj_ids in prompt_to_obj_ids.items():
                self.assertIsInstance(prompt, str)
                self.assertIsInstance(obj_ids, list)
                # Each object ID should be in the main object_ids list
                for obj_id in obj_ids:
                    self.assertIn(obj_id, processed_outputs["object_ids"].tolist())

        # Check that we detected objects from multiple prompts
        first_frame_outputs = outputs_per_frame[min(outputs_per_frame.keys())]
        prompt_to_obj_ids = first_frame_outputs["prompt_to_obj_ids"]

        # Should have at least one prompt with detections
        self.assertGreater(len(prompt_to_obj_ids), 0)

        # All prompts in prompt_to_obj_ids should be from our original prompts
        for prompt in prompt_to_obj_ids.keys():
            self.assertIn(prompt, prompts)

    def test_custom_image_size(self):
        """Test that custom image size can be set and propagates correctly to detector and tracker configs."""
        from transformers import Sam3VideoConfig

        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        config.image_size = 560

        self.assertEqual(config.image_size, 560)
        self.assertEqual(config.detector_config.image_size, 560)
        self.assertEqual(config.tracker_config.image_size, 560)
        self.assertEqual(config.detector_config.vision_config.image_size, 560)
        self.assertEqual(config.detector_config.vision_config.backbone_config.image_size, 560)

        model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(torch_device).eval()
        self.assertEqual(model.config.image_size, 560)

    def test_inference_with_different_dtypes(self):
        """Test that inference works correctly for float32, bfloat16, and float16 dtypes."""
        raw_video = prepare_video()
        dtypes_to_test = [
            (torch.float32, None),  # float32 is always available
            (torch.bfloat16, is_torch_bf16_available_on_device),
            (torch.float16, is_torch_fp16_available_on_device),
        ]

        for dtype, availability_check in dtypes_to_test:
            with self.subTest(dtype=dtype):
                # Skip if dtype is not available on device
                if availability_check is not None and not availability_check(torch_device):
                    self.skipTest(f"{dtype} not supported on {torch_device}")

                # Load model with specific dtype
                video_model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(torch_device)
                video_model.eval()

                # Initialize inference session
                inference_session = self.processor.init_video_session(
                    video=raw_video,
                    inference_device=torch_device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=dtype,
                )

                # Add text prompt
                text = "person"
                inference_session = self.processor.add_text_prompt(
                    inference_session=inference_session,
                    text=text,
                )

                # Run inference on first frame
                outputs_per_frame = {}
                model_outputs_per_frame = {}
                max_frame_num_to_track = 2
                for model_outputs in video_model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=max_frame_num_to_track,
                ):
                    processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
                    outputs_per_frame[model_outputs.frame_idx] = processed_outputs
                    model_outputs_per_frame[model_outputs.frame_idx] = model_outputs

                    # Verify dtype is maintained in model outputs
                    if len(model_outputs.object_ids) > 0:
                        first_obj_id = model_outputs.object_ids[0]
                        raw_mask = model_outputs.obj_id_to_mask[first_obj_id]
                        self.assertEqual(raw_mask.dtype, dtype)

                # Verify we processed frames
                self.assertGreaterEqual(len(outputs_per_frame), 1)
                self.assertLessEqual(len(outputs_per_frame), max_frame_num_to_track + 1)
