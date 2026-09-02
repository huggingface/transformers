# Copyright 2025 HuggingFace Inc.
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

import json
import tempfile
import unittest

import numpy as np

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import get_image_size
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import smart_resize

    if is_torchvision_available():
        from transformers import Qwen2VLVideoProcessor


class Qwen2VLVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
        temporal_patch_size=2,
        patch_size=14,
        min_pixels=20 * 20,
        max_pixels=100 * 100,
        merge_size=2,
    ):
        size = size if size is not None else {"shortest_edge": 400, "longest_edge": 10000}
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "merge_size": self.merge_size,
        }

    @require_vision
    def expected_output_video_shape(self, videos, num_frames=None):
        num_frames = num_frames if num_frames is not None else self.num_frames
        grid_t = num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for video in videos:
            if isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            height, width = get_image_size(video)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            seq_len += grid_t * grid_h * grid_w
        return [seq_len, hidden_dim]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        videos = prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )
        return videos


@require_torch
@require_vision
class Qwen2VLVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Qwen2VLVideoProcessor if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Qwen2VLVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processing, "do_resize"))
        self.assertTrue(hasattr(video_processing, "size"))
        self.assertTrue(hasattr(video_processing, "do_normalize"))
        self.assertTrue(hasattr(video_processing, "image_mean"))
        self.assertTrue(hasattr(video_processing, "image_std"))
        self.assertTrue(hasattr(video_processing, "do_convert_rgb"))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"shortest_edge": 400, "longest_edge": 10000})

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"shortest_edge": 100, "longest_edge": 200}
        )
        # min_pixels and max_pixels take precedence over size, like in the image processor.
        self.assertEqual(video_processor.size, {"shortest_edge": 400, "longest_edge": 10000})

        processor_dict = self.video_processor_dict.copy()
        processor_dict.pop("min_pixels")
        processor_dict.pop("max_pixels")
        video_processor = self.fast_video_processing_class.from_dict(
            processor_dict, size={"shortest_edge": 100, "longest_edge": 200}
        )
        self.assertEqual(video_processor.size, {"shortest_edge": 100, "longest_edge": 200})

    def test_video_processor_to_json_string(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            obj = json.loads(video_processor.to_json_string())
            for key, value in self.video_processor_dict.items():
                if key not in ["min_pixels", "max_pixels"]:
                    self.assertEqual(obj[key], value)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pil"
            )

            # Each video is a list of PIL Images
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random numpy tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random PyTorch tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )

            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                list(encoded_videos.shape),
                expected_output_video_shape,
            )

    def test_nested_input(self):
        """Tests that the processor can work with nested list where each video is a list of arrays"""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            # Test not batched input
            video_inputs_nested = [list(video) for video in video_inputs]
            encoded_videos = video_processing(video_inputs_nested[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs_nested, return_tensors="pt")[self.input_name]
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    @unittest.skip("Skip for now, the test needs adjustment fo Qwen2VL")
    def test_call_numpy_4_channels(self):
        for video_processing_class in self.video_processor_list:
            # Test that can process videos which have an arbitrary number of channels
            # Initialize video_processing
            video_processor = video_processing_class(**self.video_processor_dict)

            # create random numpy tensors
            self.video_processor_tester.num_channels = 4
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            # Test not batched input
            encoded_videos = video_processor(
                video_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processor(
                video_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)

            prev_num_frames = self.video_processor_tester.num_frames
            self.video_processor_tester.num_frames = 8
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )

            # Force set sampling to False. No sampling is expected even when `num_frames` exists
            video_processing.do_sample_frames = False

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=3)[self.input_name]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=3)[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            expected_output_video_shape_batched = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)
            self.assertListEqual(list(encoded_videos_batched.shape), expected_output_video_shape_batched)

            # Set sampling to True. Video frames should be sampled with `num_frames` in the output
            video_processing.do_sample_frames = True

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=4)[self.input_name]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=4)[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(
                [video_inputs[0]], num_frames=4
            )
            expected_output_video_shape_batched = self.video_processor_tester.expected_output_video_shape(
                video_inputs, num_frames=4
            )
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)
            self.assertListEqual(list(encoded_videos_batched.shape), expected_output_video_shape_batched)

            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]
            batched_metadata = metadata * len(video_inputs)
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)[
                self.input_name
            ]
            encoded_videos_batched = video_processing(
                video_inputs, return_tensors="pt", fps=3, video_metadata=batched_metadata
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(
                [video_inputs[0]], num_frames=6
            )
            expected_output_video_shape_batched = self.video_processor_tester.expected_output_video_shape(
                video_inputs, num_frames=6
            )
            self.assertListEqual(list(encoded_videos.shape), expected_output_video_shape)
            self.assertListEqual(list(encoded_videos_batched.shape), expected_output_video_shape_batched)

            # We should raise error when asked to sample more frames than there are in input video
            with self.assertRaises(ValueError):
                encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=10)[self.input_name]
                encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=10)[
                    self.input_name
                ]

            # Assign back the actual num frames in tester
            self.video_processor_tester.num_frames = prev_num_frames

    def test_num_frames_equal_temporal_patch_size_plus_two(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_processor_dict["size"] = {"longest_edge": 5 * 28 * 28, "shortest_edge": 28 * 28}
            video_processor_dict["do_sample_frames"] = False
            temporal_patch_size = 3
            video_processor_dict["temporal_patch_size"] = temporal_patch_size
            video_processing = video_processing_class(**video_processor_dict)

            n, w, h = 5, 28, 28
            video_inputs = [(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)) for _ in range(n)]

            video_processed = video_processing(video_inputs, return_tensors="pt")
            encoded_videos = video_processed[self.input_name]
            self.assertEqual(list(encoded_videos.shape), [8, temporal_patch_size * 3 * 14 * 14])

            video_grid_thw = video_processed["video_grid_thw"]
            self.assertEqual(video_grid_thw.tolist(), [[2, 2, 2]])

    def test_bc_min_max_pixels(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                video_processing.save_pretrained(tmpdirname)
                video_processing_loaded = video_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=True,
                return_tensors="torch",
            )
            processed = video_processing_loaded(video_inputs, return_tensors="pt")
            expected_output_video_shape = [320, 1176]
            self.assertListEqual(list(processed.pixel_values_videos.shape), expected_output_video_shape)

    def _process_frames(self, video_processing_class, num_frames, size, frame_size=256, **processor_kwargs):
        video_processor_dict = self.video_processor_dict.copy()
        video_processor_dict.pop("min_pixels", None)
        video_processor_dict.pop("max_pixels", None)
        video_processor_dict["size"] = size
        video_processor_dict["do_sample_frames"] = False
        video_processor_dict.update(processor_kwargs)
        video_processing = video_processing_class(**video_processor_dict)
        video = [np.random.randint(0, 256, (frame_size, frame_size, 3), dtype=np.uint8) for _ in range(num_frames)]
        return video_processing(video, return_tensors="pt")[self.input_name]

    def _expected_capped_seq_len(self, num_frames, frame_size, size, total_seq_len):
        factor = 28
        total_pixels = int(total_seq_len * factor * factor * 0.9)
        max_pixels = max(min(size["longest_edge"], total_pixels * 2 // num_frames), int(size["shortest_edge"] * 1.05))
        expected_height, expected_width = smart_resize(
            frame_size,
            frame_size,
            factor=factor,
            min_pixels=size["shortest_edge"],
            max_pixels=max_pixels,
        )
        return (num_frames // 2) * (expected_height // 14) * (expected_width // 14)

    def test_cap_pixels_per_frame_bounds_dense_videos(self):
        # With the total budget binding, each frame is held to the budget's even share instead of
        # the per-frame `longest_edge` cap. The budget is shrunk via `max_video_tokens` so the
        # test does not need hundreds of frames to make the bound bind.
        size = {"longest_edge": 768 * 28 * 28 * 100, "shortest_edge": 400}
        for video_processing_class in self.video_processor_list:
            uncapped = self._process_frames(video_processing_class, 8, size)
            capped = self._process_frames(
                video_processing_class, 8, size, cap_pixels_per_frame=True, max_video_tokens=128
            )
            self.assertEqual(capped.shape[0], self._expected_capped_seq_len(8, 256, size, total_seq_len=128))
            self.assertLess(capped.shape[0], uncapped.shape[0])

    def test_cap_pixels_per_frame_floors_at_min_pixels(self):
        # When the budget's share drops below `1.05 * shortest_edge`, frames keep a usable
        # resolution instead of collapsing further.
        size = {"longest_edge": 768 * 28 * 28 * 100, "shortest_edge": 40000}
        for video_processing_class in self.video_processor_list:
            capped = self._process_frames(
                video_processing_class, 8, size, cap_pixels_per_frame=True, max_video_tokens=128
            )
            self.assertEqual(capped.shape[0], self._expected_capped_seq_len(8, 256, size, total_seq_len=128))

    def test_cap_pixels_per_frame_noop_when_not_binding(self):
        # At the real budget a short clip never hits the total bound: capped and uncapped agree.
        size = {"longest_edge": 768 * 28 * 28 * 100, "shortest_edge": 400}
        for video_processing_class in self.video_processor_list:
            uncapped = self._process_frames(video_processing_class, 8, size)
            capped = self._process_frames(video_processing_class, 8, size, cap_pixels_per_frame=True)
            self.assertEqual(list(capped.shape), list(uncapped.shape))

    def test_cap_pixels_per_frame_call_time_override(self):
        size = {"longest_edge": 768 * 28 * 28 * 100, "shortest_edge": 400}
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_processor_dict.pop("min_pixels", None)
            video_processor_dict.pop("max_pixels", None)
            video_processor_dict["size"] = size
            video_processor_dict["do_sample_frames"] = False
            video_processor_dict["max_video_tokens"] = 128
            video_processing = video_processing_class(**video_processor_dict)
            video = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(8)]

            default_out = video_processing(video, return_tensors="pt")[self.input_name]
            capped_out = video_processing(video, return_tensors="pt", cap_pixels_per_frame=True)[self.input_name]

            self.assertEqual(capped_out.shape[0], self._expected_capped_seq_len(8, 256, size, total_seq_len=128))
            self.assertLess(capped_out.shape[0], default_out.shape[0])

    def test_cap_pixels_per_frame_unset_warns_and_false_is_silent(self):
        from transformers.utils import logging as transformers_logging

        logger = transformers_logging.get_logger("transformers.models.qwen2_vl.video_processing_qwen2_vl")
        size = {"longest_edge": 768 * 28 * 28, "shortest_edge": 400}
        for video_processing_class in self.video_processor_list:
            logger.warning_once.cache_clear()
            with self.assertLogs(logger.name, level="WARNING") as logs:
                self._process_frames(video_processing_class, 2, size)
            self.assertTrue(any("cap_pixels_per_frame" in line for line in logs.output))

            logger.warning_once.cache_clear()
            with self.assertNoLogs(logger.name, level="WARNING"):
                self._process_frames(video_processing_class, 2, size, cap_pixels_per_frame=False)
