# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import hashlib
import tempfile
import unittest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import AutoVideoProcessor, Glm5NextVideoProcessor
    from transformers.video_utils import VideoMetadata


@require_torch
@require_vision
class Glm5NextVideoProcessingTest(unittest.TestCase):
    def get_processor_kwargs(self, **overrides):
        kwargs = {
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "patch_expand_factor": 2,
            "min_image_tokens": 1,
            "max_image_tokens": 64,
            "fps_interval": 1,
            "max_frame_count_dynamic": 16,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        return kwargs | overrides

    def test_bit_exact_reference_output(self):
        processor = Glm5NextVideoProcessor(**self.get_processor_kwargs(do_sample_frames=False))
        video = torch.arange(4 * 3 * 5 * 9, dtype=torch.uint8).reshape(4, 3, 5, 9)
        output = processor(video, return_tensors="pt")
        digest = hashlib.sha256(output.pixel_values_videos.contiguous().numpy().tobytes()).hexdigest()
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 4, 8]])
        self.assertEqual(digest, "a9efa48c76635694a691f715667a9f40c5ddf6bdb12556a653705e1c883476f8")

    def test_sample_frames_reference_cases(self):
        processor = Glm5NextVideoProcessor(**self.get_processor_kwargs())
        cases = [
            (VideoMetadata(total_num_frames=3, fps=1, duration=3), None, None, [0, 1, 2, 2]),
            (VideoMetadata(total_num_frames=20, fps=4, duration=5), None, None, [0, 4, 8, 12, 16, 16]),
            (VideoMetadata(total_num_frames=20, fps=4, duration=5), 2, None, list(range(0, 20, 2))),
            (VideoMetadata(total_num_frames=120, fps=24, duration=5), 3, 8, [0, 17, 34, 51, 68, 85, 102, 119]),
        ]
        for metadata, target_fps, max_frames, expected in cases:
            self.assertEqual(
                processor.sample_frames(metadata, fps=target_fps, max_frames=max_frames).tolist(),
                expected,
            )

        capped = Glm5NextVideoProcessor(**self.get_processor_kwargs(max_duration=5))
        metadata = VideoMetadata(total_num_frames=1000, fps=10, duration=100)
        self.assertEqual(capped.sample_frames(metadata).tolist(), [0, 249, 499, 749, 999, 999])

    def test_new_processor_config_bit_exact(self):
        processor = Glm5NextVideoProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            patch_expand_factor=1,
            min_image_tokens=16,
            max_image_tokens=240000,
            resize_mode="pad",
            fps_interval=2,
            max_frame_count_dynamic=2048,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            do_sample_frames=False,
        )
        video = torch.arange(4 * 3 * 37 * 53, dtype=torch.uint8).reshape(4, 3, 37, 53)
        output = processor(video, return_tensors="pt")
        digest = hashlib.sha256(output.pixel_values_videos.contiguous().numpy().tobytes()).hexdigest()
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 6, 8]])
        self.assertEqual(digest, "dd5878e0f611c4a8200355322f124b9af348649b4fd028905adad4ca8e998fad")

    def test_numpy_input(self):
        processor = Glm5NextVideoProcessor(**self.get_processor_kwargs(do_sample_frames=False))
        video = torch.zeros(4, 3, 5, 9, dtype=torch.uint8).numpy()
        output = processor(video, return_tensors="pt")
        self.assertEqual(tuple(output.pixel_values_videos.shape), (64, 24))
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 4, 8]])

    def test_per_call_resize_overrides(self):
        processor = Glm5NextVideoProcessor(**self.get_processor_kwargs(do_sample_frames=False))
        video = torch.arange(4 * 3 * 37 * 53, dtype=torch.uint8).reshape(4, 3, 37, 53)
        overrides = {"max_image_tokens": 8, "patch_expand_factor": 1, "resize_mode": "resize"}
        actual = processor(video, return_tensors="pt", **overrides)
        expected = Glm5NextVideoProcessor(**self.get_processor_kwargs(do_sample_frames=False, **overrides))(
            video, return_tensors="pt"
        )
        self.assertTrue(torch.equal(actual.pixel_values_videos, expected.pixel_values_videos))
        self.assertTrue(torch.equal(actual.video_grid_thw, expected.video_grid_thw))

    def test_auto_video_processor_round_trip(self):
        processor = Glm5NextVideoProcessor(**self.get_processor_kwargs())
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            loaded = AutoVideoProcessor.from_pretrained(tmpdir)
        self.assertIsInstance(loaded, Glm5NextVideoProcessor)
        self.assertEqual(loaded.to_dict(), processor.to_dict())

    def test_sglang_processor_defaults(self):
        processor = Glm5NextVideoProcessor(
            **self.get_processor_kwargs(
                fps_interval=2,
                max_frame_count_dynamic=2048,
            )
        )
        self.assertEqual(processor.fps_interval, 2)
        self.assertEqual(processor.max_frame_count_dynamic, 2048)
        self.assertEqual(processor.max_duration, 0)

    def test_required_token_bounds(self):
        with self.assertRaisesRegex(ValueError, "min_image_tokens and max_image_tokens"):
            Glm5NextVideoProcessor(
                patch_size=2,
                temporal_patch_size=2,
                merge_size=2,
                patch_expand_factor=2,
                fps_interval=1,
                max_frame_count_dynamic=16,
            )


if __name__ == "__main__":
    unittest.main()
