# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import Ovis2_5Processor
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, require_vision
from transformers.utils import is_torchcodec_available

from ...test_processing_common import ProcessorTesterMixin


@require_vision
@require_torch
@require_torchvision
class Ovis2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ovis2_5Processor
    model_id = "ovis-community/Ovis2.5-2B"
    images_text_kwargs_max_length = 256
    images_unstructured_max_length = 256
    videos_text_kwargs_max_length = 2048
    videos_unstructured_max_length = 2048

    def _test_apply_chat_template(self, modality, batch_size, return_tensors, input_name, processor_name, input_data):
        if modality == "video":
            if batch_size > 1:
                self.skipTest("Ovis2.5 accepts exactly one video")
            if isinstance(input_data[0], str) and not is_torchcodec_available():
                self.skipTest("Video URL decoding requires Torchcodec")
        super()._test_apply_chat_template(modality, batch_size, return_tensors, input_name, processor_name, input_data)

    @property
    def video_sampling_expectations(self):
        expectations = super().video_sampling_expectations
        for expectation in expectations:
            expectation["expected_dim"] = 0
            # Each 240x320 fixture frame resizes to 416x544, yielding 26x34 patches.
            expectation["output_length"] *= 884
        return expectations

    @require_torchcodec
    def test_apply_chat_template_video_frame_sampling(self):
        super().test_apply_chat_template_video_frame_sampling()

    def test_model_input_names(self):
        # The shared test combines images and videos, which Ovis2.5 handles separately.
        processor = self.get_processor()
        input_names = set()
        for modality in ("images", "videos"):
            inputs = processor(
                text=self.prepare_text_inputs(modalities=modality),
                **{modality: getattr(self, f"prepare_{modality}_inputs")()},
                do_sample_frames=False,
                return_tensors="pt",
            )
            input_names.update(inputs)
        self.assertSetEqual(set(processor.model_input_names), input_names)

    def test_processor_with_multiple_inputs(self):
        # The shared test expects mixed visual modalities to succeed.
        with self.assertRaisesRegex(ValueError, "one visual modality at a time"):
            super().test_processor_with_multiple_inputs()

    def _test_unstructured_kwargs_batched(self, modality):
        # The shared video kwargs test supplies two videos; Ovis2.5 accepts one.
        if modality == "videos":
            with self.assertRaisesRegex(ValueError, "exactly one video"):
                super()._test_unstructured_kwargs_batched(modality)
        else:
            super()._test_unstructured_kwargs_batched(modality)

    def test_get_num_multimodal_tokens_matches_processor_call_video(self):
        # The shared test supplies multiple videos; check the supported single-video path.
        processor = self.get_processor()
        video = self.prepare_videos_inputs()[:7]
        inputs = processor(
            text=f"{processor.video_token} lower newer",
            videos=video,
            do_sample_frames=False,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        video_token_counts = (inputs.mm_token_type_ids == 2).sum(-1).tolist()
        video_sizes = [(len(video), *video.shape[-2:])]
        helper_counts = processor._get_num_multimodal_tokens(video_sizes=video_sizes)["num_video_tokens"]
        self.assertListEqual(video_token_counts, helper_counts)

    @unittest.skip("Ovis2.5 processes images and videos separately")
    def test_replacement_offsets(self):
        pass

    @unittest.skip("Ovis2.5 processes images and videos separately")
    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        pass

    def test_processor_text_has_no_visual(self):
        # The shared mixed batch also includes videos, which cannot accompany images.
        processor = self.get_processor()
        image = self.prepare_images_inputs()
        inputs = processor(
            text=["lower newer", f"{processor.image_token} lower newer", f"{processor.image_token} lower newer"],
            images=[[], [image], [image.copy()]],
            padding=True,
            return_tensors="pt",
        )

        self.assertEqual(inputs.input_ids.shape[0], 3)
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)
