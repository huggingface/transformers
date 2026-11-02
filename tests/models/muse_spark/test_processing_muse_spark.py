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

import os
import tempfile
import unittest

import torch
from huggingface_hub import download_bucket_files
from safetensors.torch import load_file

from transformers import AutoProcessor, MuseSparkProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_processing_common import ProcessorTesterMixin


@require_vision
@require_torch
class MuseSparkProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MuseSparkProcessor
    model_id = "hf-internal-testing/tiny-muse-spark"
    audio_input_name = "input_values"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(max_image_tokens=40)

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        # `replace_video_token` needs the metadata to write one timestamp per temporal group
        return video_processor_class(max_video_frame_tokens=40, do_sample_frames=False, return_metadata=True)

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 140},
            {"num_frames": None, "fps": 2, "expected_dim": 0, "output_length": 140},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 840},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 840},
            {"expected_dim": 0, "output_length": 840},
        ]

    @unittest.skip("The processor consumes `video_metadata`, so its output cannot be equal to the raw one")
    def test_video_processor_defaults(self):
        pass

    def test_image_boundary_tokens(self):
        processor = self.get_processor()
        images = self.prepare_images_inputs(batch_size=2)
        text = f"{processor.image_token}lower{processor.image_token}upper"

        inputs = processor(text=text, images=images)
        num_tokens = [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in inputs.image_grid_thw]
        expanded_text = (
            processor.image_start_token
            + processor.image_token * num_tokens[0]
            + processor.image_end_token
            + "lower"
            + processor.image_start_token
            + processor.image_token * num_tokens[1]
            + processor.image_end_token
            + "upper"
        )

        self.assertEqual(inputs.input_ids[0], processor.tokenizer(expanded_text).input_ids)
        self.assertEqual(inputs.input_ids[0].count(processor.image_start_token_id), 2)
        self.assertEqual(inputs.input_ids[0].count(processor.image_end_token_id), 2)
        self.assertEqual(inputs.input_ids[0].count(processor.image_token_id), sum(num_tokens))


@require_torch
class MuseSparkProcessingIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Check the audio prompt against the sglang reference.

    reproducer (regenerate the goldens and re-upload with), alongside the exact venv pins in
    ``hf://buckets/tiny-test-models/muse-spark-integration-test/reproducers/envs/``:
        ``hf://buckets/tiny-test-models/muse-spark-integration-test/reproducers/
          MuseSparkProcessingIntegrationTest/code/reproducer_processing_audio.py``
    It runs sglang's own `MuseSpark1xOmniMultimodalProcessor` on the rendered chat template and
    stores its `input_ids` at
    ``hf://buckets/hf-internal-testing/muse-spark-integration-test/<case>/expected_processing.safetensors``.

    No sentinel remapping is needed here (unlike Inkling): both stacks read the same tokenizer, so
    sglang emits the real `<|audio|>` / delimiter ids directly.
    """

    SAMPLES_PER_TOKEN = 1920
    AUDIO_URL = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_medium.wav"
    )
    AUDIO_URL_2 = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_short.wav"
    )
    CURRENT_DATE = "2026-08-31"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "hf-internal-testing/tiny-muse-spark"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)
        cls.bucket = "hf-internal-testing/muse-spark-integration-test"

    def _load_expected(self, case: str) -> dict:
        remote = f"{case}/expected_processing.safetensors"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "expected_processing.safetensors")
            download_bucket_files(self.bucket, files=[(remote, local)])
            return load_file(local)

    def _assert_matches_sglang(self, case: str, messages: list):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            current_date=self.CURRENT_DATE,
        )
        expected = self._load_expected(case)

        input_ids = inputs["input_ids"][0]
        torch.testing.assert_close(input_ids, expected["input_ids"].to(torch.int64), rtol=0, atol=0)

        # the waveform the model will encode must line up with the placeholders sglang emitted
        num_samples = expected["audio_num_samples"]
        audio_id = self.processor.audio_token_id
        expected_tokens = int(sum(-(-int(n) // self.SAMPLES_PER_TOKEN) for n in num_samples))
        self.assertEqual((input_ids == audio_id).sum().item(), expected_tokens)
        self.assertEqual(int(inputs["input_values_mask"].sum()), int(num_samples.sum()))
        self.assertEqual(inputs["input_values"].shape[-1] % self.SAMPLES_PER_TOKEN, 0)

    def test_apply_chat_template_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in this clip?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        self._assert_matches_sglang("audio", messages)

    def test_apply_chat_template_multi_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two clips."},
                    {"type": "audio", "url": self.AUDIO_URL},
                    {"type": "audio", "url": self.AUDIO_URL_2},
                ],
            }
        ]
        self._assert_matches_sglang("multi_audio", messages)

    def test_literal_audio_tag_in_user_text_is_escaped(self):
        """A user typing "<audio>" must not be able to forge a placeholder."""
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "what does <audio> mean?"}]}],
            add_generation_prompt=True,
            tokenize=False,
            current_date=self.CURRENT_DATE,
        )
        self.assertNotIn("<audio>", rendered)
        self.assertIn("< audio>", rendered)
