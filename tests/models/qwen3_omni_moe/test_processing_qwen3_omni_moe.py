# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
import unittest

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import Qwen3OmniMoeProcessor
from transformers.testing_utils import (
    require_librosa,
    require_torch,
    require_torchaudio,
    require_torchcodec,
    require_torchvision,
    require_vision,
)
from transformers.utils import is_torch_available

from ...test_processing_common import MODALITY_TEST_SPECS, ProcessorTesterMixin


if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchaudio
@require_torchvision
class Qwen3OmniMoeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3OmniMoeProcessor
    # Tiny processor created with make_tiny_processor.py from "Qwen/Qwen2.5-Omni-7B"
    tiny_model_id = "hf-internal-testing/tiny-processor-qwen3_omni_moe"

    videos_unstructured_max_length = 785
    videos_text_kwargs_max_length = 785
    videos_text_kwargs_override_max_length = 785
    audio_unstructured_max_length = 150

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class.from_pretrained(
            cls.tiny_model_id, size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}
        )

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        return video_processor_class.from_pretrained(
            cls.tiny_model_id, size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}
        )

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")
        # chunk_length=30s instead of the default 300s reduces input_features from
        # (batch, 128, 30000) to (batch, 128, 3000), cutting peak memory per audio test.
        return feature_extractor_class.from_pretrained(cls.tiny_model_id, chunk_length=30)

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 1440},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 2160},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 4320},
        ]

    def prepare_audio_inputs(self, batch_size: int = 3):
        """This function prepares a list of numpy audios."""
        audio_inputs = [np.random.rand(160000) * 2 - 1] * batch_size
        return audio_inputs

    @parameterized.expand(
        [
            ("text",),
            ("images",),
            ("videos",),
            ("audio",),
        ]
    )
    def test_subprocessor_defaults(self, modality):
        """
        Tests that sub-processor is called correctly when passing each modality input to the processor.
        This test verifies that processor(single_modality_data) produces the same output as subprocessor(single_modality_data).
        """
        # override to pop processor-only keys from `merged_kwargs`
        attributes = self.processor_class.get_attributes()
        component_key = self.get_subprocessor_name(modality, attributes)

        parameterized_config = MODALITY_TEST_SPECS[modality]
        subprocessor = self.get_component(component_key)

        # Get all other required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components, **self.prepare_processor_dict())
        modality_input = self._prepare_modality_input(modality)

        # merge processor defaults when calling a subprocessor
        kwargs = parameterized_config["call_time_kwargs"]
        kwargs["return_tensors"] = "pt"
        merged_kwargs = processor._merge_kwargs(
            processor.valid_processor_kwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs if hasattr(processor, "tokenizer") else {},
            **kwargs,
        )
        kwargs = merged_kwargs[f"{modality}_kwargs"]
        kwargs.pop("seconds_per_chunk", None)  # pop, used only in `processor.__call__`
        kwargs.pop("use_audio_in_video", None)
        kwargs.pop("position_id_per_seconds", None)

        input_subproc = subprocessor(modality_input, **kwargs)
        try:
            input_processor = processor(**{modality: modality_input, **kwargs})
        except Exception:
            input_processor = {}

        # Verify outputs match
        for key in input_subproc:
            if input_processor and key in processor.model_input_names:
                torch.testing.assert_close(input_subproc[key], input_processor[key])

    def test_post_process_multimodal_output_batched_audio(self):
        # Batched generation returns one waveform per sample, each trimmed to its own length.
        processor = self.processor_class.__new__(self.processor_class)
        generated_outputs = (
            torch.ones((2, 3), dtype=torch.long),
            [torch.arange(6, dtype=torch.float32), torch.arange(6, 10, dtype=torch.float32)],
        )

        audio_outputs = processor.post_process_multimodal_output(generated_outputs, generation_mode="audio")

        self.assertEqual(len(audio_outputs), 2)
        self.assertTrue(np.array_equal(audio_outputs[0], np.arange(6, dtype=np.float32)))
        self.assertTrue(np.array_equal(audio_outputs[1], np.arange(6, 10, dtype=np.float32)))

    def test_post_process_multimodal_output_single_audio(self):
        # Single-sample generation returns a lone `[1, 1, num_samples]` tensor, not a list.
        processor = self.processor_class.__new__(self.processor_class)
        generated_outputs = (
            torch.ones((1, 3), dtype=torch.long),
            torch.arange(6, dtype=torch.float32).reshape(1, 1, 6),
        )

        audio_outputs = processor.post_process_multimodal_output(generated_outputs, generation_mode="audio")

        self.assertEqual(len(audio_outputs), 1)
        self.assertTrue(np.array_equal(audio_outputs[0], np.arange(6, dtype=np.float32)))

    @require_librosa
    @require_torchcodec
    def test_chat_template_audio_from_video(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest(f"{self.processor_class} does not support video inputs")

        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        video_file_path = hf_hub_download(
            repo_id="hf-internal-testing/test-videos", filename="sample_demo_1_320x240.mp4", repo_type="dataset"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_file_path},
                    {"type": "text", "text": "Which of these animals is making the sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a cow."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me all about this animal."},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)  # batch size=1

        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            load_audio_from_video=True,
        )
        self.assertTrue(self.audio_input_name in out_dict)
        self.assertTrue(self.videos_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict["attention_mask"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict[self.audio_input_name]), 1)  # 1 audio in the conversation
        self.assertEqual(len(out_dict[self.videos_input_name]), 10800)  # 1 video in the conversation
