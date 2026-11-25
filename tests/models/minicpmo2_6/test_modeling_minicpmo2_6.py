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
"""Testing suite for the PyTorch MiniCPM-o 2.6 model."""

import unittest

from transformers import AutoProcessor, MiniCPMO2_6Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMO2_6ForConditionalGeneration, MiniCPMO2_6Model


# Shared by both testers, which build the full model (vision tower, audio encoder and a tiny Qwen2)
SMALL_CONFIG = {
    "batch_size": 2,
    "image_token_id": 100,
    "audio_token_id": 101,
    "image_size": 32,
    "patch_size": 8,  # each test image is one 4x4-patch slice
    "resampler_n_latents": 4,  # ... which the resampler turns into 4 tokens
    "resampler_n_heads": 4,
    "num_image_tokens": 4,
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 37,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "vision_hidden_act": "gelu_pytorch_tanh",
    "vision_intermediate_size": 37,
    "audio_hidden_size": 32,
    "audio_num_hidden_layers": 2,
    "audio_num_attention_heads": 2,
    "audio_intermediate_size": 37,
    "audio_max_source_positions": 32,
    "audio_chunk_size": 8,  # 64 mel frames -> 32 encoder frames in 4 attention chunks -> 16 audio tokens
    "feat_seq_length": 64,
    "num_mel_bins": 80,
}


class MiniCPMO2_6Vision2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMO2_6Model
    config_class = MiniCPMO2_6Config
    conditional_generation_class = MiniCPMO2_6ForConditionalGeneration
    small_config_overrides = SMALL_CONFIG

    def create_pixel_values(self, batch_size: int | None = None):
        # One slice per image, as a single row of 4 x 4 = 16 patches: (num_slices, channels, patch_size, 16 * patch_size)
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor([batch_size, self.num_channels, self.patch_size, num_patches * self.patch_size])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.image_size // self.patch_size
        # Every other slice is a smaller (side - 1) x side grid padded to side x side, so attention tests cover padding.
        pixel_attention_mask = torch.ones(batch_size, side * side, dtype=torch.bool, device=torch_device)
        pixel_attention_mask[1::2, (side - 1) * side :] = False
        rows, cols = torch.meshgrid(
            torch.arange(side, device=torch_device), torch.arange(side, device=torch_device), indexing="ij"
        )
        grid = torch.stack([rows, cols], dim=-1).reshape(1, side * side, 2).repeat(batch_size, 1, 1)
        patch_position_ids = grid * pixel_attention_mask[..., None]
        return {"pixel_attention_mask": pixel_attention_mask, "patch_position_ids": patch_position_ids}


class MiniCPMO2_6Audio2TextModelTester(ALMModelTester):
    base_model_class = MiniCPMO2_6Model
    config_class = MiniCPMO2_6Config
    conditional_generation_class = MiniCPMO2_6ForConditionalGeneration
    audio_mask_key = "input_features_mask"
    small_config_overrides = SMALL_CONFIG

    def create_audio_mask(self, batch_size: int | None = None):
        # Windows are right-padded
        return super().create_audio_mask(batch_size).sort(dim=-1, descending=True).values

    def get_audio_embeds_mask(self, audio_mask):
        # The encoder's strided conv and the projector's pooling each halve the frames
        lengths = ((audio_mask.sum(-1) - 1) // 2 + 1 - 2) // 2 + 1
        positions = torch.arange(int(lengths.max()), device=audio_mask.device)[None, :]
        return (positions < lengths[:, None]).long()


class MiniCPMO2_6ModelTestMixin:
    """Overrides shared by both test classes, including `get_*_features` inputs for both modalities."""

    # each class feeds a single modality, so the other tower gets no gradient
    test_all_params_have_gradient = False

    @unittest.skip(reason="Flash-only SDPA takes neither the padded slices' mask nor the audio encoder's chunked mask")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_reverse_loading_mapping(self):
        # As in llava, the base model has no `model.` prefix to reverse the `^llm\.model\.` rename onto
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = MiniCPMO2_6Vision2TextModelTester(self).prepare_config_and_inputs_for_common()
        return config, {
            key: inputs_dict[key] for key in ("pixel_values", "pixel_attention_mask", "patch_position_ids")
        }

    def _audio_features_prepare_config_and_inputs(self):
        config, inputs_dict = MiniCPMO2_6Audio2TextModelTester(self).prepare_config_and_inputs_for_common()
        return config, {key: inputs_dict[key] for key in ("input_features", "input_features_mask")}


@require_torch
class MiniCPMO2_6Vision2TextModelTest(MiniCPMO2_6ModelTestMixin, VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMO2_6Vision2TextModelTester


@require_torch
class MiniCPMO2_6Audio2TextModelTest(MiniCPMO2_6ModelTestMixin, ALMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMO2_6Audio2TextModelTester

    @unittest.skip(reason="The audio tester has no images")
    def test_multimodal_outputs_are_returned(self):
        pass

    @unittest.skip(reason="The audio tester has no images, the only modality that is pre-encoded")
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass

    @unittest.skip(reason="Checks the first checkpointing layer, a vision layer, which the audio tester doesn't run")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Checks the first checkpointing layer, a vision layer, which the audio tester doesn't run")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Checks the first checkpointing layer, a vision layer, which the audio tester doesn't run")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@slow
@require_torch_accelerator
class MiniCPMO2_6IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM-o-2_6"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_audio_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMO2_6ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
                        ),
                    },
                    {
                        "type": "text",
                        "text": "Please listen to the audio snippet carefully and transcribe the content.",
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = processor.post_process_image_text_to_text(output[:, inputs["input_ids"].shape[1] :])[0]
        expected_texts = Expectations({("cuda", None): ""})
        self.assertEqual(decoded, expected_texts.get_expectation())
