# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch glmasr model."""

import tempfile
import unittest

import pytest

from transformers import (
    AutoProcessor,
    GlmAsrConfig,
    GlmAsrForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class GlmAsrModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=35,
        feat_seq_length=64,
        text_config={
            "model_type": "llama",
            "intermediate_size": 64,
            "initializer_range": 0.02,
            "hidden_size": 16,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 1,  # can't be the same as the audio token id
        },
        is_training=True,
        audio_config={
            "model_type": "glmasr_encoder",
            "hidden_size": 128,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_mel_bins": 128,
            "max_source_positions": 32,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return GlmAsrConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                self.feat_seq_length,
            ]
        )
        config = self.get_config()
        input_features_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.bool).to(torch_device)
        return config, input_features_values, input_features_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values, input_features_mask = config_and_inputs
        num_audio_tokens_per_batch_idx = 8

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0

        input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features_values,
            "input_features_mask": input_features_mask,
        }
        return config, inputs_dict


@require_torch
class GlmAsrForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `GlmAsrForConditionalGeneration`.
    """

    all_model_classes = (GlmAsrForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"audio-text-to-text": GlmAsrForConditionalGeneration} if is_torch_available() else {}

    _is_composite = True

    def setUp(self):
        self.model_tester = GlmAsrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GlmAsrConfig, has_text_modality=False)

    @unittest.skip(
        reason="This test does not apply to GlmAsr since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported for GlmAsr models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported for GlmAsr models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="GlmAsr tests avoid right-padding equivalence; fusion is in-place.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="GlmAsr has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # GlmAsr is audio+text composite; verify SDPA toggles propagate to submodules.
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                # SDPA (default)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                audio_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == audio_attn)

                # Eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for _, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class GlmAsrForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "zai-org/GLM-ASR-Nano-2512"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single_batch_sub_30(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                    },
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            },
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "Yesterday it was thirty five degrees in Barcelona, but today the temperature will go down to minus twenty degrees."
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_single_batch_over_30(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
                    },
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            },
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_batched(self):
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                        },
                        {"type": "text", "text": "Please transcribe this audio into text"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
                        },
                        {"type": "text", "text": "Please transcribe this audio into text"},
                    ],
                },
            ],
        ]

        model = GlmAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)

        inputs_transcription = self.processor.apply_transcription_request(
            [
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
            ],
        ).to(model.device, dtype=model.dtype)

        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], inputs_transcription[key]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        EXPECTED_OUTPUT = [
            "Yesterday it was thirty five degrees in Barcelona, but today the temperature will go down to minus twenty degrees.",
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the",
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
