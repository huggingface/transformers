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
"""Testing suite for the PyTorch SmolVLM model."""

import unittest
from io import BytesIO

import pytest
import requests

from transformers import (
    AutoProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flaky,
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

    from transformers import (
        GenerationConfig,
        SmolVLMConfig,
        SmolVLMForConditionalGeneration,
        SmolVLMModel,
    )

if is_vision_available():
    from PIL import Image


class SmolVLMVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        is_training=True,
        batch_size=2,
        seq_length=7,
        scale_factor=2,
        num_images=2,
        vision_config={
            "image_size": 24,  # 9 image tokens per image, not a multiple of patch_size
            "patch_size": 4,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 56,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 2,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "image_token_id": 57,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "sliding_window": 32,
            "attention_dropout": 0.0,
        },
        use_cache=False,
        tie_word_embeddings=False,
        image_token_id=57,
    ):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_images = num_images
        self.scale_factor = scale_factor
        self.image_seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (self.scale_factor**2))
            * self.num_images
        )
        self.seq_length = seq_length + self.image_seq_length
        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.pad_token_id = text_config["pad_token_id"]
        self.tie_word_embeddings = tie_word_embeddings
        # Hack - add properties here so use common tests
        self.vocab_size = text_config["vocab_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]

        self.vision_config = vision_config
        self.text_config = text_config

    def get_config(self):
        return SmolVLMConfig(
            use_cache=self.use_cache,
            image_token_id=self.image_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            vision_config=self.vision_config.copy(),
            text_config=self.text_config.copy(),
            vocab_size=self.vocab_size,
            scale_factor=self.scale_factor,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_images,
                3,  # SmolVLMImageProcessorPil always generates RGB pixel values
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 2) + 1

        # For simplicity just set the first n tokens to the image token
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class SmolVLMModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `SmolVLM`.
    """

    all_model_classes = (SmolVLMModel,) if is_torch_available() else ()
    skip_test_image_features_output_shape = True  # SmolVLM merges batch_size with num_images in index 0
    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = SmolVLMVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=SmolVLMConfig, has_text_modality=False, common_properties=["image_token_id"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Model fails to export though it can eagerly forward with same inputs")
    def test_onnx_export_generate_dynamic(self):
        pass

    @unittest.skip(reason="Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    @unittest.skip(reason="Compile not yet supported in SmolVLM models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported in SmolVLM models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class SmolVLMForConditionalGenerationModelTest(
    GenerationTesterMixin, ModelTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `SmolVLMForConditionalGeneration`.
    """

    all_model_classes = (SmolVLMForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (SmolVLMForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": SmolVLMForConditionalGeneration,
            "any-to-any": SmolVLMForConditionalGeneration,
        }
        if is_torch_available()
        else ()
    )
    skip_test_image_features_output_shape = True  # SmolVLM merges batch_size with num_images in index 0
    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = SmolVLMVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SmolVLMConfig, has_text_modality=False)

    @unittest.skip(reason="Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    @pytest.mark.generate
    @is_flaky(description="TODO: check why flaky")
    def test_generate_methods_with_logits_to_keep(self):
        super().test_generate_methods_with_logits_to_keep()

    @unittest.skip(reason="Unsupported")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(reason="Compile not yet supported in SmolVLM models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported in SmolVLM models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class SmolVLMForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/statue_of_liberty.jpg"
                ).content
            )
        )

        self.video_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "path": "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/video/assisted_generation_gif_1_1080p.mov",
                    },
                    {"type": "text", "text": "Describe this video in detail"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_integration_test(self):
        model = SmolVLMForConditionalGeneration.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image1
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=9)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = Expectations(
            {
                (None, None): "\n\n\n\nIn this image, we see the Statue of Liberty, a green statue,",
                ("rocm", (9, 4)): "\n\n\n\nIn this image, we see the Statue of Liberty, a green statue of",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(generated_texts[0], expected_generated_text)

    @slow
    def test_integration_test_video(self):
        model = SmolVLMForConditionalGeneration.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        inputs = self.processor.apply_chat_template(
            self.video_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = Expectations(
            {
                (None, None): 'User: You are provided the following series of four frames from a 0:00:04 [H:MM:SS] video.\n\nFrame from 00:00:\nFrame from 00:01:\nFrame from 00:02:\nFrame from 00:03:\n\nDescribe this video in detail\nAssistant: The video begins with a blue sky and a yellow sun in the upper left corner, suggesting a day',
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(generated_texts[0], expected_generated_text)

    @slow
    def test_export_smolvlm_vision_encoder(self):
        from transformers import AutoConfig
        from transformers.integrations.executorch import TorchExportableModuleForVLM

        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

        # NOTE: The attention_mask is prepared internally in the vision encoder, depending on whether flash attention is used or not
        # For ExecuTorch, flash attention is not supported, so the way of exporting vison encoder should be compatible with text-decoder
        config = AutoConfig.from_pretrained(model_id)
        config.text_config._flash_attn_2_enabled = False

        # Load model and extract vision encoder
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float32,
            config=config,
        )

        exportable_module = TorchExportableModuleForVLM(model)
        exported_program = exportable_module.export_vision_encoder()
        self.assertIsInstance(exported_program, torch.export.ExportedProgram)

    @slow
    def test_export_smolvlm_connector(self):
        from transformers import AutoConfig
        from transformers.integrations.executorch import TorchExportableModuleForVLM

        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

        # NOTE: The attention_mask is prepared internally in the vision encoder, depending on whether flash attention is used or not
        # For ExecuTorch, flash attention is not supported, so the way of exporting vison encoder should be compatible with text-decoder
        config = AutoConfig.from_pretrained(model_id)
        config.text_config._flash_attn_2_enabled = False

        # Load the model and extract the connector (multi-modal projector)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float32,
            config=config,
        )

        connector = model.model.connector
        connector.eval()

        exportable_module = TorchExportableModuleForVLM(model)
        exported_program = exportable_module.export_connector()
        self.assertIsInstance(exported_program, torch.export.ExportedProgram)

    @slow
    def test_export_smolvlm_text_decoder(self):
        from transformers import AutoConfig
        from transformers.integrations.executorch import TorchExportableModuleForVLM

        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

        # NOTE: The attention_mask is prepared internally in the vision encoder, depending on whether flash attention is used or not
        # For ExecuTorch, flash attention is not supported, so the way of exporting vison encoder should be compatible with text-decoder
        config = AutoConfig.from_pretrained(model_id)
        config.text_config._flash_attn_2_enabled = False
        config.text_config.use_cache = True
        config.text_config.attn_implementation = "sdpa"

        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=1234,
            cache_config={
                "batch_size": 1,
                "max_cache_len": 1234,
            },
        )

        # Load the model and extract the text decoder
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float32,
            config=config,
        )

        model.model.text_model.generation_config = generation_config

        text_decoder = model.model.text_model
        text_decoder.eval()

        exportable_module = TorchExportableModuleForVLM(model)
        exported_program = exportable_module.export_text_decoder()
        self.assertIsInstance(exported_program, torch.export.ExportedProgram)
