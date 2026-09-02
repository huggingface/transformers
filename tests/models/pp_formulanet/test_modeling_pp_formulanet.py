# coding = utf-8
# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PPFormulaNet model."""

import copy
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    PPFormulaNetConfig,
    PPFormulaNetForConditionalGeneration,
    PPFormulaNetModel,
    PPFormulaNetTextConfig,
    PPFormulaNetVisionConfig,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


# NOTE: PPFormulaNet is not a typical VLM; it follows an encoder-decoder architecture.
class PPFormulaNetModelTester(VLMModelTester):
    base_model_class = PPFormulaNetModel
    config_class = PPFormulaNetConfig
    text_config_class = PPFormulaNetTextConfig
    vision_config_class = PPFormulaNetVisionConfig
    conditional_generation_class = PPFormulaNetForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("hidden_size", 48)
        kwargs.setdefault("image_size", 768)
        kwargs.setdefault("patch_size", 768)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("num_hidden_layers", 1)
        kwargs.setdefault("is_training", False)
        kwargs.setdefault(
            "vision_config",
            {
                "image_size": 768,
                "patch_size": 16,
                "hidden_size": 48,
                "windows_size": 14,
                "num_hidden_layers": 1,
                "output_channels": 16,
                "num_attention_heads": 2,
                "global_attn_indexes": [1, 1, 1, 1],
                "mlp_dim": 1,
                "post_conv_in_channels": 16,
                "post_conv_mid_channels": 16,
                "post_conv_out_channels": 16,
                "decoder_hidden_size": 48,
            },
        )
        kwargs.setdefault(
            "text_config",
            {
                "decoder_ffn_dim": 16,
                "decoder_layers": 1,
                "d_model": 48,
                "vocab_size": 99,
            },
        )
        super().__init__(parent, **kwargs)
        self.seq_length = self.image_size // self.patch_size
        self.encoder_seq_length = self.vision_config["windows_size"] ** 2
        self.decoder_seq_length = 1

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        decoder_input_ids = torch.full((self.batch_size, 1), 2, dtype=torch.long, device=torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "input_ids": decoder_input_ids,
        }
        return config, inputs_dict

    def get_config(self) -> PPFormulaNetConfig:
        config = PPFormulaNetConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            num_hidden_layers=self.num_hidden_layers,
        )

        return config


@require_torch
class PPFormulaNetModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = PPFormulaNetModelTester
    all_model_classes = (PPFormulaNetForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-text-to-text": PPFormulaNetForConditionalGeneration} if is_torch_available() else {}
    )

    test_resize_embeddings = False
    is_encoder_decoder = True

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, prompt_length):
        # Ignoring batch size for now as it is dynamically changed during window partitioning
        encoder_config = self.model_tester.vision_config
        prompt_length = encoder_config["windows_size"] ** 2
        encoder_expected_shape = (prompt_length, prompt_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape[-2:] for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, prompt_length):
        # update encoder_expected_shape
        encoder_config = self.model_tester.vision_config
        patched_image_size = encoder_config["image_size"] // encoder_config["patch_size"]
        encoder_expected_shape = (patched_image_size, patched_image_size, encoder_config["hidden_size"])
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape[-3:] for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    # use encoder_seq_length and decoder_seq_length to replace seq_len
    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            self._set_subconfig_attributes(config, "output_attentions", True)
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            # Ignoring batch size for now as it is dynamically changed during window partitioning
            self.assertListEqual(
                list(attentions[0].shape[-2:]),
                [self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            )

            attentions = outputs.decoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            # Ignoring batch size for now as it is dynamically changed during window partitioning
            self.assertListEqual(
                list(attentions[0].shape[-2:]),
                [self.model_tester.decoder_seq_length, self.model_tester.decoder_seq_length],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            encoder_config = self.model_tester.vision_config
            seq_length = encoder_config["image_size"] // encoder_config["patch_size"]

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            self._set_subconfig_attributes(config, "output_hidden_states", True)
            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="PPFormulaNet does not use inputs_embeds")
    def test_enable_input_require_grads(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPFormulaNetTextModel has no attribute `shared`")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPFormulaNet does not support generation from no inputs")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not support image_token")
    def test_mismatching_num_image_tokens(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not support data parallel")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="PPFormulaNet does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPFormulaNet does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        reason="PPFormulaNet does not support continuing generation from past_key_values across generate calls."
    )
    def test_generate_continue_from_past_key_values(self):
        pass


@require_torch
@require_vision
@slow
class PPFormulaNetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-FormulaNet_plus-L_safetensors"
        self.model = PPFormulaNetForConditionalGeneration.from_pretrained(model_path).to(torch_device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        img_url = url_to_local_path(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png"
        )
        self.image = load_image(img_url)

    def test_inference_formula_recognition_head(self):
        inputs = self.processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        formula_text = self.processor.post_process(outputs)
        expected_formula_text = [
            "\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d z\\frac{2z^{2}}{(z^{2}+\\omega^{2})^{\\nu+1}}\\breve{\\Psi}(\\omega;z)e^{i\\epsilon z}\\quad,"
        ]

        self.assertEqual(formula_text, expected_formula_text)
