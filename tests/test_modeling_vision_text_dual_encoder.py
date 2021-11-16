# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch VisionTextDualEncoder model. """


import tempfile
import unittest

import numpy as np

from tests.test_modeling_common import floats_tensor
from transformers import VisionTextDualEncoderConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import VisionTextDualEncoderConfig, VisionTextDualEncoderModel


@require_torch
class VisionTextDualEncoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_model_from_pretrained_configs(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        config = VisionTextDualEncoderConfig.from_text_vision_configs(text_config, vision_config)

        model = VisionTextDualEncoderModel(config)
        model.to(torch_device)
        model.eval()

        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], config.projection_dim))

    def check_vision_text_dual_encoder_model(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_vision_text_dual_encoder_from_pretrained(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):

        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        kwargs = {"vision_model": vision_model, "text_model": text_model}
        model = VisionTextDualEncoderModel.from_text_vision_pretrained(**kwargs)
        model.to(torch_device)
        model.eval()

        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_save_load(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            out_1 = output[0].cpu().numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = VisionTextDualEncoderModel.from_pretrained(tmpdirname)
                model.to(torch_device)

                after_output = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )
                out_2 = after_output[0].cpu().numpy()
                max_diff = np.amax(np.abs(out_2 - out_1))
                self.assertLessEqual(max_diff, 1e-5)

    def check_save_load_vision_text_model(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):

        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            out_1 = output[0].cpu().numpy()

            with tempfile.TemporaryDirectory() as vision_tmpdirname, tempfile.TemporaryDirectory() as text_tmpdirname:
                vision_model.save_pretrained(vision_tmpdirname)
                text_model.save_pretrained(text_tmpdirname)

                model = VisionTextDualEncoderModel.from_text_vision_pretrained(text_tmpdirname, vision_tmpdirname)
                model.to(torch_device)

                after_output = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )
                out_2 = after_output[0].cpu().numpy()
                max_diff = np.amax(np.abs(out_2 - out_1))
                self.assertLessEqual(max_diff, 1e-5)
