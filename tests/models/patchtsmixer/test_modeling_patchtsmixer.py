# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch PatchTSMixer model. """

import inspect
import tempfile
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import is_torch_available
from transformers.testing_utils import is_flaky, require_torch, torch_device, slow
from transformers.models.auto import get_values
import random
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    import torch
    from transformers import PatchTSMixerConfig, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING
    from transformers import PatchTSMixerModel, PatchTSMixerForForecasting, PatchTSMixerForPretraining
    # \
        # PatchTSMixerForClassification, PatchTSMixerForRegression



@require_torch
class PatchTSMixerModelTester:
    def __init__(
        self,
        seq_len: int = 32,
        patch_len: int = 8,
        in_channels: int = 3,
        stride: int = 8,
        num_features: int = 128,
        expansion_factor: int = 2,
        num_layers: int = 8,
        dropout: float = 0.5,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp="LayerNorm",
        swin_hier: int = 0,
        # masking related
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        mask_value=0,
        masked_loss: bool = False,
        mask_mode: str = "mask_before_encoder",
        channel_consistent_masking: bool = True,
        revin: bool = True,
        # Head related
        head_dropout: float = 0.2,
        # forecast related
        forecast_len: int = 16,
        out_channels: int = None,
        # Classification/regression related
        n_classes: int = 3,
        n_targets: int = 3,
        output_range: list = None,
        head_agg: str = None,
        # Trainer related
        batch_size=13,
        is_training=True
    ):
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.swin_hier = swin_hier
        self.revin = revin
        self.head_dropout = head_dropout
        # masking related
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.mask_value = mask_value
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_mode = mask_mode
        self.masked_loss = masked_loss
        # patching related
        self.patch_last = True
        # forecast related
        self.forecast_len = forecast_len
        self.out_channels = out_channels
        # classification/regression related
        self.n_classes = n_classes
        self.n_targets = n_targets
        self.output_range = output_range
        self.head_agg = head_agg
        # Trainer related
        self.batch_size = batch_size
        self.is_training = is_training

    def get_config(self):
        return PatchTSMixerConfig(
            in_channels = self.in_channels,
            seq_len = self.seq_len,
            patch_len = self.patch_len,
            stride = self.stride,
            num_features = self.num_features,
            expansion_factor = self.expansion_factor,
            num_layers = self.num_layers,
            dropout = self.dropout,
            mode = self.mode,
            gated_attn = self.gated_attn,
            norm_mlp = self.norm_mlp,
            swin_hier = self.swin_hier,
            revin = self.revin,
            head_dropout = self.head_dropout,
            mask_type = self.mask_type,
            mask_ratio = self.mask_ratio,
            mask_patches = self.mask_patches,
            mask_patch_ratios = self.mask_patch_ratios,
            mask_value = self.mask_value,
            channel_consistent_masking = self.channel_consistent_masking,
            mask_mode = self.mask_mode,
            masked_loss = self.masked_loss,
            forecast_len = self.forecast_len,
            out_channels = self.out_channels,
            n_classes = self.n_classes,
            n_targets = self.n_targets,
            output_range = self.output_range,
            head_agg = self.head_agg
        )

    def prepare_patchtsmixer_inputs_dict(self, config):
        _past_length = config.seq_len
        # bs, n_vars, num_patch, patch_len

        # [bs x seq_len x n_vars]
        past_values = floats_tensor([self.batch_size, _past_length, self.in_channels])

        future_values = floats_tensor([self.batch_size, config.forecast_len, self.in_channels])

        inputs_dict = {
            "past_values": past_values,
            "future_values": future_values,
        }
        return inputs_dict

    def prepare_config_and_inputs(self):
        config = self.get_config()
        inputs_dict = self.prepare_patchtsmixer_inputs_dict(config)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class PatchTSMixerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (PatchTSMixerModel,
         PatchTSMixerForForecasting,
         PatchTSMixerForPretraining,)
        #  PatchTSMixerForClassification,
        #  PatchTSMixerForRegression)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (PatchTSMixerForForecasting, PatchTSMixerForPretraining) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": PatchTSMixerModel} if is_torch_available() else {}
    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False
    test_model_common_attributes = False


    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_model_parallel = False
    has_attentions = False

    def setUp(self):
        self.model_tester = PatchTSMixerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PatchTSMixerConfig,
            has_text_modality=False,
            forecast_len=self.model_tester.forecast_len,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        # if classification model:
        if model_class in get_values(MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING):
            rng = random.Random(self.model_tester.seed_number)
            labels = ids_tensor([self.model_tester.batch_size], self.model_tester.num_classes, rng=rng)
            inputs_dict["labels"] = labels
            inputs_dict.pop("future_values")
        elif model_class in get_values(MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING):
            rng = random.Random(self.model_tester.seed_number)
            labels = floats_tensor([self.model_tester.batch_size, self.model_tester.target_dimension], rng=rng)
            inputs_dict["labels"] = labels
            inputs_dict.pop("future_values")
        return inputs_dict

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

#
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            num_patch = self.model_tester.num_patches
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [num_patch, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            print('model_class: ', model_class)

            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)
#
#     # Ignore since we have no tokens embeddings

    def test_resize_tokens_embeddings(self):
        pass

    def test_model_outputs_equivalence(self):
        pass

    def test_determinism(self):
        pass

    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(PatchTSMixerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(PatchTSMixerModel.main_input_name, observed_main_input_name)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "past_values",
                "future_values",
            ]
            if model_class in get_values(MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING) or \
                    model_class in get_values(MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING):
                expected_arg_names.remove("future_values")
                expected_arg_names.append("labels")
            expected_arg_names.extend(
                [
                    "output_hidden_states",
                ]
            )

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @is_flaky()
    def test_retain_grad_hidden_states_attentions(self):
        super().test_retain_grad_hidden_states_attentions()


# def prepare_batch(repo_id="diepi/test-etth1", file='train-batch.pt'):
#     file = hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset")
#     batch = torch.load(file, map_location=torch_device)
#     return batch


# @require_torch
# @slow
# class PatchTSMixerModelIntegrationTests(unittest.TestCase):
    # def test_pretrain_head(self):
    #     model = PatchTSMixerForPretraining.from_pretrained('diepi/test_patchtsmixer_pretrained_etth1').to(torch_device)
    #     batch = prepare_batch()

    #     torch.manual_seed(0)
    #     with torch.no_grad():
    #         output = model(
    #             past_values=batch["past_values"].to(torch_device)
    #         ).prediction_output
    #     num_patch = (max(model.config.context_length,
    #                      model.config.patch_length) - model.config.patch_length) // model.config.stride + 1
    #     expected_shape = torch.Size([64, model.config.in_channels, num_patch, model.config.patch_length])
    #     self.assertEqual(output.shape, expected_shape)

    #     expected_slice = torch.tensor([[[-0.0170]], [[0.0163]], [[0.0090]], [[0.0139]], [[0.0067]],
    #                                    [[0.0246]], [[0.0090]]],
    #                                   device=torch_device)
    #     self.assertTrue(torch.allclose(output[0, :7, :1, :1], expected_slice, atol=TOLERANCE))

    # # def test_classification_head(self):
    # #     # mock data, test
    # #     model = PatchTSMixerForClassification.from_pretrained('diepi/test_patchtsmixer_classification_mock').to(torch_device)
    # #     batch = prepare_batch(repo_id="diepi/mock-data", file="test-mock-patchtsmixer.pt")
    # #
    # #     torch.manual_seed(0)
    # #     with torch.no_grad():
    # #         output = model(
    # #             past_values=batch["past_values"].to(torch_device)
    # #         ).prediction_logits
    # #     expected_shape = torch.Size([1, model.config.num_classes])
    # #     self.assertEqual(output.shape, expected_shape)
    # #
    # #     expected_slice = torch.tensor([[-0.2774, -0.1081, 0.6771]],
    # #                                   device=torch_device,
    # #                                   )
    # #     self.assertTrue(torch.allclose(output, expected_slice, atol=TOLERANCE))

    # def test_prediction_head(self):
    #     model = PatchTSMixerForForecasting.from_pretrained('diepi/test_patchtsmixer_prediction_etth1').to(torch_device)
    #     batch = prepare_batch(file="test-batch.pt")

    #     torch.manual_seed(0)
    #     with torch.no_grad():
    #         output = model(
    #             past_values=batch["past_values"].to(torch_device),
    #             future_values=batch["future_values"].to(torch_device)
    #         ).prediction_output
    #     expected_shape = torch.Size([64, model.config.forecast_len, model.config.in_channels])
    #     self.assertEqual(output.shape, expected_shape)

    #     expected_slice = torch.tensor([[-0.8200, 0.3741, -0.7543, 0.3971, -0.6659, -0.0124, -0.8308]],
    #                                   device=torch_device,
    #                                   )
    #     self.assertTrue(torch.allclose(output[0, :1, :7], expected_slice, atol=TOLERANCE))

    # # def test_seq_to_seq_generation(self):
    # #     model = PatchTSMixerForForecasting.from_pretrained("diepi/test_patchtsmixer_prediction_etth1").to(torch_device)
    # #     batch = prepare_batch("val-batch.pt")
    # #
    # #     torch.manual_seed(0)
    # #     with torch.no_grad():
    # #         outputs = model.generate(
    # #             past_values=batch["past_values"].to(torch_device),
    # #             future_values=batch["future_values"].to(torch_device)
    # #         ).prediction_output
    # #     expected_shape = torch.Size((64, model.config.num_parallel_samples, model.config.forecast_len))
    # #     # self.assertEqual(outputs.sequences.shape, expected_shape)
    # #     #
    # #     # expected_slice = torch.tensor([3400.8005, 4289.2637, 7101.9209], device=torch_device)
    # #     # mean_prediction = outputs.sequences.mean(dim=1)
    # #     # self.assertTrue(torch.allclose(mean_prediction[0, -3:], expected_slice, rtol=1e-1))
    # #
    # #     # expected_shape = torch.Size([64, model.config.forecast_len, model.config.in_channels])
    # #     self.assertEqual(outputs.shape, expected_shape)
    # #
    # #     expected_slice = torch.tensor([[-0.8200, 0.3741, -0.7543, 0.3971, -0.6659, -0.0124, -0.8308]],
    # #                                   device=torch_device,
    # #                                   )
    # #     self.assertTrue(torch.allclose(outputs[0, :1, :7], expected_slice, atol=TOLERANCE))