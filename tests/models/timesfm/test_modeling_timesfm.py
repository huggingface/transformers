# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
import torch

from transformers import TimesFmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    from transformers import TimesFmModelForPrediction

TOLERANCE = 1e-4


class TimesFmModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        freq_size: int = 3,
        num_hidden_layers: int = 1,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        head_dim: int = 8,
        num_heads: int = 2,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        use_positional_embedding: bool = True,
        initializer_factor: float = 0.0,
        is_training: bool = False,
        batch_size: int = 3,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.use_positional_embedding = use_positional_embedding
        self.initializer_factor = initializer_factor
        self.is_training = is_training
        self.batch_size = batch_size

        # The size of test input
        self.seq_length = context_length // patch_length
        self.hidden_size = hidden_size

    def get_config(self):
        return TimesFmConfig(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            quantiles=self.quantiles,
            pad_val=self.pad_val,
            freq_size=self.freq_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            tolerance=self.tolerance,
            rms_norm_eps=self.rms_norm_eps,
            use_positional_embedding=self.use_positional_embedding,
            initializer_factor=self.initializer_factor,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        forecast_input = [
            torch.tensor(np.sin(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.cos(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.tan(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
        ]
        frequency_input = torch.tensor([0, 1, 2], dtype=torch.long, device=torch_device)

        return (self.get_config(), torch.stack(forecast_input, dim=0), frequency_input)

    def prepare_config_and_inputs_for_common(self):
        (config, forecast_input, frequency_input) = self.prepare_config_and_inputs()

        inputs_dict = {
            "past_values": forecast_input,
            "freq": frequency_input,
        }
        return config, inputs_dict


@require_torch
class TimesFmModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimesFmModelForPrediction,) if is_torch_available() else ()
    all_generative_model_classes = ()
    all_parallelizable_model_classes = ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimesFmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesFmConfig)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimesFmModelForPrediction(config)
        model.to(torch_device)
        model.eval()
        results = model(**inputs_dict)
        assert results.mean_predictions is not None

    @unittest.skip(reason="Compile not yet supported because of masks")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Model does not have head mask")
    def test_headmasking(self):
        pass

    # the main input name is `inputs`
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TimesFmModelForPrediction, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TimesFmModelForPrediction.main_input_name, observed_main_input_name)


@require_torch
@slow
class TimesFmModelIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(torch_device)
        forecast_input = [
            np.sin(np.linspace(0, 20, 100)),
            np.sin(np.linspace(0, 20, 200)),
            np.sin(np.linspace(0, 20, 400)),
        ]
        forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32, device=torch_device) for ts in forecast_input]
        frequency_input = [0, 1, 2]

        with torch.no_grad():
            output = model(past_values=forecast_input_tensor, freq=frequency_input)

        mean_predictions = output.mean_predictions
        self.assertEqual(mean_predictions.shape, torch.Size([3, model.config.horizon_length]))
        # fmt: off
        expected_slice = torch.tensor(
            [ 0.9813,  1.0086,  0.9985,  0.9432,  0.8505,  0.7203,  0.5596,  0.3788,
              0.1796, -0.0264, -0.2307, -0.4255, -0.5978, -0.7642, -0.8772, -0.9670,
             -1.0110, -1.0162, -0.9848, -0.9151, -0.8016, -0.6511, -0.4707, -0.2842,
             -0.0787,  0.1260,  0.3293,  0.5104,  0.6818,  0.8155,  0.9172,  0.9843,
              1.0101,  1.0025,  0.9529,  0.8588,  0.7384,  0.5885,  0.4022,  0.2099,
             -0.0035, -0.2104, -0.4146, -0.6033, -0.7661, -0.8818, -0.9725, -1.0191,
             -1.0190, -0.9874, -0.9137, -0.8069, -0.6683, -0.4939, -0.3086, -0.1106,
              0.0846,  0.2927,  0.4832,  0.6612,  0.8031,  0.9051,  0.9772,  1.0064
            ],
            device=torch_device)
        # fmt: on
        self.assertTrue(torch.allclose(mean_predictions[0, :64], expected_slice, atol=TOLERANCE))
