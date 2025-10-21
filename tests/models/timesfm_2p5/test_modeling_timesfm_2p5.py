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

import inspect
import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers import Timesfm2P5Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin


if is_torch_available():
    from transformers import Timesfm2P5ModelForPrediction

TOLERANCE = 1e-4


class Timesfm2P5ModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 32,  # Same as original TimesFM 2.5
        context_length: int = 256,  # Smaller for tests
        horizon_length: int = 16,  # 1/8 of original (128/8)
        num_hidden_layers: int = 1,  # Minimal for testing
        hidden_size: int = 128,  # Much smaller: 128 = 16 * 8 quantiles per step
        intermediate_size: int = 256,  # 2x hidden_size
        head_dim: int = 16,  # hidden_size // num_heads = 128 // 8
        num_heads: int = 8,  # Smaller for tests
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        output_quantile_len: int = 64,  # Scaled down from 1024 (1024/16 = 64)
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
        self.output_quantile_len = output_quantile_len
        self.pad_val = pad_val
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
        return Timesfm2P5Config(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            quantiles=self.quantiles,
            output_quantile_len=self.output_quantile_len,
            pad_val=self.pad_val,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,  # Same for full attention
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
            # Note: TimesFM 2.5 doesn't use freq parameter (simplified API)
        }
        return config, inputs_dict


@require_torch
class Timesfm2P5ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Timesfm2P5ModelForPrediction,) if is_torch_available() else ()
    all_generative_model_classes = ()
    all_parallelizable_model_classes = ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = Timesfm2P5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Timesfm2P5Config)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Timesfm2P5ModelForPrediction(config)
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
        model_signature = inspect.signature(getattr(Timesfm2P5ModelForPrediction, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(Timesfm2P5ModelForPrediction.main_input_name, observed_main_input_name)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(
        "TimesFM 2.5 has ~5% numerical differences between eager and SDPA backends "
        "due to complex attention mechanisms with RoPE and per-dim scaling"
    )
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        """
        TimesFM 2.5 has numerical stability issues with fp16 SDPA due to the complex attention mechanisms
        and rotary embeddings that lead to NaN values in fp16 precision.
        """
        # Check for string dtype before torch.dtype conversion happens in _test_eager_matches_sdpa_inference
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        from ...test_modeling_common import _test_eager_matches_sdpa_inference

        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
        )

    def test_retain_grad_hidden_states_attentions(self):
        """
        TimesFM 2.5 specific test for retain_grad since the model returns mean_predictions
        as the first tensor, not last_hidden_state like standard models.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # force eager attention to support output attentions
        if self.has_attentions:
            config._attn_implementation = "eager"

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        # TimesFM 2.5 returns mean_predictions as first output, not last_hidden_state
        output_tensor = outputs.mean_predictions

        # Encoder-/Decoder-only models
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

        if self.has_attentions and outputs.attentions is not None:
            attentions = outputs.attentions[0]
            attentions.retain_grad()

        output_tensor.flatten()[0].backward(retain_graph=True)

        if outputs.hidden_states is not None:
            self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions and outputs.attentions is not None:
            self.assertIsNotNone(attentions.grad)


@require_torch
@slow
class Timesfm2P5ModelIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model = Timesfm2P5ModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(torch_device)
        forecast_input = [
            np.sin(np.linspace(0, 20, 100)),
            np.sin(np.linspace(0, 20, 200)),
            np.sin(np.linspace(0, 20, 400)),
        ]
        forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32, device=torch_device) for ts in forecast_input]

        with torch.no_grad():
            output = model(past_values=forecast_input_tensor)

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
