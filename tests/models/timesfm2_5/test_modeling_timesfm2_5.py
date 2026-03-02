# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers import TimesFm2_5Config, is_torch_available
from transformers.testing_utils import require_flash_attn, require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin


if is_torch_available():
    from transformers import TimesFm2_5ModelForPrediction


class TimesFm2_5ModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 32,
        context_length: int = 128,
        horizon_length: int = 8,
        num_hidden_layers: int = 1,
        hidden_size: int = 32,  # 2 heads * 16 head_dim
        intermediate_size: int = 64,
        head_dim: int = 16,
        num_heads: int = 2,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        output_quantile_len: int = 16,
        is_training: bool = False,
        batch_size: int = 2,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = quantiles
        self.output_quantile_len = output_quantile_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_heads
        self.rms_norm_eps = rms_norm_eps
        self.is_training = is_training
        self.batch_size = batch_size

        # The size of test input
        self.seq_length = context_length // patch_length

    def get_config(self):
        return TimesFm2_5Config(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            quantiles=self.quantiles,
            output_quantile_len=self.output_quantile_len,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            rms_norm_eps=self.rms_norm_eps,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        forecast_input = torch.stack(
            [
                torch.tensor(np.sin(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
                torch.tensor(np.cos(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            ]
        )
        return self.get_config(), forecast_input

    def prepare_config_and_inputs_for_common(self):
        config, forecast_input = self.prepare_config_and_inputs()
        inputs_dict = {"past_values": forecast_input}
        return config, inputs_dict


@require_torch
class TimesFm2_5ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimesFm2_5ModelForPrediction,) if is_torch_available() else ()
    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimesFm2_5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesFm2_5Config)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimesFm2_5ModelForPrediction(config)
        model.to(torch_device)
        model.eval()
        results = model(**inputs_dict)
        assert results.mean_predictions is not None

    @unittest.skip(reason="FA backend not yet supported because of forced masks")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """
        TimesFM 2.5 computes its own causal attention mask internally from the input padding,
        so the generic test harness (which injects external attention masks and sets RMSNorm eps=1.0
        on QK-norm layers) is not compatible. This override directly verifies eager vs SDPA equivalence.
        """
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest("Model does not support SDPA")

        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp32":
            dtype = torch.float32

        tolerance = {torch.float32: 1e-5, torch.bfloat16: 1e-3, torch.float16: 1e-3}[dtype]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        model_eager = TimesFm2_5ModelForPrediction._from_config(config, attn_implementation="eager")
        model_eager.to(dtype=dtype, device=torch_device)
        model_eager.eval()

        model_sdpa = TimesFm2_5ModelForPrediction._from_config(config, attn_implementation="sdpa")
        model_sdpa.load_state_dict(model_eager.state_dict())
        model_sdpa.to(dtype=dtype, device=torch_device)
        model_sdpa.eval()

        past_values = inputs_dict["past_values"].to(dtype=dtype, device=torch_device)

        with torch.no_grad():
            out_eager = model_eager(past_values=past_values)
            out_sdpa = model_sdpa(past_values=past_values)

        # Compare mean predictions
        self.assertTrue(
            torch.allclose(out_eager.mean_predictions, out_sdpa.mean_predictions, atol=tolerance),
            f"mean_predictions max diff: {(out_eager.mean_predictions - out_sdpa.mean_predictions).abs().max().item():.2e}",
        )
        # Compare full predictions
        self.assertTrue(
            torch.allclose(out_eager.full_predictions, out_sdpa.full_predictions, atol=tolerance),
            f"full_predictions max diff: {(out_eager.full_predictions - out_sdpa.full_predictions).abs().max().item():.2e}",
        )
        # Compare last hidden state
        hs_eager = out_eager.hidden_states[-1]
        hs_sdpa = out_sdpa.hidden_states[-1]
        self.assertTrue(
            torch.allclose(hs_eager, hs_sdpa, atol=tolerance),
            f"hidden_states max diff: {(hs_eager - hs_sdpa).abs().max().item():.2e}",
        )

    def _test_flash_or_flex_attn_inference_equivalence(self, attn_implementation):
        """
        TimesFM 2.5 computes its own attention mask internally, so the generic
        flash/flex equivalence test (which injects external attention masks) does not apply.
        This override directly verifies eager vs flash/flex equivalence.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        dtype = torch.bfloat16
        tolerance = 1e-2

        model_eager = TimesFm2_5ModelForPrediction._from_config(config, attn_implementation="eager")
        model_eager.to(dtype=dtype, device=torch_device)
        model_eager.eval()

        model_fa = TimesFm2_5ModelForPrediction._from_config(config, attn_implementation=attn_implementation)
        model_fa.load_state_dict(model_eager.state_dict())
        model_fa.to(dtype=dtype, device=torch_device)
        model_fa.eval()

        past_values = inputs_dict["past_values"].to(dtype=dtype, device=torch_device)

        with torch.no_grad():
            out_eager = model_eager(past_values=past_values)
            out_fa = model_fa(past_values=past_values)

        self.assertTrue(
            torch.allclose(out_eager.mean_predictions, out_fa.mean_predictions, atol=tolerance),
            f"mean_predictions max diff: {(out_eager.mean_predictions - out_fa.mean_predictions).abs().max().item():.2e}",
        )
        hs_eager = out_eager.hidden_states[-1]
        hs_fa = out_fa.hidden_states[-1]
        self.assertTrue(
            torch.allclose(hs_eager, hs_fa, atol=tolerance),
            f"hidden_states max diff: {(hs_eager - hs_fa).abs().max().item():.2e}",
        )

    @require_flash_attn
    @require_torch_accelerator
    def test_flash_attn_2_inference_equivalence(self):
        self._test_flash_or_flex_attn_inference_equivalence("flash_attention_2")

    @require_flash_attn
    @require_torch_accelerator
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self._test_flash_or_flex_attn_inference_equivalence("flash_attention_2")

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
class TimesFm2_5ModelIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", revision="refs/pr/3"
        ).to(torch_device)
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
            [ 0.9745,  1.0047,  0.9707,  0.9161,  0.8041,  0.6829,  0.5378,  0.3563,
              0.1698, -0.0396, -0.2508, -0.4358, -0.6150, -0.7491, -0.8659, -0.9535,
             -1.0024, -0.9977, -0.9557, -0.8840, -0.7716, -0.6092, -0.4526, -0.2582,
             -0.0554,  0.1263,  0.3258,  0.5207,  0.6667,  0.7989,  0.9002,  0.9782,
              0.9848,  0.9877,  0.9339,  0.8473,  0.7109,  0.5525,  0.3799,  0.1756,
             -0.0285, -0.2325, -0.4137, -0.5926, -0.7425, -0.8532, -0.9444, -0.9878,
             -0.9985, -0.9828, -0.8972, -0.7833, -0.6414, -0.4881, -0.2838, -0.0878,
              0.1169,  0.3137,  0.4918,  0.6508,  0.7762,  0.8961,  0.9666,  0.9910
            ],
            device=torch_device)
        # fmt: on
        self.assertTrue(torch.allclose(mean_predictions[0, :64], expected_slice, atol=1e-4))
