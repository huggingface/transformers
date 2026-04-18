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

import random
import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers import CtsmConfig, is_torch_available
from transformers.testing_utils import require_flash_attn, require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin, floats_tensor


if is_torch_available():
    from transformers import CtsmModel, CtsmModelForPrediction


class CtsmModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 8,
        context_length: int = 64,
        horizon_length: int = 8,
        num_hidden_layers: int = 2,
        hidden_size: int = 32,
        intermediate_size: int = 32,
        head_dim: int = 16,
        num_attention_heads: int = 2,
        num_key_value_heads: int = 2,
        rms_norm_eps: float = 1e-6,
        quantiles=(0.1, 0.5, 0.9),
        agg_factor: int = 4,
        max_position_embeddings: int = 64,
        batch_size: int = 2,
        is_training: bool = True,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.quantiles = list(quantiles)
        self.agg_factor = agg_factor
        self.max_position_embeddings = max_position_embeddings
        self.batch_size = batch_size
        self.is_training = is_training

        # Total patches in the concatenated sequence (coarse + special + fine).
        self.seq_length = 2 * (context_length // patch_length) + 1

    def get_config(self):
        return CtsmConfig(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            rms_norm_eps=self.rms_norm_eps,
            quantiles=self.quantiles,
            agg_factor=self.agg_factor,
            max_position_embeddings=self.max_position_embeddings,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        bsize = self.batch_size
        past_values = [
            torch.tensor(
                np.sin(np.linspace(0, 20, self.agg_factor * self.context_length)),
                dtype=torch.float32,
                device=torch_device,
            )
            for _ in range(bsize)
        ]
        return self.get_config(), past_values

    def prepare_config_and_inputs_for_common(self):
        config, past_values = self.prepare_config_and_inputs()
        inputs_dict = {"past_values": past_values}
        return config, inputs_dict


@require_torch
class CtsmModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CtsmModelForPrediction,) if is_torch_available() else ()
    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False
    test_all_params_have_gradient = False
    test_headmasking = False
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False

    def setUp(self):
        self.model_tester = CtsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CtsmConfig, has_text_modality=False)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = CtsmModelForPrediction(config)
        model.to(torch_device)
        model.eval()
        results = model(**inputs_dict)
        self.assertEqual(results.mean_predictions.shape, (self.model_tester.batch_size, config.horizon_length))
        self.assertEqual(
            results.full_predictions.shape,
            (self.model_tester.batch_size, config.horizon_length, 1 + len(config.quantiles)),
        )

    def test_encoder_forward_matches_predict(self):
        """The low-level `CtsmModel.forward` should accept the two-stream interface directly."""
        config = self.model_tester.get_config()
        model = CtsmModel(config).to(torch_device).eval()

        coarse = torch.randn(self.model_tester.batch_size, config.context_length, device=torch_device)
        fine = torch.randn(self.model_tester.batch_size, config.context_length, device=torch_device)
        with torch.no_grad():
            out = model(past_values_coarse=coarse, past_values_fine=fine)

        coarse_patches = config.context_length // config.patch_length
        fine_patches = config.context_length // config.patch_length
        self.assertEqual(
            out.last_hidden_state.shape,
            (self.model_tester.batch_size, coarse_patches + 1 + fine_patches, config.hidden_size),
        )
        self.assertEqual(out.loc.shape, (self.model_tester.batch_size,))
        self.assertEqual(out.loc_coarse.shape, (self.model_tester.batch_size,))

    @unittest.skip(reason="CTSM uses a custom multi-resolution attention mask built internally.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """CTSM builds its own mask from the concatenated stream paddings; the generic harness, which
        injects external attention masks and mutates QK-norm RMSNorm eps, is not compatible. We verify
        eager vs. SDPA equivalence on the low-level `CtsmModel` instead."""
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest("Model does not support SDPA")
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        tolerance = {torch.float32: 1e-4, torch.bfloat16: 5e-3, torch.float16: 5e-3}[torch_dtype]
        self._attn_kernel_equivalence("sdpa", dtype=torch_dtype, tolerance=tolerance)

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="CTSM does not support gradient checkpointing in this version")
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    def _attn_kernel_equivalence(self, attn_implementation, dtype=torch.float32, tolerance=1e-4):
        """Compare eager vs `attn_implementation` on the low-level `CtsmModel`.

        Uses the two-stream interface so we bypass the prediction-head AR loop which
        adds numerical noise unrelated to the kernel choice.
        """
        config = self.model_tester.get_config()
        model_eager = CtsmModel._from_config(config, attn_implementation="eager")
        model_eager.to(dtype=dtype, device=torch_device).eval()

        model_other = CtsmModel._from_config(config, attn_implementation=attn_implementation)
        model_other.load_state_dict(model_eager.state_dict())
        model_other.to(dtype=dtype, device=torch_device).eval()

        coarse = torch.randn(self.model_tester.batch_size, config.context_length, device=torch_device, dtype=dtype)
        fine = torch.randn(self.model_tester.batch_size, config.context_length, device=torch_device, dtype=dtype)

        with torch.no_grad():
            out_e = model_eager(past_values_coarse=coarse, past_values_fine=fine)
            out_o = model_other(past_values_coarse=coarse, past_values_fine=fine)

        diff = (out_e.last_hidden_state - out_o.last_hidden_state).abs().max().item()
        self.assertLess(diff, tolerance, f"{attn_implementation} vs eager last_hidden_state max diff: {diff:.2e}")

    def test_eager_matches_sdpa(self):
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest("Model does not support SDPA")
        self._attn_kernel_equivalence("sdpa", dtype=torch.float32, tolerance=1e-4)

    @require_flash_attn
    @require_torch_accelerator
    def test_flash_attn_2_inference_equivalence(self):
        self._attn_kernel_equivalence("flash_attention_2", dtype=torch.bfloat16, tolerance=1e-2)

    def test_retain_grad_hidden_states_attentions(self):
        """CTSM returns `mean_predictions` as the first tensor, not `last_hidden_state`."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions
        if self.has_attentions:
            config._attn_implementation = "eager"

        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)
        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)

        output_tensor = outputs.mean_predictions
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

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            batch_size = len(inputs_dict["past_values"])
            rng = random.Random(42)
            inputs_dict["future_values"] = floats_tensor([batch_size, self.model_tester.horizon_length], rng=rng)
        return inputs_dict

    def test_kv_cache_matches_full_recompute(self):
        """Cached autoregressive decoding should produce close-to-identical predictions to the
        full-recompute path (the small gap is from the stream-stats-freezing approximation)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = CtsmModelForPrediction(config).to(torch_device).eval()

        # Long enough to trigger AR (horizon > config.horizon_length).
        horizon_len = config.horizon_length * 3
        with torch.no_grad():
            out_full = model(**inputs_dict, horizon_len=horizon_len, use_cache=False)
            out_cache = model(**inputs_dict, horizon_len=horizon_len, use_cache=True)

        # First horizon_length predictions must match bit-exactly (step 1 is identical in both paths).
        step1 = config.horizon_length
        self.assertTrue(
            torch.allclose(out_full.mean_predictions[:, :step1], out_cache.mean_predictions[:, :step1], atol=1e-5),
            msg="Step-1 predictions must match bit-exactly between cached and non-cached paths.",
        )
        # On subsequent AR steps the stats-freezing approximation introduces a small bounded drift.
        # The bound is generous here because the tiny tester model has random weights and a horizon of 8,
        # so compounding any small per-step shift over multiple steps is amplified.
        relative = (out_full.mean_predictions - out_cache.mean_predictions).abs().max() / (
            out_full.mean_predictions.abs().max().clamp_min(1.0)
        )
        self.assertLess(relative.item(), 0.5, f"cached vs full-recompute AR drift {relative.item():.2e} too large")


@require_torch
@slow
class CtsmModelIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model = CtsmModelForPrediction.from_pretrained("cisco-ai/cisco-time-series-model-1.0").to(torch_device)
        rng = np.random.default_rng(42)
        series = (np.sin(np.linspace(0, 200, 512 * 60)) + 0.05 * rng.standard_normal(512 * 60)).astype(np.float32)
        past_values = [torch.tensor(series, device=torch_device)]

        with torch.no_grad():
            output = model(past_values=past_values, horizon_len=128)

        self.assertEqual(output.mean_predictions.shape, (1, 128))
        self.assertEqual(output.full_predictions.shape, (1, 128, 1 + len(model.config.quantiles)))
