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

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoConfig, Chronos2Config, T5Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    import torch

    from transformers import Chronos2Model


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ORIGINAL_REPO = _REPO_ROOT / "chronos-forecasting"
_DUMMY_MODEL_PATH = _ORIGINAL_REPO / "test" / "dummy-chronos2-model"
_OFFICIAL_MODEL_PATH = _REPO_ROOT / "chronos-2"
_HAS_LOCAL_ORIGINAL = _DUMMY_MODEL_PATH.is_dir() and importlib.util.find_spec("chronos") is not None

_DUMMY_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]


def _tiny_chronos2_config(use_reg_token=True, attn_implementation="eager", **chronos_overrides):
    chronos_config = {
        "context_length": 16,
        "input_patch_size": 4,
        "input_patch_stride": 4,
        "output_patch_size": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "use_reg_token": use_reg_token,
        "use_arcsinh": True,
        "max_output_patches": 2,
        "time_encoding_scale": 16,
    }
    chronos_config.update(chronos_overrides)
    return Chronos2Config(
        d_model=12,
        d_kv=4,
        d_ff=16,
        num_layers=2,
        num_heads=3,
        dropout_rate=0.0,
        initializer_factor=0.05,
        feed_forward_proj="relu",
        rope_theta=10000.0,
        attn_implementation=attn_implementation,
        chronos_config=chronos_config,
    )


def _expected_state_dict_keys(num_layers):
    keys = {
        "encoder.final_layer_norm.weight",
        "input_patch_embedding.hidden_layer.bias",
        "input_patch_embedding.hidden_layer.weight",
        "input_patch_embedding.output_layer.bias",
        "input_patch_embedding.output_layer.weight",
        "input_patch_embedding.residual_layer.bias",
        "input_patch_embedding.residual_layer.weight",
        "output_patch_embedding.hidden_layer.bias",
        "output_patch_embedding.hidden_layer.weight",
        "output_patch_embedding.output_layer.bias",
        "output_patch_embedding.output_layer.weight",
        "output_patch_embedding.residual_layer.bias",
        "output_patch_embedding.residual_layer.weight",
        "shared.weight",
    }
    for layer_idx in range(num_layers):
        prefix = f"encoder.block.{layer_idx}.layer"
        keys.update(
            {
                f"{prefix}.0.layer_norm.weight",
                f"{prefix}.0.self_attention.k.weight",
                f"{prefix}.0.self_attention.o.weight",
                f"{prefix}.0.self_attention.q.weight",
                f"{prefix}.0.self_attention.v.weight",
                f"{prefix}.1.layer_norm.weight",
                f"{prefix}.1.self_attention.k.weight",
                f"{prefix}.1.self_attention.o.weight",
                f"{prefix}.1.self_attention.q.weight",
                f"{prefix}.1.self_attention.v.weight",
                f"{prefix}.2.layer_norm.weight",
                f"{prefix}.2.mlp.wi.weight",
                f"{prefix}.2.mlp.wo.weight",
            }
        )
    return keys


class Chronos2ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.context_length = 16
        self.num_output_patches = 2
        self.output_patch_size = 4
        self.num_quantiles = 3
        self.num_hidden_layers = 2
        self.num_attention_heads = 3
        self.hidden_size = 12
        self.is_training = False
        self.seq_length = self.context_length // 4 + 1 + self.num_output_patches

    def get_config(self):
        return _tiny_chronos2_config()

    def prepare_config_and_inputs(self):
        config = self.get_config()
        context = torch.linspace(
            -2.0,
            3.0,
            self.batch_size * self.context_length,
            dtype=torch.float32,
            device=torch_device,
        ).reshape(self.batch_size, self.context_length)
        return config, {"context": context, "num_output_patches": self.num_output_patches}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()


@require_torch
class Chronos2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Chronos2Model,) if is_torch_available() else ()
    all_generative_model_classes = ()
    is_encoder_decoder = False
    has_attentions = False
    test_all_params_have_gradient = False
    test_inputs_embeds = False
    test_mismatched_shapes = False
    test_resize_embeddings = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = Chronos2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Chronos2Config,
            has_text_modality=False,
            common_properties=["hidden_size", "num_attention_heads", "num_hidden_layers"],
        )

    def _get_model(self, config=None, attn_implementation="eager"):
        torch.manual_seed(0)
        config = config if config is not None else self.model_tester.get_config()
        model = Chronos2Model._from_config(config, attn_implementation=attn_implementation)
        return model.to(torch_device).eval()

    def test_config(self):
        self.config_tester.run_common_tests()
        config = self.model_tester.get_config()
        self.assertEqual(config.model_type, "chronos2")
        self.assertEqual(config.chronos_config["context_length"], self.model_tester.context_length)
        self.assertEqual(config.chronos_config["quantiles"], [0.1, 0.5, 0.9])

    def test_config_always_disables_cache_and_encoder_decoder_mode(self):
        self.assertFalse(Chronos2Config().use_cache)
        self.assertFalse(Chronos2Config(use_cache=True).use_cache)
        self.assertFalse(Chronos2Config().is_encoder_decoder)
        self.assertFalse(Chronos2Config(is_encoder_decoder=True).is_encoder_decoder)

    def test_config_rejects_invalid_forecasting_settings(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must be a positive integer"):
            _tiny_chronos2_config(context_length=16.5)
        with self.assertRaisesRegex(
            StrictDataclassClassValidationError, "must be finite and strictly between 0 and 1"
        ):
            _tiny_chronos2_config(quantiles=[0.1, float("nan"), 0.9])

    def test_create_and_run_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = self._get_model(config)

        with torch.no_grad():
            output = model(**inputs)

        self.assertEqual(
            output.quantile_preds.shape,
            (self.model_tester.batch_size, self.model_tester.num_quantiles, 8),
        )
        self.assertTrue(torch.isfinite(output.quantile_preds).all())

    def test_rejects_integer_future_tensors(self):
        model = self._get_model()
        context = torch.ones(2, 16, dtype=torch.float32, device=torch_device)
        integer_future = torch.ones(2, 8, dtype=torch.long, device=torch_device)

        with self.assertRaisesRegex(ValueError, "`future_covariates` must be a floating-point tensor"):
            model(context=context, future_covariates=integer_future, num_output_patches=2)
        with self.assertRaisesRegex(ValueError, "`future_target` must be a floating-point tensor"):
            model(context=context, future_target=integer_future, num_output_patches=2)

    @unittest.skip(reason="Chronos-2 has prepared time-series values instead of public input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_independent_univariate_groups_do_not_mix_batch_rows(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 1.0, 16, device=torch_device),
                torch.linspace(-1.0, 2.0, 16, device=torch_device),
            ]
        )
        perturbed_context = context.clone()
        perturbed_context[1] = 5.0 * torch.sin(torch.linspace(0.0, 12.0, 16, device=torch_device))
        group_ids = torch.tensor([0, 1], device=torch_device)

        with torch.no_grad():
            original = model(context=context, group_ids=group_ids).quantile_preds
            perturbed = model(context=perturbed_context, group_ids=group_ids).quantile_preds

        torch.testing.assert_close(original[0], perturbed[0], atol=1e-6, rtol=1e-6)

    def test_multivariate_group_attention_mixes_rows(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 1.0, 16, device=torch_device),
                torch.linspace(-1.0, 2.0, 16, device=torch_device),
            ]
        )
        perturbed_context = context.clone()
        perturbed_context[1] = 5.0 * torch.sin(torch.linspace(0.0, 12.0, 16, device=torch_device))
        group_ids = torch.zeros(2, dtype=torch.long, device=torch_device)

        with torch.no_grad():
            original = model(context=context, group_ids=group_ids).quantile_preds
            perturbed = model(context=perturbed_context, group_ids=group_ids).quantile_preds

        self.assertGreater((original[0] - perturbed[0]).abs().max().item(), 1e-7)

    def test_known_future_covariates_affect_grouped_target(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 3.0, 16, device=torch_device),
                torch.linspace(4.0, 7.0, 16, device=torch_device),
            ]
        )
        group_ids = torch.zeros(2, dtype=torch.long, device=torch_device)
        future_covariates = torch.full((2, 8), torch.nan, device=torch_device)
        future_covariates[1] = torch.linspace(8.0, 11.5, 8, device=torch_device)
        changed_covariates = future_covariates.clone()
        changed_covariates[1] = torch.linspace(-8.0, -11.5, 8, device=torch_device)

        with torch.no_grad():
            baseline = model(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=2,
            ).quantile_preds
            changed = model(
                context=context,
                group_ids=group_ids,
                future_covariates=changed_covariates,
                num_output_patches=2,
            ).quantile_preds

        self.assertEqual(baseline.shape, (2, 3, 8))
        self.assertGreater((baseline[0] - changed[0]).abs().max().item(), 1e-7)

    def test_inferred_and_explicit_masks_match_with_nans(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 3.0, 16, device=torch_device),
                torch.linspace(-2.0, 2.0, 16, device=torch_device),
                torch.linspace(4.0, 7.0, 16, device=torch_device),
            ]
        )
        context[0, 2] = torch.nan
        context[1, :4] = torch.nan
        context[2] = torch.nan
        context_mask = torch.isfinite(context)
        explicit_context_values = torch.nan_to_num(context, nan=1234.0)
        future_covariates = torch.full((3, 8), torch.nan, device=torch_device)
        future_covariates[1, :5] = torch.linspace(2.5, 4.5, 5, device=torch_device)
        future_covariates_mask = torch.isfinite(future_covariates)
        explicit_future_values = torch.nan_to_num(future_covariates, nan=1234.0)

        with torch.no_grad():
            inferred = model(
                context=context,
                future_covariates=future_covariates,
                num_output_patches=2,
            ).quantile_preds
            explicit = model(
                context=explicit_context_values,
                context_mask=context_mask,
                future_covariates=explicit_future_values,
                future_covariates_mask=future_covariates_mask,
                num_output_patches=2,
            ).quantile_preds

        torch.testing.assert_close(inferred, explicit, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(inferred).all())

    def test_loss_masks_missing_targets_and_known_future_values(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 3.0, 16, device=torch_device),
                torch.linspace(4.0, 7.0, 16, device=torch_device),
                torch.linspace(-4.0, -1.0, 16, device=torch_device),
            ]
        )
        group_ids = torch.tensor([0, 0, 1], device=torch_device)
        future_covariates = torch.full((3, 8), torch.nan, device=torch_device)
        future_covariates[1] = torch.linspace(8.0, 11.5, 8, device=torch_device)
        future_target = torch.stack(
            [
                torch.linspace(3.5, 7.0, 8, device=torch_device),
                torch.linspace(8.0, 11.5, 8, device=torch_device),
                torch.linspace(-0.5, 3.0, 8, device=torch_device),
            ]
        )
        future_target_mask = torch.ones_like(future_target, dtype=torch.bool)
        future_target_mask[2, 4:] = False

        with torch.no_grad():
            reference_loss = model(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=2,
                future_target=future_target,
                future_target_mask=future_target_mask,
            ).loss

            changed_target = future_target.clone()
            changed_target[1] += 1000.0
            changed_target[2, 4:] -= 1000.0
            changed_loss = model(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=2,
                future_target=changed_target,
                future_target_mask=future_target_mask,
            ).loss

        self.assertIsNotNone(reference_loss)
        torch.testing.assert_close(reference_loss, changed_loss, atol=1e-6, rtol=1e-6)

    def test_backward_has_finite_nonzero_gradients_through_forecasting_paths(self):
        torch.manual_seed(0)
        model = Chronos2Model._from_config(
            _tiny_chronos2_config(),
            attn_implementation="eager",
        ).train()
        context = torch.stack(
            [
                torch.linspace(0.0, 3.0, 16),
                torch.linspace(4.0, 7.0, 16),
                torch.linspace(-4.0, -1.0, 16),
            ]
        )
        context[0, 3] = torch.nan
        group_ids = torch.tensor([0, 0, 1])

        future_covariates = torch.zeros(3, 8)
        future_covariates[1] = torch.linspace(8.0, 11.5, 8)
        future_covariates_mask = torch.zeros_like(future_covariates, dtype=torch.bool)
        future_covariates_mask[1] = True
        future_target = torch.stack(
            [
                torch.linspace(3.5, 7.0, 8),
                torch.linspace(8.0, 11.5, 8),
                torch.linspace(-0.5, 3.0, 8),
            ]
        )
        future_target_mask = torch.ones_like(future_target, dtype=torch.bool)
        future_target_mask[0, [2, 5]] = False
        future_target_mask[2, 6:] = False

        output = model(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=2,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

        representative_parameters = {
            "input patch embedding": model.input_patch_embedding.hidden_layer.weight,
            "time attention": model.encoder.block[0].layer[0].self_attention.q.weight,
            "group attention": model.encoder.block[0].layer[1].self_attention.q.weight,
            "feed forward": model.encoder.block[0].layer[2].mlp.wi.weight,
            "output head": model.output_patch_embedding.output_layer.weight,
        }
        for name, parameter in representative_parameters.items():
            self.assertIsNotNone(parameter.grad, msg=f"Missing gradient for {name}")
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=f"Non-finite gradient for {name}")
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, msg=f"Zero gradient for {name}")

        computed_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(computed_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in computed_gradients))

    def test_with_and_without_reg_token(self):
        context = torch.linspace(-1.0, 2.0, 32, device=torch_device).reshape(2, 16)

        for use_reg_token, expected_vocab_size, expected_sequence_length in [(True, 2, 7), (False, 1, 6)]:
            with self.subTest(use_reg_token=use_reg_token):
                model = self._get_model(_tiny_chronos2_config(use_reg_token=use_reg_token))
                with torch.no_grad():
                    output = model(context=context, num_output_patches=2, output_attentions=True)

                self.assertEqual(model.shared.weight.shape, (expected_vocab_size, 12))
                self.assertEqual(output.quantile_preds.shape, (2, 3, 8))
                self.assertEqual(output.enc_time_self_attn_weights[0].shape[-2:], (expected_sequence_length,) * 2)

    def test_eager_matches_sdpa_and_returns_dual_attentions(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model_eager = self._get_model(config, attn_implementation="eager")
        model_sdpa = self._get_model(config, attn_implementation="sdpa")
        model_sdpa.load_state_dict(model_eager.state_dict(), strict=True)

        with torch.no_grad():
            eager_output = model_eager(**inputs, output_attentions=True)
            sdpa_output = model_sdpa(**inputs)
            sdpa_with_attentions = model_sdpa(**inputs, output_attentions=True)

        torch.testing.assert_close(eager_output.quantile_preds, sdpa_output.quantile_preds, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            eager_output.quantile_preds, sdpa_with_attentions.quantile_preds, atol=1e-5, rtol=1e-5
        )

        expected_time_shape = (3, 3, 7, 7)
        expected_group_shape = (7, 3, 3, 3)
        self.assertEqual(len(eager_output.enc_time_self_attn_weights), 2)
        self.assertEqual(len(eager_output.enc_group_self_attn_weights), 2)
        self.assertEqual(eager_output.enc_time_self_attn_weights[0].shape, expected_time_shape)
        self.assertEqual(eager_output.enc_group_self_attn_weights[0].shape, expected_group_shape)
        self.assertEqual(sdpa_with_attentions.enc_time_self_attn_weights[0].shape, expected_time_shape)
        self.assertEqual(sdpa_with_attentions.enc_group_self_attn_weights[0].shape, expected_group_shape)

    def test_attention_implementation_can_be_switched_dynamically(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = self._get_model(config, attn_implementation="eager")

        with torch.no_grad():
            eager_output = model(**inputs).quantile_preds
            model.set_attn_implementation("sdpa")
            sdpa_output = model(**inputs).quantile_preds

        self.assertEqual(model.config._attn_implementation, "sdpa")
        torch.testing.assert_close(eager_output, sdpa_output, atol=1e-5, rtol=1e-5)

    def test_strict_state_dict_and_save_reload(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = self._get_model(config)
        expected_keys = _expected_state_dict_keys(config.num_layers)
        self.assertSetEqual(set(model.state_dict()), expected_keys)

        with torch.no_grad():
            expected_output = model(**inputs).quantile_preds

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded, loading_info = Chronos2Model.from_pretrained(
                tmp_dir,
                attn_implementation="eager",
                output_loading_info=True,
            )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertSetEqual(set(reloaded.state_dict()), expected_keys)
        reloaded.to(torch_device).eval()
        with torch.no_grad():
            actual_output = reloaded(**inputs).quantile_preds
        torch.testing.assert_close(expected_output, actual_output, atol=0.0, rtol=0.0)

    def test_tensor_predict_unrolls_long_horizon(self):
        model = self._get_model()
        context = torch.stack(
            [
                torch.linspace(0.0, 3.0, 16, device=torch_device),
                torch.linspace(4.0, 7.0, 16, device=torch_device),
            ]
        )
        group_ids = torch.zeros(2, dtype=torch.long, device=torch_device)
        future_covariates = torch.full((2, 12), torch.nan, device=torch_device)
        future_covariates[1] = torch.linspace(8.0, 13.5, 12, device=torch_device)

        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            prediction = model.predict(
                context=context,
                prediction_length=12,
                group_ids=group_ids,
                future_covariates=future_covariates,
                max_output_patches=1,
            )

        self.assertEqual(prediction.shape, (2, 3, 12))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreaterEqual(forward.call_count, 3)

        with self.assertRaisesRegex(ValueError, "must be strictly increasing"):
            model.predict(
                context=context,
                prediction_length=12,
                group_ids=group_ids,
                future_covariates=future_covariates,
                max_output_patches=1,
                unrolled_quantiles=[0.9, 0.5, 0.1],
            )

    def test_tensor_predict_direct_horizon_uses_model_quantiles(self):
        model = self._get_model()
        context = torch.linspace(0.0, 3.0, 16, device=torch_device).unsqueeze(0)

        prediction = model.predict(context=context, prediction_length=4)

        self.assertEqual(prediction.shape, (1, 3, 4))
        self.assertTrue(torch.isfinite(prediction).all())


@require_torch
@unittest.skipUnless(_HAS_LOCAL_ORIGINAL, "requires the pinned local Chronos source and dummy checkpoint")
class Chronos2LocalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chronos.chronos2.model import Chronos2Model as OriginalChronos2Model
        from chronos.chronos2.pipeline import Chronos2Pipeline as OriginalChronos2Pipeline

        cls.original_model = OriginalChronos2Model.from_pretrained(
            _DUMMY_MODEL_PATH, attn_implementation="eager"
        ).eval()
        cls.original_pipeline = OriginalChronos2Pipeline(cls.original_model)
        cls.native_model, cls.loading_info = Chronos2Model.from_pretrained(
            _DUMMY_MODEL_PATH,
            attn_implementation="eager",
            output_loading_info=True,
        )
        cls.native_model.eval()

    def test_legacy_config_disables_cache_but_keeps_generic_auto_dispatch(self):
        direct_config = Chronos2Config.from_pretrained(_DUMMY_MODEL_PATH)
        self.assertIsInstance(direct_config, Chronos2Config)
        self.assertEqual(direct_config.model_type, "chronos2")
        self.assertFalse(direct_config.use_cache)
        self.assertFalse(direct_config.is_encoder_decoder)

        if _OFFICIAL_MODEL_PATH.is_dir():
            official_legacy_config = Chronos2Config.from_pretrained(_OFFICIAL_MODEL_PATH)
            self.assertFalse(official_legacy_config.use_cache)

        auto_config = AutoConfig.from_pretrained(_DUMMY_MODEL_PATH)
        self.assertIs(type(auto_config), T5Config)
        self.assertNotIsInstance(auto_config, Chronos2Config)
        self.assertEqual(auto_config.model_type, "t5")

    def test_dummy_checkpoint_strict_load_and_forward_parity(self):
        self.assertFalse(self.loading_info["missing_keys"])
        self.assertFalse(self.loading_info["unexpected_keys"])
        self.assertFalse(self.loading_info["mismatched_keys"])
        self.assertEqual(len(self.native_model.state_dict()), 40)
        self.assertEqual(self.native_model.config.chronos_config["quantiles"], _DUMMY_QUANTILES)
        self.assertSetEqual(set(self.native_model.state_dict()), set(self.original_model.state_dict()))
        for key, original_value in self.original_model.state_dict().items():
            torch.testing.assert_close(self.native_model.state_dict()[key], original_value, atol=0.0, rtol=0.0)

        context = torch.stack(
            [
                torch.arange(1, 19, dtype=torch.float32),
                torch.linspace(-3.0, 6.0, 18),
                torch.tensor([(-1) ** idx * (idx + 1) for idx in range(18)], dtype=torch.float32),
            ]
        )
        context[0, 2] = torch.nan
        context[2, 0] = torch.nan
        future_covariates = torch.full((3, 20), torch.nan)
        future_covariates[1] = torch.linspace(7.0, 16.5, 20)
        future_target = torch.stack(
            [
                torch.linspace(19.0, 38.0, 20),
                torch.linspace(7.0, 16.5, 20),
                torch.linspace(-10.0, 9.0, 20),
            ]
        )
        group_ids = torch.tensor([5, 5, 9])
        model_inputs = {
            "context": context,
            "group_ids": group_ids,
            "future_covariates": future_covariates,
            "num_output_patches": 2,
            "future_target": future_target,
            "output_attentions": True,
        }

        with torch.no_grad():
            original_output = self.original_model(**model_inputs)
            native_output = self.native_model(**model_inputs)

        self.assertEqual(native_output.quantile_preds.shape, (3, 21, 32))
        torch.testing.assert_close(native_output.quantile_preds, original_output.quantile_preds, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(native_output.loss, original_output.loss, atol=1e-5, rtol=1e-4)
        self.assertAlmostEqual(native_output.loss.item(), 10.4460029602, places=5)

        expected_slice = torch.tensor(
            [
                [9.88252449, 9.88307858, 9.88096523, 9.88293934, 9.88433266, 9.88105202],
                [9.88260651, 9.88078880, 9.88410282, 9.88440800, 9.87867928, 9.88221836],
                [9.88123322, 9.88461971, 9.88191891, 9.88217640, 9.88414192, 9.88357639],
            ]
        )
        torch.testing.assert_close(
            native_output.quantile_preds[0, [0, 10, 20], :6], expected_slice, atol=1e-5, rtol=1e-4
        )
        self.assertEqual(len(native_output.enc_time_self_attn_weights), 2)
        self.assertEqual(len(native_output.enc_group_self_attn_weights), 2)
        self.assertEqual(native_output.enc_time_self_attn_weights[0].shape, (3, 4, 5, 5))
        self.assertEqual(native_output.enc_group_self_attn_weights[0].shape, (5, 4, 3, 3))

    def test_dummy_checkpoint_component_and_intermediate_parity(self):
        context = torch.stack(
            [
                torch.arange(1, 19, dtype=torch.float32),
                torch.linspace(-3.0, 6.0, 18),
                torch.tensor([(-1) ** idx * (idx + 1) for idx in range(18)], dtype=torch.float32),
            ]
        )
        context[0, 2] = torch.nan
        context[2, 0] = torch.nan
        future_covariates = torch.full((3, 20), torch.nan)
        future_covariates[1] = torch.linspace(7.0, 16.5, 20)
        group_ids = torch.tensor([5, 5, 9])

        original_patched_context, original_context_mask, original_loc_scale = (
            self.original_model._prepare_patched_context(context)
        )
        native_patched_context, native_context_mask, native_loc_scale = self.native_model._prepare_patched_context(
            context
        )
        torch.testing.assert_close(
            native_patched_context, original_patched_context, atol=1e-5, rtol=1e-4, equal_nan=True
        )
        torch.testing.assert_close(native_context_mask, original_context_mask, atol=0.0, rtol=0.0)
        for native_stat, original_stat in zip(native_loc_scale, original_loc_scale):
            torch.testing.assert_close(native_stat, original_stat, atol=1e-5, rtol=1e-4, equal_nan=True)

        original_patched_future, original_future_mask = self.original_model._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=None,
            loc_scale=original_loc_scale,
            num_output_patches=2,
            batch_size=context.shape[0],
        )
        native_patched_future, native_future_mask = self.native_model._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=None,
            loc_scale=native_loc_scale,
            num_output_patches=2,
            batch_size=context.shape[0],
        )
        torch.testing.assert_close(
            native_patched_future, original_patched_future, atol=1e-5, rtol=1e-4, equal_nan=True
        )
        torch.testing.assert_close(native_future_mask, original_future_mask, atol=0.0, rtol=0.0)

        def first_tensor(output):
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "to_tuple"):
                output = output.to_tuple()
            values = output.values() if isinstance(output, dict) else output
            for value in values:
                if value is None:
                    continue
                try:
                    return first_tensor(value)
                except TypeError:
                    continue
            raise TypeError(f"Hook output of type {type(output)} contains no tensor")

        def capture_intermediates(model):
            captured = {
                "input_patch_embedding": [],
                "encoder_final_layer_norm": [],
                "output_patch_embedding": [],
            }
            modules = {
                "input_patch_embedding": model.input_patch_embedding,
                "encoder_final_layer_norm": model.encoder.final_layer_norm,
                "output_patch_embedding": model.output_patch_embedding,
            }
            for block_index, block in enumerate(model.encoder.block):
                block_modules = {
                    f"block_{block_index}_time_self_attention": block.layer[0],
                    f"block_{block_index}_group_self_attention": block.layer[1],
                    f"block_{block_index}_feed_forward": block.layer[2],
                }
                captured.update({name: [] for name in block_modules})
                modules.update(block_modules)
            handles = []
            for name, module in modules.items():
                handles.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output, name=name: captured[name].append(
                            first_tensor(output).detach().clone()
                        )
                    )
                )

            try:
                with torch.no_grad():
                    model(
                        context=context,
                        group_ids=group_ids,
                        future_covariates=future_covariates,
                        num_output_patches=2,
                        output_attentions=True,
                    )
            finally:
                for handle in handles:
                    handle.remove()
            return captured

        original_intermediates = capture_intermediates(self.original_model)
        native_intermediates = capture_intermediates(self.native_model)
        self.assertSetEqual(set(native_intermediates), set(original_intermediates))
        for name in native_intermediates:
            self.assertEqual(len(native_intermediates[name]), len(original_intermediates[name]), msg=name)
            for native_tensor, original_tensor in zip(native_intermediates[name], original_intermediates[name]):
                torch.testing.assert_close(
                    native_tensor,
                    original_tensor,
                    atol=1e-5,
                    rtol=1e-4,
                    equal_nan=True,
                    msg=f"Intermediate mismatch for {name}",
                )

    def test_dummy_checkpoint_long_horizon_parity_oracle(self):
        context = torch.stack(
            [
                torch.arange(1, 19, dtype=torch.float32),
                torch.linspace(-3.0, 6.0, 18),
                torch.linspace(10.0, 1.5, 18),
                torch.linspace(100.0, 117.0, 18),
            ]
        )
        context[0, 2] = torch.nan
        context[2, 0] = torch.nan
        future_covariates = torch.full((4, 40), torch.nan)
        future_covariates[3] = torch.linspace(118.0, 157.0, 40)
        group_ids = torch.zeros(4, dtype=torch.long)
        unrolled_quantiles = [0.1, 0.5, 0.9]

        prediction = self.native_model.predict(
            context=context.clone(),
            prediction_length=40,
            group_ids=group_ids.clone(),
            future_covariates=future_covariates.clone(),
            max_output_patches=1,
            unrolled_quantiles=unrolled_quantiles,
        )
        original_prediction = self.original_pipeline._predict_batch(
            context=context.clone(),
            group_ids=group_ids.clone(),
            future_covariates=future_covariates.clone(),
            unrolled_quantiles_tensor=torch.tensor(unrolled_quantiles),
            prediction_length=40,
            max_output_patches=1,
            target_idx_ranges=[(0, 4)],
        )[0]

        self.assertEqual(prediction.shape, (4, 21, 40))
        torch.testing.assert_close(prediction, original_prediction, atol=1e-5, rtol=1e-4)

        changed_future_covariates = future_covariates.clone()
        changed_future_covariates[3, 16:] = torch.linspace(-300.0, -100.0, 24)
        changed_prediction = self.native_model.predict(
            context=context.clone(),
            prediction_length=40,
            group_ids=group_ids.clone(),
            future_covariates=changed_future_covariates.clone(),
            max_output_patches=1,
            unrolled_quantiles=unrolled_quantiles,
        )
        changed_original_prediction = self.original_pipeline._predict_batch(
            context=context.clone(),
            group_ids=group_ids.clone(),
            future_covariates=changed_future_covariates,
            unrolled_quantiles_tensor=torch.tensor(unrolled_quantiles),
            prediction_length=40,
            max_output_patches=1,
            target_idx_ranges=[(0, 4)],
        )[0]
        torch.testing.assert_close(changed_prediction, changed_original_prediction, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(prediction[..., :16], changed_prediction[..., :16], atol=0.0, rtol=0.0)
        self.assertGreater((prediction[0, :, 16:] - changed_prediction[0, :, 16:]).abs().max().item(), 1e-5)

        expected_slice = torch.tensor(
            [
                [
                    9.88230324,
                    9.88268566,
                    9.88117790,
                    9.88305378,
                    9.88385010,
                    9.88139629,
                    9.88115597,
                    9.87986660,
                ],
                [
                    9.88193798,
                    9.88081074,
                    9.88364792,
                    9.88416672,
                    9.87808323,
                    9.88252544,
                    9.88209152,
                    9.88129711,
                ],
                [
                    9.88107300,
                    9.88509369,
                    9.88189507,
                    9.88238335,
                    9.88414860,
                    9.88333130,
                    9.88165569,
                    9.88221741,
                ],
            ]
        )
        torch.testing.assert_close(prediction[0, [0, 10, 20], :8], expected_slice, atol=1e-5, rtol=1e-4)


@require_torch
@slow
class Chronos2ModelIntegrationTests(unittest.TestCase):
    def test_official_checkpoint_direct_load_and_inference(self):
        checkpoint = _OFFICIAL_MODEL_PATH if _OFFICIAL_MODEL_PATH.is_dir() else "amazon/chronos-2"
        from_pretrained_kwargs = {
            "attn_implementation": "eager",
            "dtype": torch.float32,
            "output_loading_info": True,
        }
        if isinstance(checkpoint, str):
            from_pretrained_kwargs["revision"] = "29ec3766d36d6f73f0696f85560a422f50e8498c"

        model, loading_info = Chronos2Model.from_pretrained(checkpoint, **from_pretrained_kwargs)
        model.eval()
        self.assertIsInstance(model.config, Chronos2Config)
        self.assertEqual(model.config.model_type, "chronos2")
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertEqual(len(model.state_dict()), 170)

        context = torch.stack(
            [
                torch.sin(torch.linspace(0.0, 6.0, 64)),
                torch.cos(torch.linspace(0.0, 6.0, 64)),
            ]
        )
        context[0, 5] = torch.nan
        future_covariates = torch.full((2, 16), torch.nan)
        future_covariates[1] = torch.cos(torch.linspace(6.1, 7.6, 16))

        with torch.no_grad():
            output = model(
                context=context,
                group_ids=torch.tensor([3, 3]),
                future_covariates=future_covariates,
                num_output_patches=1,
            )

        self.assertEqual(output.quantile_preds.shape, (2, 21, 16))
        expected_slice = torch.tensor(
            [
                [-0.26775339, -0.16152540, -0.05796464, 0.02774637, 0.12215428, 0.21638122, 0.29484451, 0.37644988],
                [-0.17860347, -0.08282575, 0.01348860, 0.09975394, 0.20543714, 0.29426062, 0.37280804, 0.46145421],
                [-0.15346092, -0.05532935, 0.03936860, 0.12756307, 0.23162898, 0.32012457, 0.40567940, 0.49183056],
                [-0.12341391, -0.02952025, 0.06087688, 0.14681213, 0.25629246, 0.34129411, 0.43456110, 0.52576351],
                [-0.05564229, 0.02899557, 0.12807941, 0.22299704, 0.34397709, 0.44035363, 0.53641635, 0.64615732],
            ]
        )
        torch.testing.assert_close(
            output.quantile_preds[0, [0, 5, 10, 15, 20], :8], expected_slice, atol=1e-4, rtol=1e-4
        )
