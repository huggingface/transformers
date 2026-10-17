# Copyright 2026 Google DeepMind and HuggingFace Inc. team.
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

import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from transformers import WeatherNext2Config, WeatherNext2FeatureExtractor, is_torch_available
from transformers.testing_utils import (
    Expectations,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    sdpa_kernel,
)


# Real geometry on a 15-degree grid and a 162-node mesh.
TINY_CHECKPOINT = "hf-internal-testing/tiny-random-WeatherNext2ForWeatherForecasting"


if is_torch_available():
    import torch

    from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2GenerationConfig, WeatherNext2Model


class WeatherNext2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        edge_hidden_size=4,
        noise_channels=4,
        mesh_splits=2,
        attention_k_hop=2,
        attention_bandwidth=43,
        num_grid_to_mesh_edges=576,
        grid_latitudes=13,
        grid_longitudes=24,
        pressure_levels=(500, 850),
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.edge_hidden_size = edge_hidden_size
        self.noise_channels = noise_channels
        self.mesh_splits = mesh_splits
        self.attention_k_hop = attention_k_hop
        self.attention_bandwidth = attention_bandwidth
        self.num_grid_to_mesh_edges = num_grid_to_mesh_edges
        self.grid_latitudes = grid_latitudes
        self.grid_longitudes = grid_longitudes
        self.pressure_levels = pressure_levels
        self.is_training = is_training

    def get_config(self):
        return WeatherNext2Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            graph_config={"edge_hidden_size": self.edge_hidden_size, "aggregate_normalization": None},
            noise_channels=self.noise_channels,
            mesh_splits=self.mesh_splits,
            attention_k_hop=self.attention_k_hop,
            attention_bandwidth=self.attention_bandwidth,
            num_grid_to_mesh_edges=self.num_grid_to_mesh_edges,
            grid_latitudes=self.grid_latitudes,
            grid_longitudes=self.grid_longitudes,
            pressure_levels=self.pressure_levels,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        grid_features = floats_tensor(
            [self.batch_size, config.num_grid_input_channels - 3, self.grid_latitudes, self.grid_longitudes]
        )
        global_features = floats_tensor([self.batch_size, config.num_mesh_input_channels - 3])
        noise = floats_tensor([self.batch_size, self.noise_channels])
        return config, grid_features, global_features, noise

    def create_and_check_model(self, config, grid_features, global_features, noise):
        model = WeatherNext2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            grid_features=grid_features,
            global_features=global_features,
            noise=noise,
            attention_mask=self.prepare_attention_mask(config),
        )
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, config.num_grid_points, self.hidden_size)
        )
        self.parent.assertEqual(
            result.mesh_hidden_state.shape, (self.batch_size, config.num_mesh_nodes, self.hidden_size)
        )

    def get_feature_extractor(self):
        config = self.get_config()
        variables = set(config.input_variables) | set(config.target_variables)
        return WeatherNext2FeatureExtractor(
            input_variables=config.input_variables,
            target_variables=config.target_variables,
            forcing_variables=config.forcing_variables,
            atmospheric_variables=config.atmospheric_variables,
            static_variables=config.static_variables,
            global_variables=config.global_variables,
            pressure_levels=config.pressure_levels,
            mean_by_level={name: [0.0] * config.num_levels(name) for name in variables},
            stddev_by_level={name: [1.0] * config.num_levels(name) for name in variables},
            diffs_stddev_by_level={name: [1.0] * config.num_levels(name) for name in variables},
            num_input_timesteps=config.num_input_timesteps,
            time_step_hours=config.time_step_hours,
            grid_latitudes=config.grid_latitudes,
            grid_longitudes=config.grid_longitudes,
            noise_channels=config.noise_channels,
            attention_mask=self.prepare_attention_mask(config),
        )

    def prepare_state(self, extractor, batch_size, seed=0):
        generator = np.random.default_rng(seed)
        frames = extractor.num_input_timesteps
        latitudes, longitudes = extractor.grid_latitudes, extractor.grid_longitudes
        state = {}
        for name in extractor.input_variables:
            if name in extractor.static_variables:
                shape = (latitudes, longitudes)
            elif name in extractor.global_variables:
                shape = (batch_size, frames)
            elif name in extractor.forcing_variables:
                shape = (batch_size, frames, longitudes)
            elif name in extractor.atmospheric_variables:
                shape = (batch_size, frames, len(extractor.pressure_levels), latitudes, longitudes)
            else:
                shape = (batch_size, frames, latitudes, longitudes)
            state[name] = generator.standard_normal(shape).astype(np.float32)
        return state

    def prepare_attention_mask(self, config):
        block_size = min(config.attention_bandwidth, config.num_mesh_nodes)
        num_blocks = -(-config.num_mesh_nodes // block_size)
        attention_mask = torch.zeros(num_blocks, 1, block_size, 3 * block_size, dtype=torch.bool, device=torch_device)
        attention_mask[..., block_size : 2 * block_size] = True
        return attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, grid_features, global_features, noise = self.prepare_config_and_inputs()
        return config, {
            "grid_features": grid_features,
            "global_features": global_features,
            "noise": noise,
            "attention_mask": self.prepare_attention_mask(config),
        }


@require_torch
class WeatherNext2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (WeatherNext2Model, WeatherNext2ForWeatherForecasting) if is_torch_available() else ()
    all_generative_model_classes = ()
    additional_model_inputs = ["global_features", "noise", "attention_mask"]

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = WeatherNext2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WeatherNext2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @require_torch_accelerator
    @slow
    def test_sdpa_can_compile_dynamic(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        inputs = {
            key: value.to(torch.bfloat16) if value.is_floating_point() else value for key, value in inputs.items()
        }
        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="sdpa").to(torch_device, dtype=torch.bfloat16)
            model = torch.compile(model.eval(), dynamic=True)
            with torch.no_grad():
                model(**inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forecast_backward(self):
        """Use real geometry so the edge encoders participate in the forecast."""
        model = WeatherNext2ForWeatherForecasting.from_pretrained(TINY_CHECKPOINT).to(torch_device).train()
        config = model.config
        prediction = model(
            attention_mask=WeatherNext2FeatureExtractor.from_pretrained(TINY_CHECKPOINT).attention_mask.to(
                torch_device
            ),
            grid_features=floats_tensor(
                [2, config.num_grid_input_channels - 3, config.grid_latitudes, config.grid_longitudes]
            ).to(torch_device),
            global_features=floats_tensor([2, config.num_mesh_input_channels - 3]).to(torch_device),
            noise=floats_tensor([2, config.noise_channels]).to(torch_device),
        ).prediction
        prediction.square().mean().backward()

        starved = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
        }
        expected = {
            name for name, _ in model.named_parameters() if name.startswith("model.mesh_to_grid.mesh_node_update.")
        }
        self.assertEqual(starved, expected)
        self.assertTrue(expected)

    def test_noise_drives_the_ensemble(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = WeatherNext2ForWeatherForecasting(config).to(torch_device).eval()

        different_noise = dict(inputs)
        different_noise["noise"] = inputs["noise"].flip(0)
        with torch.no_grad():
            baseline = model(**inputs).prediction
            repeated = model(**inputs).prediction
            perturbed = model(**different_noise).prediction

        torch.testing.assert_close(baseline, repeated)
        self.assertFalse(torch.allclose(baseline, perturbed))

    @parameterized.expand([(None, False, None), (1, False, 2), (1, True, 3)])
    def test_generate_matches_manual_rollout(self, batch_size, sample_noise, configured_steps):
        config = self.model_tester.get_config()
        extractor = self.model_tester.get_feature_extractor()
        model = WeatherNext2ForWeatherForecasting(config).to(torch_device).eval()
        state = self.model_tester.prepare_state(extractor, batch_size=2)
        valid_time = np.full(2, np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64))
        noise = torch.randn(2, 2, config.noise_channels, device=torch_device)
        generation_config = None
        if configured_steps is not None:
            model.generation_config.num_members = 1
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model = WeatherNext2ForWeatherForecasting.from_pretrained(tmp_dir).to(torch_device).eval()
            self.assertEqual(model.generation_config.num_members, 1)
            generation_config = WeatherNext2GenerationConfig(num_steps=configured_steps)

        generated = model.generate(
            state=state,
            feature_extractor=extractor,
            seconds_since_epoch=valid_time,
            num_steps=None if configured_steps == 2 else 2,
            generation_config=generation_config,
            noise=None if sample_noise else noise,
            attention_mask=extractor.attention_mask,
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(0) if sample_noise else None,
        )
        if generation_config is not None:
            self.assertEqual(generation_config.num_steps, configured_steps)
            self.assertIsNone(generation_config.num_members)

        manual_state = state
        manual_time = valid_time
        manual_forecasts = []
        generator = torch.Generator().manual_seed(0) if sample_noise else None
        for step in range(2):
            inputs = extractor(
                manual_state,
                seconds_since_epoch=manual_time,
                attention_mask=extractor.attention_mask,
                noise=None if sample_noise else noise[step],
                generator=generator,
            ).to(torch_device)
            with torch.no_grad():
                prediction = model(**inputs).prediction
            forecast = extractor.postprocess(prediction, manual_state)
            manual_forecasts.append(forecast)
            manual_state = extractor.advance_state(manual_state, forecast, manual_time)
            manual_time = manual_time + extractor.time_step_hours * 3600

        np.testing.assert_array_equal(generated.valid_time, manual_time)
        for generated_step, manual_step in zip(generated.forecasts, manual_forecasts):
            for name in extractor.target_variables:
                torch.testing.assert_close(generated_step[name], manual_step[name])
        for name in extractor.input_variables:
            np.testing.assert_allclose(generated.state[name], manual_state[name], rtol=1e-5, atol=1e-5)

    @parameterized.expand([(None,), (1,), (4,)])
    def test_generate_num_members(self, batch_size):
        """Batched ensemble members must match individual forecasts with the same noise."""
        config = self.model_tester.get_config()
        extractor = self.model_tester.get_feature_extractor()
        model = WeatherNext2ForWeatherForecasting(config).to(torch_device).eval()
        state = self.model_tester.prepare_state(extractor, batch_size=2)
        valid_time = torch.full((2,), 1_728_280_800)  # 2024-10-07T06:00:00Z
        members = 3
        model.generation_config.num_members = members
        noise = torch.randn(1, 2 * members, config.noise_channels, device=torch_device)
        name = "2m_temperature"

        forecast = model.generate(
            state=state,
            feature_extractor=extractor,
            seconds_since_epoch=valid_time,
            num_steps=1,
            num_members=members if batch_size is not None else None,
            noise=noise,
            batch_size=batch_size,
        ).forecasts[0][name]
        self.assertEqual(forecast.shape[0], 2 * members)

        for condition in range(2):
            alone = {
                k: v if k in extractor.static_variables else v[condition : condition + 1] for k, v in state.items()
            }
            for member in range(members):
                row = condition * members + member
                expected = model.generate(
                    state=alone,
                    feature_extractor=extractor,
                    seconds_since_epoch=valid_time[condition : condition + 1],
                    num_steps=1,
                    num_members=1,
                    noise=noise[:, row : row + 1],
                ).forecasts[0][name][0]
                torch.testing.assert_close(forecast[row], expected, rtol=1e-4, atol=1e-4)
        self.assertFalse(torch.allclose(forecast[0], forecast[1]))

    def test_attention_outputs(self):
        """Shared attention-output checks with block-local mesh shapes."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            num_blocks, _, block_size, key_length = inputs_dict["attention_mask"].shape
            expected = (
                self.model_tester.batch_size * num_blocks,
                self.model_tester.num_attention_heads,
                block_size,
                key_length,
            )

            # via the forward argument
            with torch.no_grad():
                outputs = model(**inputs_dict, output_attentions=True)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertEqual(tuple(attentions[0].shape), expected)

            # via the config
            config.output_attentions = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            self.assertEqual(len(outputs.attentions), self.model_tester.num_hidden_layers)
            self.assertEqual(tuple(outputs.attentions[0].shape), expected)
            config.output_attentions = False

            # and off by default
            with torch.no_grad():
                outputs = model_class._from_config(config).to(torch_device).eval()(**inputs_dict)
            self.assertIsNone(outputs.attentions)

    def test_hidden_states_output(self):
        """Shared hidden-state checks with blocked mesh-node shapes."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            num_blocks, _, block_size, _ = inputs_dict["attention_mask"].shape
            expected = (self.model_tester.batch_size, num_blocks, block_size, self.model_tester.hidden_size)

            # via the forward argument
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
            self.assertEqual(len(outputs.hidden_states), self.model_tester.num_hidden_layers + 1)
            self.assertEqual(tuple(outputs.hidden_states[0].shape), expected)

            # via the config
            config.output_hidden_states = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            self.assertEqual(len(outputs.hidden_states), self.model_tester.num_hidden_layers + 1)
            self.assertEqual(tuple(outputs.hidden_states[0].shape), expected)
            config.output_hidden_states = False

            # and off by default
            with torch.no_grad():
                outputs = model_class(config).to(torch_device).eval()(**inputs_dict)
            self.assertIsNone(outputs.hidden_states)

    @unittest.skip(reason="WeatherNext 2 has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        if not use_attention_mask or padding_side != "left":
            self.skipTest("WeatherNext2 requires a geometry mask and has no left/right token padding.")

        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        if dtype == torch.float16 and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 is not supported on {torch_device}")
        if dtype == torch.bfloat16 and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(f"bfloat16 is not supported on {torch_device}")

        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                model = model_class(config).to(torch_device, dtype=dtype).eval()
                prepared_inputs = self._prepare_for_class(inputs, model_class)
                with (
                    torch.no_grad(),
                    sdpa_kernel(enable_flash=enable_kernels, enable_math=True, enable_mem_efficient=enable_kernels),
                ):
                    model.set_attn_implementation("eager")
                    eager_outputs = model(**prepared_inputs, output_attentions=output_attentions)
                    model.set_attn_implementation("sdpa")
                    sdpa_outputs = model(**prepared_inputs, output_attentions=output_attentions)
                for key in eager_outputs:
                    if key != "attentions":
                        torch.testing.assert_close(sdpa_outputs[key], eager_outputs[key])

    @unittest.skip(reason="SDPA flash attention cannot take the mesh-adjacency mask.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
@slow
class WeatherNext2ModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    checkpoint = "kashif/weathernext2-mini"

    def test_inference_expected_values(self):
        """Mini forecast with seeded normalized inputs; tolerances allow FP32 reduction drift."""
        model = WeatherNext2ForWeatherForecasting.from_pretrained(self.checkpoint).to(torch_device)
        config = model.config

        generator = torch.Generator().manual_seed(0)
        attention_mask = WeatherNext2FeatureExtractor.from_pretrained(self.checkpoint).attention_mask.to(torch_device)
        grid_features = torch.randn(
            1, config.num_grid_input_channels - 3, config.grid_latitudes, config.grid_longitudes, generator=generator
        )
        global_features = torch.randn(1, config.num_mesh_input_channels - 3, generator=generator)
        noise = torch.randn(1, config.noise_channels, generator=generator)

        with torch.no_grad():
            prediction = model(
                grid_features=grid_features.to(torch_device),
                global_features=global_features.to(torch_device),
                noise=noise.to(torch_device),
                attention_mask=attention_mask,
            ).prediction

        self.assertEqual(
            prediction.shape, (1, config.num_output_channels, config.grid_latitudes, config.grid_longitudes)
        )
        self.assertTrue(torch.isfinite(prediction).all())

        expected_pole = Expectations({(None, None): [-0.0312, -5.3831, -3.9925, -7.2537, -4.8488]}).get_expectation()
        expected_equator = Expectations(
            {(None, None): [-5.7967, -4.5518, -2.5775, -2.2516, -4.0098]}
        ).get_expectation()
        expected_pole = torch.tensor(expected_pole, device=torch_device)
        expected_equator = torch.tensor(expected_equator, device=torch_device)
        torch.testing.assert_close(prediction[0, 0, 0, :5].float(), expected_pole, rtol=1e-3, atol=3e-3)
        torch.testing.assert_close(
            prediction[0, 0, config.grid_latitudes // 2, :5].float(), expected_equator, rtol=1e-3, atol=3e-3
        )
        expected_mean = Expectations({(None, None): 0.3225}).get_expectation()
        self.assertAlmostEqual(prediction.float().mean().item(), expected_mean, delta=1e-3)
