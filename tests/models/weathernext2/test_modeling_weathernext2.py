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

import unittest

import numpy as np
from parameterized import parameterized

from transformers import WeatherNext2Config, WeatherNext2FeatureExtractor, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
)


# A 15 degree grid on a twice-split icosahedron, randomly initialized, carrying the geometry trimesh
# and scipy build at conversion time. 152 KB, so the tests that need a real mesh can have one.
TINY_CHECKPOINT = "hf-internal-testing/tiny-random-WeatherNext2ForWeatherForecasting"


if is_torch_available():
    import torch

    from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2Model
    from transformers.models.weathernext2.modeling_weathernext2 import WeatherNext2Attention


class WeatherNext2ModelTester:
    """Builds a model small enough to test: a twice-refined icosahedron (162 nodes) on a 15 degree grid.

    The shape matches `TINY_CHECKPOINT`, so a model built here and one loaded from the Hub differ only
    in whether their geometry is real.
    """

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
            edge_hidden_size=self.edge_hidden_size,
            noise_channels=self.noise_channels,
            mesh_splits=self.mesh_splits,
            attention_k_hop=self.attention_k_hop,
            attention_bandwidth=self.attention_bandwidth,
            num_grid_to_mesh_edges=self.num_grid_to_mesh_edges,
            grid_latitudes=self.grid_latitudes,
            grid_longitudes=self.grid_longitudes,
            pressure_levels=self.pressure_levels,
            aggregate_normalization=None,
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
        result = model(grid_features=grid_features, global_features=global_features, noise=noise)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, config.num_grid_points, self.hidden_size)
        )
        self.parent.assertEqual(
            result.mesh_hidden_state.shape, (self.batch_size, config.num_mesh_nodes, self.hidden_size)
        )

    def get_feature_extractor(self):
        """A feature extractor describing exactly the variables the config expects, with trivial statistics."""
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
        )

    def prepare_state(self, extractor, batch_size, seed=0):
        """A physical atmospheric state, shaped the way the model documentation describes it."""
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

    def prepare_config_and_inputs_for_common(self):
        config, grid_features, global_features, noise = self.prepare_config_and_inputs()
        return config, {
            "grid_features": grid_features,
            "global_features": global_features,
            "noise": noise,
        }


@require_torch
class WeatherNext2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (WeatherNext2Model, WeatherNext2ForWeatherForecasting) if is_torch_available() else ()
    additional_model_inputs = ["global_features", "noise"]

    test_resize_embeddings = False
    # The persistent geometry tensors live on the base model rather than on independently
    # dispatchable submodules, so Accelerate cannot split them across CPU/disk and an accelerator.
    test_cpu_offload = False
    test_disk_offload_bin = False
    test_disk_offload_safetensors = False

    def setUp(self):
        self.model_tester = WeatherNext2ModelTester(self)
        # WeatherNext 2 has no text modality, so there is no vocabulary to check for.
        self.config_tester = ConfigTester(self, config_class=WeatherNext2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_shapes(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = WeatherNext2ForWeatherForecasting(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(
            outputs.prediction.shape,
            (self.model_tester.batch_size, config.num_output_channels, config.grid_latitudes, config.grid_longitudes),
        )
        self.assertTrue(torch.isfinite(outputs.prediction).all())

    def test_noise_drives_the_ensemble(self):
        """Two members that share inputs but not noise must differ; two that share both must not."""
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

    def test_batched_rollout_matches_running_members_alone(self):
        """The batch axis carries ensemble members, so a batched rollout must equal member-by-member runs.

        This walks the whole documented loop - feature extractor, model, `postprocess`, `advance_state` - because
        a batch axis that only works in the model is not much use.
        """
        config = self.model_tester.get_config()
        extractor = self.model_tester.get_feature_extractor()
        model = WeatherNext2ForWeatherForecasting(config).to(torch_device).eval()

        batch_size = 3
        state = self.model_tester.prepare_state(extractor, batch_size)
        valid_time = np.full(batch_size, np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64))
        noise = torch.randn(batch_size, config.noise_channels, device=torch_device)

        for _ in range(2):
            inputs = extractor(state, seconds_since_epoch=valid_time).to(torch_device)
            with torch.no_grad():
                prediction = model(**inputs, noise=noise).prediction
            forecast = extractor.postprocess(prediction, state)
            valid_time = valid_time + extractor.time_step_hours * 3600
            state = extractor.advance_state(state, forecast, valid_time)

        for member in range(batch_size):
            solo_state = {
                name: values if name in extractor.static_variables else values[member : member + 1]
                for name, values in self.model_tester.prepare_state(extractor, batch_size).items()
            }
            solo_time = np.full(1, np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64))
            for _ in range(2):
                inputs = extractor(solo_state, seconds_since_epoch=solo_time).to(torch_device)
                with torch.no_grad():
                    prediction = model(**inputs, noise=noise[member : member + 1]).prediction
                forecast = extractor.postprocess(prediction, solo_state)
                solo_time = solo_time + extractor.time_step_hours * 3600
                solo_state = extractor.advance_state(solo_state, forecast, solo_time)

            for name, values in solo_state.items():
                if name in extractor.static_variables:
                    continue
                np.testing.assert_allclose(values[0], state[name][member], rtol=1e-4, atol=1e-4)

    def test_channel_layout_matches_projection_shapes(self):
        config = self.model_tester.get_config()
        model = WeatherNext2ForWeatherForecasting(config)
        self.assertEqual(model.model.grid_encoder.in_proj.in_features, config.num_grid_input_channels)
        self.assertEqual(model.model.mesh_encoder.in_proj.in_features, config.num_mesh_input_channels)
        self.assertEqual(model.output_proj.out_features, config.num_output_channels)
        self.assertEqual(
            sum(levels for _, _, levels in config.input_channel_layout) + 3, config.num_grid_input_channels
        )

    def test_attention_outputs(self):
        """Same contract as the shared test, with this model's block-local attention shape.

        Attention runs over three block-diagonals of the mesh adjacency, so a layer's weights are
        `[batch * num_blocks, heads, block_size, 3 * block_size]` rather than `[batch, heads, seq, seq]`.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            num_blocks, _, block_size, key_length = model.get_submodule(
                "model" if model_class is not WeatherNext2Model else ""
            ).attention_mask.shape
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

    # WeatherNext 2 consumes gridded physical fields, not tokens or images, and masks by mesh
    # adjacency rather than by sequence position, so several of the shared tests do not apply.
    @unittest.skip(reason="WeatherNext 2 has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Hidden states are per grid point and per mesh node, not per token.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Hidden states are per grid point and per mesh node, not per token.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="The attention mask is geometric and internal, so external masks do not apply.")
    def test_eager_matches_sdpa_inference(self, *args, **kwargs):
        pass

    @unittest.skip(
        reason="The shared test calls `state_dict()` on the state dict it already has whenever a buffer is a "
        "`BoolTensor`, which the banded attention mask is."
    )
    def test_torch_save_load(self):
        pass


@require_torch
class WeatherNext2GeometryTest(unittest.TestCase):
    """Checks the things that only hold for a real mesh, on the tiny checkpoint.

    A model built from a config has no geometry, only a placeholder, so none of this can be tested
    there. The geometry itself is built by the conversion script; what these check is the shape of
    what it produced and that the model's banded attention reads it correctly.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = WeatherNext2ForWeatherForecasting.from_pretrained(TINY_CHECKPOINT).to(torch_device).eval()

    def test_graphs_are_shaped_the_way_the_model_reads_them(self):
        model, config = self.model.model, self.model.config
        self.assertEqual(config.num_mesh_nodes, 10 * 4**2 + 2)
        # Mesh-to-grid connects every grid point to the three vertices of one face.
        self.assertEqual(len(model.mesh_to_grid_senders), 3 * config.num_grid_points)
        # Edges are sorted by receiver so the aggregation can use a segmented sum.
        self.assertTrue((model.mesh_to_grid_receivers.diff() >= 0).all())
        self.assertTrue((model.grid_to_mesh_receivers.diff() >= 0).all())
        # Indices stay inside the node sets they point at.
        self.assertLess(int(model.grid_to_mesh_senders.max()), config.num_grid_points)
        self.assertLess(int(model.mesh_to_grid_senders.max()), config.num_mesh_nodes)
        # Edge features are normalized to the unit interval.
        self.assertLessEqual(float(model.mesh_to_grid_edge_features[:, 0].max()), 1.0 + 1e-6)
        self.assertLessEqual(float(model.mesh_to_grid_edge_features[:, 1:].abs().max()), 1.0 + 1e-6)

    def test_every_mesh_node_attends_and_the_padding_does_not(self):
        model = self.model.model
        num_nodes = self.model.config.num_mesh_nodes
        reaches = model.attention_mask.any(-1).reshape(-1)
        self.assertTrue(reaches[:num_nodes].all(), "a mesh node attends to nothing, which is a NaN row")
        self.assertFalse(reaches[num_nodes:].any(), "the padding past the last mesh node is attended to")

    def test_attention_backends_agree(self):
        """`eager` and `sdpa` must produce the same forecast.

        The tiny checkpoint has head dimension 8, while flex attention requires at least 16. The
        shared flex-attention test exercises that backend after increasing the test head dimension.
        """
        config = self.model.config
        inputs = {
            "grid_features": floats_tensor(
                [2, config.num_grid_input_channels - 3, config.grid_latitudes, config.grid_longitudes]
            ).to(torch_device),
            "global_features": floats_tensor([2, config.num_mesh_input_channels - 3]).to(torch_device),
            "noise": floats_tensor([2, config.noise_channels]).to(torch_device),
        }
        reference = None
        for implementation in ("eager", "sdpa"):
            model = WeatherNext2ForWeatherForecasting.from_pretrained(
                TINY_CHECKPOINT, attn_implementation=implementation
            )
            model.to(torch_device).eval()
            with torch.no_grad():
                prediction = model(**inputs).prediction
            if reference is None:
                reference = prediction
            else:
                torch.testing.assert_close(prediction, reference, atol=1e-5, rtol=1e-5)

    def test_the_forecast_reaches_every_parameter_that_shapes_it(self):
        """Backward through a forecast, to catch anything accidentally cut out of the graph.

        This needs the real geometry: with the placeholder, the edge features are zero and the edge
        encoders would look dead when they are not.

        The six weights of the decoder's mesh-node update are expected to get nothing. That block
        updates the mesh nodes as well as the grid points, and the head reads only the grid, so they
        cannot move. The original implementation has the same shape, and names the value it drops
        `unused_updated_latent_mesh_data`.
        """
        model = WeatherNext2ForWeatherForecasting.from_pretrained(TINY_CHECKPOINT).to(torch_device).train()
        config = model.config
        prediction = model(
            grid_features=floats_tensor(
                [2, config.num_grid_input_channels - 3, config.grid_latitudes, config.grid_longitudes]
            ).to(torch_device),
            global_features=floats_tensor([2, config.num_mesh_input_channels - 3]).to(torch_device),
            noise=floats_tensor([2, config.noise_channels]).to(torch_device),
        ).prediction
        prediction.square().mean().backward()

        starved = {name for name, parameter in model.named_parameters() if parameter.grad is None}
        expected = {
            name for name, _ in model.named_parameters() if name.startswith("model.mesh_to_grid.mesh_node_update.")
        }
        self.assertEqual(starved, expected)
        self.assertTrue(expected, "the decoder's mesh-node update went away; this test needs rethinking")

    def test_banded_attention_matches_dense_masking(self):
        """The three-block-diagonal attention must equal masking the full node-by-node matrix."""
        model = self.model.model
        config = self.model.config
        attention: WeatherNext2Attention = model.mesh_transformer.layers[0].self_attn

        banded_mask = model.attention_mask
        num_blocks, _, block_size, _ = banded_mask.shape
        padded = num_blocks * block_size
        hidden_states = floats_tensor([1, padded, config.hidden_size]).to(torch_device)

        with torch.no_grad():
            banded, _ = attention(hidden_states.view(1, num_blocks, block_size, -1), banded_mask)
            banded = banded.reshape(1, padded, -1)

            # Reassemble the equivalent dense mask from the banded one.
            dense_mask = torch.zeros(padded, padded, dtype=torch.bool, device=torch_device)
            for block in range(num_blocks):
                for offset, position in ((-1, 0), (0, 1), (1, 2)):
                    neighbour = block + offset
                    if 0 <= neighbour < num_blocks:
                        dense_mask[
                            block * block_size : (block + 1) * block_size,
                            neighbour * block_size : (neighbour + 1) * block_size,
                        ] = banded_mask[block, 0, :, position * block_size : (position + 1) * block_size]

            head_shape = (1, padded, config.num_attention_heads, attention.head_dim)
            query = attention.q_proj(hidden_states).view(head_shape).transpose(1, 2)
            key = attention.k_proj(hidden_states).view(head_shape).transpose(1, 2)
            value = attention.v_proj(hidden_states).view(head_shape).transpose(1, 2)
            dense = torch.nn.functional.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), attn_mask=dense_mask, scale=attention.scaling
            )
            dense = attention.o_proj(dense.transpose(1, 2).reshape(1, padded, -1))

        torch.testing.assert_close(banded, dense, atol=1e-4, rtol=1e-4)


@require_torch
@slow
class WeatherNext2ModelIntegrationTest(unittest.TestCase):
    """End-to-end check against the released 1 degree Mini checkpoint.

    Marked slow: it downloads ~230 MB of weights, the mesh and both graphs among them.
    """

    checkpoint = "kashif/weathernext2-mini"

    def test_inference_shapes_and_determinism(self):
        model = WeatherNext2ForWeatherForecasting.from_pretrained(self.checkpoint).to(torch_device).eval()
        processor = WeatherNext2FeatureExtractor.from_pretrained(self.checkpoint)
        config = model.config

        grid_features = torch.zeros(
            1,
            config.num_grid_input_channels - 3,
            config.grid_latitudes,
            config.grid_longitudes,
            device=torch_device,
        )
        global_features = torch.zeros(1, config.num_mesh_input_channels - 3, device=torch_device)
        noise = torch.zeros(1, config.noise_channels, device=torch_device)

        with torch.no_grad():
            first = model(grid_features=grid_features, global_features=global_features, noise=noise).prediction
            second = model(grid_features=grid_features, global_features=global_features, noise=noise).prediction

        self.assertEqual(first.shape, (1, config.num_output_channels, config.grid_latitudes, config.grid_longitudes))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second)
        self.assertEqual(len(processor.target_variables), len(config.target_variables))
