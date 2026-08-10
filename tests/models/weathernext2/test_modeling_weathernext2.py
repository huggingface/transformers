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

from transformers import WeatherNext2Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2Model


class WeatherNext2ModelTester:
    """Builds a model small enough to test: a twice-refined icosahedron (162 nodes) on a 10 degree grid."""

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
        grid_latitudes=19,
        grid_longitudes=36,
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

    def prepare_config_and_inputs_for_common(self):
        config, grid_features, global_features, noise = self.prepare_config_and_inputs()
        return config, {
            "grid_features": grid_features,
            "global_features": global_features,
            "noise": noise,
        }


def floats_tensor(shape):
    return torch.randn(*shape, device=torch_device, dtype=torch.float32)


@require_torch
class WeatherNext2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (WeatherNext2Model, WeatherNext2ForWeatherForecasting) if is_torch_available() else ()
    # There is no `weather-forecasting` pipeline yet; the inputs are global gridded states rather than
    # anything the generic pipeline machinery handles.
    pipeline_model_mapping = {}

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_inputs_embeds = False
    test_torchscript = False
    is_encoder_decoder = False

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

    def test_channel_layout_matches_projection_shapes(self):
        config = self.model_tester.get_config()
        model = WeatherNext2ForWeatherForecasting(config)
        self.assertEqual(model.model.grid_encoder.in_proj.in_features, config.num_grid_input_channels)
        self.assertEqual(model.model.mesh_encoder.in_proj.in_features, config.num_mesh_input_channels)
        self.assertEqual(model.output_proj.out_features, config.num_output_channels)
        self.assertEqual(
            sum(levels for _, _, levels in config.input_channel_layout) + 3, config.num_grid_input_channels
        )

    def test_attention_backends_agree(self):
        """`eager`, `sdpa` and `flex_attention` must all produce the same forecast."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        reference, state_dict = None, None
        for implementation in ("eager", "sdpa", "flex_attention"):
            model = WeatherNext2ForWeatherForecasting._from_config(config, attn_implementation=implementation)
            if state_dict is None:
                state_dict = model.state_dict()
            else:
                model.load_state_dict(state_dict)
            model.to(torch_device).eval()
            with torch.no_grad():
                prediction = model(**inputs).prediction
            if reference is None:
                reference = prediction
            else:
                torch.testing.assert_close(prediction, reference, atol=1e-5, rtol=1e-5)

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

    def test_banded_attention_matches_dense_masking(self):
        """The three-block-diagonal attention must equal masking the full node-by-node matrix."""
        from transformers.models.weathernext2.modeling_weathernext2 import WeatherNext2Attention

        config = self.model_tester.get_config()
        model = WeatherNext2Model(config).to(torch_device).eval()
        attention: WeatherNext2Attention = model.mesh_transformer.layers[0].self_attn

        banded_mask = model.attention_mask
        num_blocks, _, block_size, _ = banded_mask.shape
        padded = num_blocks * block_size
        hidden_states = floats_tensor([1, padded, config.hidden_size])

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

    # WeatherNext 2 consumes gridded physical fields, not tokens or images, and masks by mesh
    # adjacency rather than by sequence position, so several of the shared tests do not apply.
    @unittest.skip(reason="WeatherNext 2 has no token embeddings.")
    def test_resize_tokens_embeddings(self):
        pass

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


@require_torch
class WeatherNext2GeometryTest(unittest.TestCase):
    def test_mesh_sizes_and_banded_mask(self):
        from transformers.models.weathernext2.geometry_weathernext2 import build_geometry

        geometry = build_geometry(
            mesh_splits=2,
            grid_lat=np.linspace(-90, 90, 19),
            grid_lon=np.arange(36) * 10.0,
            attention_k_hop=2,
            ball_query_radius_fraction=0.6,
        )
        self.assertEqual(geometry.num_mesh_nodes, 10 * 4**2 + 2)
        # Mesh-to-grid connects every grid point to the three vertices of one face.
        self.assertEqual(len(geometry.mesh_to_grid_senders), 19 * 36 * 3)
        # Edges are sorted by receiver so the aggregation can use a segmented sum.
        self.assertTrue(np.all(np.diff(geometry.mesh_to_grid_receivers) >= 0))
        self.assertTrue(np.all(np.diff(geometry.grid_to_mesh_receivers) >= 0))
        # Every non-zero of the mask lies inside the band.
        coo = geometry.attention_mask.tocoo()
        self.assertLessEqual(np.abs(coo.row - coo.col).max() + 1, geometry.attention_bandwidth)
        # Edge features are normalized to the unit interval.
        self.assertLessEqual(geometry.mesh_to_grid_edge_features[:, 0].max(), 1.0 + 1e-6)
        self.assertLessEqual(np.abs(geometry.mesh_to_grid_edge_features[:, 1:]).max(), 1.0 + 1e-6)

    def test_geometry_is_deterministic(self):
        from transformers.models.weathernext2.geometry_weathernext2 import build_geometry

        kwargs = {
            "mesh_splits": 1,
            "grid_lat": np.linspace(-90, 90, 13),
            "grid_lon": np.arange(24) * 15.0,
            "attention_k_hop": 2,
            "ball_query_radius_fraction": 0.6,
        }
        first, second = build_geometry(**kwargs), build_geometry(**kwargs)
        np.testing.assert_array_equal(first.mesh_lat, second.mesh_lat)
        np.testing.assert_array_equal(first.grid_to_mesh_senders, second.grid_to_mesh_senders)
        np.testing.assert_array_equal(first.mesh_to_grid_senders, second.mesh_to_grid_senders)


@require_torch
@slow
class WeatherNext2ModelIntegrationTest(unittest.TestCase):
    """End-to-end check against the released 1 degree Mini checkpoint.

    Marked slow: it downloads ~230 MB of weights and builds the icosahedral mesh on first run.
    """

    checkpoint = "kashif/weathernext2-mini"

    def test_inference_shapes_and_determinism(self):
        from transformers import WeatherNext2FeatureExtractor

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
