# Copyright 2026 Google DeepMind and The HuggingFace Inc. team. All rights reserved.
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
import unittest
from unittest.mock import patch

import numpy as np

from transformers import WeatherNext2Config
from transformers.testing_utils import require_scipy, require_torch
from transformers.utils import is_scipy_available, is_torch_available


geometry_dependencies_available = (
    importlib.util.find_spec("trimesh") is not None and importlib.util.find_spec("rtree") is not None
)
if is_torch_available() and is_scipy_available() and geometry_dependencies_available:
    from transformers.models.weathernext2 import convert_weathernext2_original_checkpoint as converter


@require_torch
@require_scipy
@unittest.skipUnless(geometry_dependencies_available, "Geometry conversion requires trimesh and rtree")
class WeatherNext2ConversionTest(unittest.TestCase):
    def test_connectivity_uses_lat_lon_roundtrip(self):
        grid_lat = np.linspace(-90.0, 90.0, 13, dtype=np.float32)
        grid_lon = np.arange(24, dtype=np.float32) * 15.0
        with (
            patch.object(converter, "ball_query_edges", wraps=converter.ball_query_edges) as ball_query,
            patch.object(converter, "in_triangle_edges", wraps=converter.in_triangle_edges) as triangle_query,
        ):
            geometry = converter.build_geometry(2, grid_lat, grid_lon, 2, 0.6)

        mesh_xyz = converter.lat_lon_to_cartesian(geometry.mesh_lat, geometry.mesh_lon)
        np.testing.assert_array_equal(ball_query.call_args.args[1], mesh_xyz)
        np.testing.assert_array_equal(triangle_query.call_args.args[1], mesh_xyz)
        self.assertEqual(
            ball_query.call_args.args[2], 0.6 * converter.max_mesh_edge_length(mesh_xyz, geometry.mesh_faces)
        )

    def test_edge_order_matches_upstream(self):
        senders = np.arange(64)
        receivers = np.tile(np.arange(4), 16)
        sorted_senders, sorted_receivers = converter._sort_by_receiver(senders, receivers)
        order = np.argsort(receivers)
        np.testing.assert_array_equal(sorted_senders, senders[order])
        np.testing.assert_array_equal(sorted_receivers, receivers[order])

    def test_geometry_preserves_grid_coordinates(self):
        config = WeatherNext2Config(mesh_splits=2, grid_latitudes=13, grid_longitudes=24, attention_k_hop=2)
        grid_lat = np.linspace(-90.0, 90.0, 13, dtype=np.float64)
        grid_lon = np.arange(24, dtype=np.float64) * 15.0
        with patch.object(converter, "build_geometry", wraps=converter.build_geometry) as build:
            converter.geometry_state_dict(config, grid_lat, grid_lon)
            self.assertIs(build.call_args.kwargs["grid_lat"], grid_lat)
            self.assertIs(build.call_args.kwargs["grid_lon"], grid_lon)
            converter.geometry_state_dict(config)
            self.assertEqual(build.call_args.kwargs["grid_lat"].dtype, np.float32)
            self.assertEqual(build.call_args.kwargs["grid_lon"].dtype, np.float32)
