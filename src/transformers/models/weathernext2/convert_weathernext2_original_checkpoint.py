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
"""Converts an original WeatherNext 2 checkpoint to the Transformers format.

The upstream checkpoints are Haiku parameter trees saved as `.npz`, published alongside a Fiddle
JSON config that carries both the architecture hyper-parameters and the normalization statistics.
Both live in a public bucket:

```bash
BASE=https://storage.googleapis.com/dm_graphcast/weathernext2
curl -o params.npz  "$BASE/params/WeatherNextCyclones_Mini_%3C2024.npz"
curl -o config.json "$BASE/configs/WeatherNextCyclones_Mini.json"   # or take it from the repo

python src/transformers/models/weathernext2/convert_weathernext2_original_checkpoint.py \
    --checkpoint_path params.npz \
    --fiddle_config_path config.json \
    --output_dir weathernext2-mini
```

Two structural differences are worth knowing about:

* Haiku stores linear weights as `[in, out]`; `nn.Linear` wants `[out, in]`, so every weight is
  transposed.
* The encoders and decoder keep one weight array *per input variable*, so a variable can be added or
  removed at fine-tuning time without disturbing the rest. Mathematically this is a single matmul
  over the concatenated inputs, which is what we store: the converter stacks the per-variable arrays
  in the canonical channel order defined by [`WeatherNext2Config.input_channel_layout`].
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import trimesh
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from transformers.models.weathernext2.configuration_weathernext2 import WeatherNext2Config
from transformers.models.weathernext2.feature_extraction_weathernext2 import WeatherNext2FeatureExtractor
from transformers.models.weathernext2.modeling_weathernext2 import WeatherNext2ForWeatherForecasting


PARAM_PREFIX = "params:multimodality_forward/"
SPATIAL_NODE_FEATURES = "sin_lat,sin_lon,cos_lon"
SPATIAL_EDGE_FEATURES = "distance,rel_x,rel_y,rel_z"


# ============================================================================================================
# Mesh geometry
#
# Static geometry for WeatherNext 2, used when converting a checkpoint.
#
# WeatherNext 2 has no learned positional encodings: every position-dependent quantity is a
# deterministic function of the icosahedral mesh and of the lat/lon grid. None of it is learned and
# none of it ever changes, so it is computed once here, at conversion time, and written into the
# checkpoint as buffers. The modeling code only ever loads it, and so needs neither scipy nor trimesh.
#
# The construction mirrors ``weathernext/utils/icosahedral_mesh.py`` and
# ``weathernext/utils/model_utils.py`` from https://github.com/google-deepmind/weathernext, and must
# stay bit-compatible with them: the reverse Cuthill-McKee permutation in particular decides the node
# ordering that the attention mask is built from, and the triangle lookup is trimesh's, whose
# tie-breaking is not reproduced by a hand-written one.
# ============================================================================================================


def lat_lon_deg_to_spherical(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in degrees -> (phi=azimuth, theta=polar) in radians."""
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90.0 - lat)
    return phi, theta


def spherical_to_cartesian(phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):
        theta = np.arccos(np.clip(z, -1.0, 1.0))
    return phi, theta


def lat_lon_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Returns unit-sphere cartesian positions stacked on the last axis."""
    return np.stack(spherical_to_cartesian(*lat_lon_deg_to_spherical(lat, lon)), axis=-1)


def cartesian_to_lat_lon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi, theta = cartesian_to_spherical(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    return 90.0 - np.rad2deg(theta), np.mod(np.rad2deg(phi), 360.0)


def get_icosahedron(pole_parallel_faces: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Regular icosahedron inscribed in the unit sphere, faces counter-clockwise from outside."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in (1.0, -1.0):
        for c2 in (phi, -phi):
            vertices.append((c1, c2, 0.0))
            vertices.append((0.0, c1, c2))
            vertices.append((c2, 0.0, c1))
    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    faces = np.array(
        [
            (0, 1, 2),
            (0, 6, 1),
            (8, 0, 2),
            (8, 4, 0),
            (3, 8, 2),
            (3, 2, 7),
            (7, 2, 1),
            (0, 4, 6),
            (4, 11, 6),
            (6, 11, 5),
            (1, 5, 7),
            (4, 10, 11),
            (4, 8, 10),
            (10, 8, 3),
            (10, 3, 9),
            (11, 10, 9),
            (11, 9, 5),
            (5, 9, 7),
            (9, 3, 7),
            (1, 6, 5),
        ],
        dtype=np.int32,
    )

    if pole_parallel_faces:
        # Rotate so the top/bottom faces are parallel to the X-Y plane, which keeps mesh nodes off
        # the exact poles.
        angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
        rotation_angle = (np.pi - angle_between_faces) / 2
        rotation_matrix = Rotation.from_euler(seq="y", angles=rotation_angle).as_matrix()
        vertices = np.dot(vertices, rotation_matrix)

    return vertices.astype(np.float32), faces


class _ChildVerticesBuilder:
    """Deduplicates the edge-midpoint vertices created when splitting faces."""

    def __init__(self, parent_vertices: np.ndarray):
        self._index_mapping: dict[tuple[int, ...], int] = {}
        self._parent_vertices = parent_vertices
        self._all_vertices = list(parent_vertices)

    def get_new_child_vertex_index(self, parent_vertex_indices) -> int:
        key = tuple(sorted(parent_vertex_indices))
        if key not in self._index_mapping:
            position = self._parent_vertices[list(parent_vertex_indices)].mean(0)
            position = position / np.linalg.norm(position)
            self._index_mapping[key] = len(self._all_vertices)
            self._all_vertices.append(position)
        return self._index_mapping[key]

    def get_all_vertices(self) -> np.ndarray:
        return np.array(self._all_vertices)


def _two_split_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits every triangle into 4, keeping the counter-clockwise orientation."""
    builder = _ChildVerticesBuilder(vertices)
    new_faces = []
    for ind1, ind2, ind3 in faces:
        ind12 = builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = builder.get_new_child_vertex_index((ind3, ind1))
        new_faces.extend([[ind1, ind12, ind31], [ind12, ind2, ind23], [ind31, ind23, ind3], [ind12, ind23, ind31]])
    return builder.get_all_vertices(), np.array(new_faces, dtype=np.int32)


def get_triangular_mesh(splits: int, pole_parallel_faces: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Icosahedron refined `splits` times. `splits=6` gives 40962 vertices / 81920 faces."""
    vertices, faces = get_icosahedron(pole_parallel_faces=pole_parallel_faces)
    for _ in range(splits):
        vertices, faces = _two_split_faces(vertices, faces)
    return vertices, faces


def faces_to_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turns each triangle [a, b, c] into the directed edges a->b, b->c, c->a."""
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def get_permutation_to_banded(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Reverse Cuthill-McKee ordering that makes the mesh adjacency banded.

    The banded structure is what makes local (k-hop) attention expressible as a few dense blocks
    around the diagonal. The algorithm is deterministic for a given adjacency matrix.
    """
    num_nodes = vertices.shape[0]
    senders, receivers = faces_to_edges(faces)
    adjacency = sp.csr_matrix((np.ones(len(senders)), (senders, receivers)), shape=(num_nodes, num_nodes))
    return reverse_cuthill_mckee(adjacency, symmetric_mode=True)


def get_khop_adjacency(faces: np.ndarray, num_nodes: int, k_hop: int) -> sp.csr_matrix:
    """Boolean matrix that is True where two mesh nodes are within `k_hop` mesh edges.

    Self-edges are added before raising to the power, so the result includes every node within
    *at most* `k_hop` hops (and the node itself).
    """
    senders, receivers = faces_to_edges(faces)
    indices = np.concatenate([senders, np.arange(num_nodes)])
    values = np.concatenate([receivers, np.arange(num_nodes)])
    adjacency = sp.csr_matrix((np.ones(len(indices), dtype=bool), (indices, values)), shape=(num_nodes, num_nodes))
    mask = adjacency
    for _ in range(k_hop - 1):
        mask = mask @ adjacency
    mask.data = np.ones_like(mask.data, dtype=bool)
    return mask.tocsr()


def get_mask_bandwidth(mask: sp.spmatrix) -> int:
    """Half-width of the band, i.e. the block size that makes the mask tri-block-diagonal."""
    coo = mask.tocoo()
    return int(np.abs(coo.row.astype(np.int64) - coo.col.astype(np.int64)).max()) + 1


def max_mesh_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    senders, receivers = faces_to_edges(faces)
    return float(np.linalg.norm(vertices[senders] - vertices[receivers], axis=-1).max())


def ball_query_edges(grid_xyz: np.ndarray, mesh_xyz: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Connect every grid point to all mesh nodes within `radius` (euclidean, unit sphere)."""
    tree = cKDTree(mesh_xyz)
    neighbours = tree.query_ball_point(x=grid_xyz, r=radius)
    grid_indices = np.repeat(np.arange(len(neighbours)), [len(n) for n in neighbours])
    mesh_indices = np.concatenate([np.sort(n) for n in neighbours]) if len(grid_indices) else np.empty(0)
    return grid_indices.astype(np.int64), mesh_indices.astype(np.int64)


def in_triangle_edges(
    grid_xyz: np.ndarray, mesh_xyz: np.ndarray, mesh_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Connect every grid point to the 3 vertices of its nearest mesh face.

    "Nearest" is the euclidean-closest point on the triangulated surface. Grid points that fall
    exactly on a shared mesh edge are equidistant from both adjacent faces, and which one is picked
    is decided by the order the query returns rather than by geometry, so this calls trimesh exactly
    as the original implementation does. A different tie-break changes the forecast locally by up to
    about 1 K in 2m temperature.
    """
    mesh = trimesh.Trimesh(vertices=mesh_xyz, faces=mesh_faces, process=False)
    _, _, face_indices = mesh.nearest.on_surface(grid_xyz)

    num_points = grid_xyz.shape[0]
    mesh_indices = mesh_faces[face_indices].reshape(-1).astype(np.int64)
    grid_indices = np.repeat(np.arange(num_points), 3).astype(np.int64)
    return grid_indices, mesh_indices


def get_spatial_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """`[sin_lat, sin_lon, cos_lon]`, the only node features the model gets from geometry."""
    return np.stack([np.sin(np.deg2rad(lat)), np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon))], axis=-1)


def get_rotation_matrices_to_local_coordinates(reference_phi: np.ndarray, reference_theta: np.ndarray) -> np.ndarray:
    """Rotations taking each receiver to (lat=0, lon=0), used to express edges in a local frame."""
    azimuthal_rotation = -reference_phi
    polar_rotation = -reference_theta + np.pi / 2
    return Rotation.from_euler("zy", np.stack([azimuthal_rotation, polar_rotation], axis=1)).as_matrix()


def get_edge_features(
    sender_lat: np.ndarray,
    sender_lon: np.ndarray,
    receiver_lat: np.ndarray,
    receiver_lon: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    edge_normalization_factor: float | None = None,
) -> np.ndarray:
    """`[distance, rel_x, rel_y, rel_z]` per edge, in the receiver's local frame.

    Positions are rotated so the receiver sits at (lat=0, lon=0), the sender position is taken
    relative to it, and everything is divided by the longest edge in this edge set so distances land
    in [0, 1] and offsets in [-1, 1].
    """
    sender_phi, sender_theta = lat_lon_deg_to_spherical(sender_lat, sender_lon)
    receiver_phi, receiver_theta = lat_lon_deg_to_spherical(receiver_lat, receiver_lon)

    sender_pos = np.stack(spherical_to_cartesian(sender_phi, sender_theta), axis=-1)
    receiver_pos = np.stack(spherical_to_cartesian(receiver_phi, receiver_theta), axis=-1)

    rotations = get_rotation_matrices_to_local_coordinates(receiver_phi, receiver_theta)
    edge_rotations = rotations[receiver_indices]

    def rotate(matrices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        # Upstream uses "...ji,...i->...j", i.e. the transposed rotation.
        return np.einsum("...ji,...i->...j", matrices, positions)

    relative_position = rotate(edge_rotations, sender_pos[sender_indices]) - rotate(
        edge_rotations, receiver_pos[receiver_indices]
    )

    distances = np.linalg.norm(relative_position, axis=-1, keepdims=True)
    if edge_normalization_factor is None:
        edge_normalization_factor = float(distances.max())
    distances = distances / edge_normalization_factor
    relative_position = relative_position / edge_normalization_factor

    return np.concatenate([distances, relative_position], axis=-1)


@dataclass
class WeatherNext2Geometry:
    """Everything position-dependent, computed once per (mesh, grid) pair."""

    mesh_lat: np.ndarray  # [num_mesh_nodes]
    mesh_lon: np.ndarray  # [num_mesh_nodes]
    mesh_faces: np.ndarray  # [num_faces, 3], in the permuted node indexing
    mesh_spatial_features: np.ndarray  # [num_mesh_nodes, 3]
    grid_spatial_features: np.ndarray  # [num_grid_points, 3]
    attention_bandwidth: int
    attention_mask: sp.csr_matrix  # [num_mesh_nodes, num_mesh_nodes] bool
    grid_to_mesh_senders: np.ndarray  # grid point index, sorted by receiver
    grid_to_mesh_receivers: np.ndarray  # mesh node index
    grid_to_mesh_edge_features: np.ndarray  # [num_edges, 4]
    mesh_to_grid_senders: np.ndarray  # mesh node index, sorted by receiver
    mesh_to_grid_receivers: np.ndarray  # grid point index
    mesh_to_grid_edge_features: np.ndarray  # [num_edges, 4]

    @property
    def num_mesh_nodes(self) -> int:
        return len(self.mesh_lat)


def _sort_by_receiver(senders: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upstream sorts edges by receiver so the scatter-add can use a sorted segment sum."""
    order = np.argsort(receivers, kind="stable")
    return senders[order], receivers[order]


def build_geometry(
    mesh_splits: int,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    attention_k_hop: int,
    ball_query_radius_fraction: float,
    grid_major_axis: str = "lat",
) -> WeatherNext2Geometry:
    """Builds the mesh, the local-attention mask and both bipartite graphs.

    `grid_major_axis="lat"` flattens the grid with longitude varying fastest, matching upstream's
    default.
    """
    if grid_major_axis not in ("lat", "lon"):
        raise ValueError(f"`grid_major_axis` must be 'lat' or 'lon', got {grid_major_axis}.")

    vertices, faces = get_triangular_mesh(mesh_splits)
    permutation = get_permutation_to_banded(vertices, faces)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(len(permutation))

    vertices = vertices[permutation]
    faces = inverse_permutation[faces]

    mesh_lat, mesh_lon = cartesian_to_lat_lon(vertices)
    num_mesh_nodes = vertices.shape[0]

    attention_mask = get_khop_adjacency(faces, num_mesh_nodes, attention_k_hop)

    # Flatten the grid.
    if grid_major_axis == "lat":
        flat_lat = np.repeat(grid_lat, len(grid_lon))
        flat_lon = np.tile(grid_lon, len(grid_lat))
    else:
        flat_lat = np.tile(grid_lat, len(grid_lon))
        flat_lon = np.repeat(grid_lon, len(grid_lat))
    grid_xyz = lat_lon_to_cartesian(flat_lat, flat_lon)

    radius = ball_query_radius_fraction * max_mesh_edge_length(vertices, faces)
    g2m_senders, g2m_receivers = ball_query_edges(grid_xyz, vertices, radius)
    g2m_senders, g2m_receivers = _sort_by_receiver(g2m_senders, g2m_receivers)
    g2m_edge_features = get_edge_features(flat_lat, flat_lon, mesh_lat, mesh_lon, g2m_senders, g2m_receivers)

    m2g_grid_indices, m2g_mesh_indices = in_triangle_edges(grid_xyz, vertices, faces)
    m2g_senders, m2g_receivers = _sort_by_receiver(m2g_mesh_indices, m2g_grid_indices)
    m2g_edge_features = get_edge_features(mesh_lat, mesh_lon, flat_lat, flat_lon, m2g_senders, m2g_receivers)

    return WeatherNext2Geometry(
        mesh_lat=mesh_lat,
        mesh_lon=mesh_lon,
        mesh_faces=faces,
        mesh_spatial_features=get_spatial_features(mesh_lat, mesh_lon).astype(np.float32),
        grid_spatial_features=get_spatial_features(flat_lat, flat_lon).astype(np.float32),
        attention_bandwidth=get_mask_bandwidth(attention_mask),
        attention_mask=attention_mask,
        grid_to_mesh_senders=g2m_senders,
        grid_to_mesh_receivers=g2m_receivers,
        grid_to_mesh_edge_features=g2m_edge_features.astype(np.float32),
        mesh_to_grid_senders=m2g_senders,
        mesh_to_grid_receivers=m2g_receivers,
        mesh_to_grid_edge_features=m2g_edge_features.astype(np.float32),
    )


def build_banded_attention_mask(geometry: WeatherNext2Geometry) -> torch.Tensor:
    """Rewrites the k-hop mask as `[num_blocks, 1, block_size, 3 * block_size]`.

    After the reverse Cuthill-McKee permutation every non-zero of the mask lies within
    `geometry.attention_bandwidth` of the diagonal, so a block of that many consecutive nodes can
    only reach itself and its two neighbours. This is the form the model stores and attends with.
    """
    block_size = min(geometry.attention_bandwidth, geometry.num_mesh_nodes)
    num_blocks = -(-geometry.num_mesh_nodes // block_size)
    padded_size = num_blocks * block_size

    mask = geometry.attention_mask.tocsr()
    mask = sp.vstack([mask, sp.csr_matrix((padded_size - mask.shape[0], mask.shape[1]), dtype=bool)])
    mask = sp.hstack([mask, sp.csr_matrix((padded_size, padded_size - mask.shape[1]), dtype=bool)]).tocsr()

    blocks = np.zeros((num_blocks, 1, block_size, 3 * block_size), dtype=bool)
    for block in range(num_blocks):
        rows = slice(block * block_size, (block + 1) * block_size)
        for offset, position in ((-1, 0), (0, 1), (1, 2)):
            neighbour = block + offset
            if 0 <= neighbour < num_blocks:
                columns = slice(neighbour * block_size, (neighbour + 1) * block_size)
                target = slice(position * block_size, (position + 1) * block_size)
                blocks[block, 0, :, target] = mask[rows, columns].toarray()
    return torch.from_numpy(blocks)


# ============================================================================================================
# Checkpoint conversion
# ============================================================================================================


def load_fiddle_config(path: str) -> dict[str, Any]:
    """Materializes a Fiddle JSON graph into plain Python containers."""
    document = json.load(open(path))
    objects = document["objects"]

    def field_name(key: str) -> str:
        match = re.search(r"name='([^']*)'", key) or re.search(r"key='([^']*)'", key) or re.search(r"index=(\d+)", key)
        return match.group(1) if match else key

    def resolve(node):
        kind = node.get("type")
        if kind == "leaf":
            return node.get("value")
        if kind == "pyref":
            return f"{node['module']}.{node['name']}"
        if kind != "ref":
            return None
        obj = objects[node["key"]]
        items = obj.get("items")
        if items is None:
            return None
        obj_type = obj.get("type")
        if isinstance(obj_type, dict) and obj_type.get("name") in ("tuple", "list"):
            return [resolve(value) for _, value in items]
        return {field_name(key): resolve(value) for key, value in items}

    return resolve(document["root"])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def config_from_fiddle(fiddle: dict[str, Any], grid_latitudes: int, grid_longitudes: int) -> WeatherNext2Config:
    task = fiddle["task"]
    architecture = fiddle["predictor_kwargs"]["noisy_function_kwargs"]
    transformer = architecture["mesh_model_ctor"]["transformer_kwargs"]
    latent = architecture["latent_dense_kwargs"]

    shifted = {}
    for variable, (function, kwargs) in (architecture.get("per_var_activation_fns") or {}).items():
        if not function.endswith("shifted_activation") or not kwargs["activation_fn"].endswith("sigmoid"):
            raise ValueError(f"Unsupported output activation for {variable!r}: {function}")
        shifted[variable] = -float(kwargs["input_offset"])

    input_duration_hours = int(re.fullmatch(r"(\d+)h", task["input_duration"]).group(1))
    time_step_hours = 6

    return WeatherNext2Config(
        hidden_size=transformer["d_model"],
        intermediate_size=transformer["ffw_hidden"],
        num_hidden_layers=transformer["num_layers"],
        num_attention_heads=transformer["num_heads"],
        edge_hidden_size=architecture["points_to_mesh_model_ctor"]["edge_encoder_dense_kwargs"]["output_size"],
        noise_channels=architecture["norm_conditioning_latent_dense_kwargs"]["output_size"],
        mesh_splits=architecture["mesh_num_splits"],
        attention_k_hop=transformer["attention_k_hop"],
        ball_query_radius_fraction=architecture["points_to_mesh_model_ctor"]["ball_query_radius_fraction"],
        aggregate_normalization=_optional_float(
            architecture["points_to_mesh_model_ctor"]["deep_gnn_kwargs"].get("aggregate_normalization")
        ),
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        input_variables=tuple(task["input_variables"]),
        target_variables=tuple(task["target_variables"]),
        forcing_variables=tuple(task["forcing_variables"]),
        pressure_levels=tuple(task["pressure_levels"]),
        num_input_timesteps=input_duration_hours // time_step_hours,
        time_step_hours=time_step_hours,
        sigmoid_shifted_outputs=shifted,
        hidden_act="gelu_pytorch_tanh",
        mlp_act="silu" if latent["activation"] == "swish" else latent["activation"],
    )


def statistics_from_fiddle(fiddle: dict[str, Any], name: str) -> dict[str, Any]:
    data_vars = fiddle["predictor_wrappers"][0]["kwargs"][name]["data"]["data_vars"]
    return {variable: entry["data"] for variable, entry in data_vars.items()}


def nan_fill_values_from_fiddle(fiddle: dict[str, Any]) -> dict[str, float]:
    """Value the original `NaNCleaner` wrapper substitutes for missing data, per variable.

    This is not the variable's mean: `sea_surface_temperature` is filled with a fixed temperature
    that normalizes to roughly -1.5, and the wrapper runs before normalization.
    """
    fill_values = {}
    for wrapper in fiddle["predictor_wrappers"]:
        if wrapper["constructor"].endswith("NaNCleaner"):
            variable = wrapper["kwargs"]["var_to_clean"]
            data_vars = wrapper["kwargs"]["fill_value"]["data"]["data_vars"]
            fill_values[variable] = float(data_vars[variable]["data"])
    return fill_values


def split_weight_name(config: WeatherNext2Config, variable: str, time_offset: int | None, prefix: str) -> str:
    """Rebuilds the per-variable weight name used by the original `xarray_dense` encoders.

    The name records the coordinates the array covers, e.g.
    `w_input_temperature_level=50,...,1000_time=-21600`: pressure levels first (they stay inside one
    array), then the time slice (which gets its own array).
    """
    name = f"w_{prefix}{variable}"
    if variable in config.atmospheric_variables:
        name += "_level=" + ",".join(str(level) for level in config.pressure_levels)
    if time_offset is not None:
        name += f"_time={time_offset * 3600}"
    return name


def stacked_input_weight(
    params: dict[str, np.ndarray],
    config: WeatherNext2Config,
    module: str,
    layout: list[tuple[str, int | None, int]],
    spatial_features: str,
) -> np.ndarray:
    """Concatenates the split first matmul into one `[out, in]` weight."""
    parts = [params[f"{module}/split_input_matmul:w_spatial_feature={spatial_features}"]]
    for variable, time_offset, _ in layout:
        prefix = "forcing_" if time_offset is not None and time_offset > 0 else "input_"
        parts.append(params[f"{module}/split_input_matmul:{split_weight_name(config, variable, time_offset, prefix)}"])
    return np.concatenate(parts, axis=0).T


def stacked_output_weight(
    params: dict[str, np.ndarray], config: WeatherNext2Config, module: str
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates the split output linear into one `[out, in]` weight and `[out]` bias."""
    weights, biases = [], []
    for variable, time_offset, _ in config.target_channel_layout:
        name = split_weight_name(config, variable, time_offset, prefix="")
        weights.append(params[f"{module}/split_output_linear:{name}"])
        biases.append(params[f"{module}/split_output_linear:{name.replace('w_', 'b_', 1)}"])
    return np.concatenate(weights, axis=1).T, np.concatenate(biases, axis=0)


def convert_state_dict(params: dict[str, np.ndarray], config: WeatherNext2Config) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(name: str) -> np.ndarray:
        consumed.add(name)
        return params[name]

    def put(target: str, value: np.ndarray) -> None:
        state_dict[target] = torch.from_numpy(np.ascontiguousarray(value))

    def convert_conditioned_mlp(target: str, module: str) -> None:
        """The `Linear -> act -> Linear -> LayerNorm -> FiLM` block, minus its first weight."""
        put(f"{target}.in_proj.bias", take(f"{module}/shared_dense/mlp/linear_0:b"))
        put(f"{target}.out_proj.weight", take(f"{module}/shared_dense/mlp/linear_1:w").T)
        put(f"{target}.out_proj.bias", take(f"{module}/shared_dense/mlp/linear_1:b"))
        film = f"{module}/shared_dense/normalization/linear_norm_conditioning/linear"
        put(f"{target}.norm.film.linear.weight", take(f"{film}:w").T)
        put(f"{target}.norm.film.linear.bias", take(f"{film}:b"))

    # --- noise, encoders
    put(
        "model.noise_encoder.weight",
        take(
            "global_norm_conditioning_encoder/split_input_matmul:"
            f"w_input_noise_noise_channels=range({config.noise_channels})"
        ).T,
    )

    for target, module, layout in (
        ("model.grid_encoder", "grid_encoder", config.input_channel_layout),
        ("model.mesh_encoder", "mesh_encoder", config.mesh_channel_layout),
    ):
        put(f"{target}.in_proj.weight", stacked_input_weight(params, config, module, layout, SPATIAL_NODE_FEATURES))
        for variable, time_offset, _ in layout:
            prefix = "forcing_" if time_offset is not None and time_offset > 0 else "input_"
            consumed.add(f"{module}/split_input_matmul:{split_weight_name(config, variable, time_offset, prefix)}")
        consumed.add(f"{module}/split_input_matmul:w_spatial_feature={SPATIAL_NODE_FEATURES}")
        convert_conditioned_mlp(target, module)

    # --- graph networks
    for target, module, edge_set, receiver in (
        ("model.grid_to_mesh", "grid_to_mesh_gnn", "points_to_mesh_nodes", "mesh"),
        ("model.mesh_to_grid", "mesh_to_grid_gnn", "mesh_to_points_nodes", "point"),
    ):
        edge_module = f"{module}/edge_encoder"
        put(
            f"{target}.edge_encoder.in_proj.weight",
            take(f"{edge_module}/split_input_matmul:w_spatial_feature={SPATIAL_EDGE_FEATURES}").T,
        )
        convert_conditioned_mlp(f"{target}.edge_encoder", edge_module)

        gnn = f"{module}/deep_gnn"
        put(f"{target}.edge_update.edge_proj.weight", take(f"{gnn}/processor_edges_0_edge_{edge_set}:w").T)
        put(f"{target}.edge_update.sender_proj.weight", take(f"{gnn}/processor_edges_0_sender_{edge_set}:w").T)
        if receiver == "point":
            # Only the mesh-to-grid direction folds the receiver's own features into the message.
            put(
                f"{target}.edge_update.receiver_proj.weight",
                take(f"{gnn}/processor_edges_0_receiver_{edge_set}:w").T,
            )
        edge_update = f"{gnn}/processor_edges_0_{edge_set}"
        put(f"{target}.edge_update.bias", take(f"{edge_update}/mlp/linear_0:b"))
        put(f"{target}.edge_update.out_proj.weight", take(f"{edge_update}/mlp/linear_1:w").T)
        put(f"{target}.edge_update.out_proj.bias", take(f"{edge_update}/mlp/linear_1:b"))
        film = f"{edge_update}/normalization/linear_norm_conditioning/linear"
        put(f"{target}.edge_update.norm.film.linear.weight", take(f"{film}:w").T)
        put(f"{target}.edge_update.norm.film.linear.bias", take(f"{film}:b"))

        for node_target, node_set in (("mesh_node_update", "mesh_nodes"), ("grid_node_update", "point_nodes")):
            node_module = f"{gnn}/processor_nodes_0_{node_set}"
            put(f"{target}.{node_target}.in_proj.weight", take(f"{node_module}/mlp/linear_0:w").T)
            put(f"{target}.{node_target}.in_proj.bias", take(f"{node_module}/mlp/linear_0:b"))
            put(f"{target}.{node_target}.out_proj.weight", take(f"{node_module}/mlp/linear_1:w").T)
            put(f"{target}.{node_target}.out_proj.bias", take(f"{node_module}/mlp/linear_1:b"))
            film = f"{node_module}/normalization/linear_norm_conditioning/linear"
            put(f"{target}.{node_target}.norm.film.linear.weight", take(f"{film}:w").T)
            put(f"{target}.{node_target}.norm.film.linear.bias", take(f"{film}:b"))

    # --- mesh transformer
    for layer_idx in range(config.num_hidden_layers):
        block = f"mesh_transformer/transformer/block_{layer_idx:02d}"
        target = f"model.mesh_transformer.layers.{layer_idx}"
        for projection in ("q", "k", "v"):
            put(f"{target}.self_attn.{projection}_proj.weight", take(f"{block}/mha_proj_{projection}:w").T)
        put(f"{target}.self_attn.o_proj.weight", take(f"{block}/mha_final:w").T)
        put(f"{target}.self_attn.o_proj.bias", take(f"{block}/mha_final:b"))
        put(f"{target}.mlp.fc1.weight", take(f"{block}/ffw_up:w").T)
        put(f"{target}.mlp.fc1.bias", take(f"{block}/ffw_up:b"))
        put(f"{target}.mlp.fc2.weight", take(f"{block}/ffw_down:w").T)
        put(f"{target}.mlp.fc2.bias", take(f"{block}/ffw_down:b"))
        # Haiku names the two FiLM layers of a block by call order, so the second gets a `_1` suffix.
        for norm, suffix in (("input_layernorm", ""), ("post_attention_layernorm", "_1")):
            film = f"{block}/block_{layer_idx:02d}_norm_conditioning{suffix}/linear"
            put(f"{target}.{norm}.film.linear.weight", take(f"{film}:w").T)
            put(f"{target}.{norm}.film.linear.bias", take(f"{film}:b"))

    film = "mesh_transformer/transformer/transformer_final_norm_conditioning/linear"
    put("model.mesh_transformer.norm.film.linear.weight", take(f"{film}:w").T)
    put("model.mesh_transformer.norm.film.linear.bias", take(f"{film}:b"))

    # --- decoder
    put("decoder_proj.weight", take("grid_decoder/shared_dense/mlp/linear_0:w").T)
    put("decoder_proj.bias", take("grid_decoder/shared_dense/mlp/linear_0:b"))
    output_weight, output_bias = stacked_output_weight(params, config, "grid_decoder")
    put("output_proj.weight", output_weight)
    put("output_proj.bias", output_bias)
    for variable, time_offset, _ in config.target_channel_layout:
        name = split_weight_name(config, variable, time_offset, prefix="")
        consumed.add(f"grid_decoder/split_output_linear:{name}")
        consumed.add(f"grid_decoder/split_output_linear:{name.replace('w_', 'b_', 1)}")

    unconsumed = sorted(set(params) - consumed)
    if unconsumed:
        raise ValueError(f"{len(unconsumed)} checkpoint parameters were not converted, e.g. {unconsumed[:5]}")
    return state_dict


def geometry_state_dict(config: WeatherNext2Config) -> dict[str, torch.Tensor]:
    """Builds the mesh and both bipartite graphs, and records the two sizes that follow from them.

    This is the only place the geometry is ever derived. It needs scipy and trimesh, takes about two
    minutes at 0.25 degrees, and its result is written into the checkpoint so that loading a model
    never has to repeat it.
    """
    geometry = build_geometry(
        mesh_splits=config.mesh_splits,
        grid_lat=np.linspace(-90.0, 90.0, config.grid_latitudes),
        grid_lon=np.arange(config.grid_longitudes) * (360.0 / config.grid_longitudes),
        attention_k_hop=config.attention_k_hop,
        ball_query_radius_fraction=config.ball_query_radius_fraction,
    )
    config.num_grid_to_mesh_edges = int(geometry.grid_to_mesh_senders.shape[0])
    config.attention_bandwidth = int(geometry.attention_bandwidth)

    tensors = {
        "grid_spatial_features": geometry.grid_spatial_features,
        "mesh_spatial_features": geometry.mesh_spatial_features,
        "grid_to_mesh_senders": geometry.grid_to_mesh_senders,
        "grid_to_mesh_receivers": geometry.grid_to_mesh_receivers,
        "grid_to_mesh_edge_features": geometry.grid_to_mesh_edge_features,
        "mesh_to_grid_senders": geometry.mesh_to_grid_senders,
        "mesh_to_grid_receivers": geometry.mesh_to_grid_receivers,
        "mesh_to_grid_edge_features": geometry.mesh_to_grid_edge_features,
        "attention_mask": build_banded_attention_mask(geometry),
    }
    return {f"model.{name}": torch.as_tensor(value) for name, value in tensors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Original `.npz` parameter file.")
    parser.add_argument("--fiddle_config_path", required=True, help="Matching Fiddle JSON config.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model.")
    parser.add_argument(
        "--grid_latitudes", type=int, default=None, help="Defaults to 181 for the 1 degree mini model."
    )
    parser.add_argument("--grid_longitudes", type=int, default=None)
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repository id.")
    args = parser.parse_args()

    archive = np.load(args.checkpoint_path, allow_pickle=True)
    params = {key[len(PARAM_PREFIX) :]: archive[key] for key in archive.files if key.startswith(PARAM_PREFIX)}
    print(f"Loaded {len(params)} parameter arrays ({sum(v.size for v in params.values()) / 1e6:.1f}M values).")

    fiddle = load_fiddle_config(args.fiddle_config_path)
    # The mini model is trained at 1 degree, the rest at 0.25.
    is_mini = fiddle["predictor_kwargs"]["noisy_function_kwargs"]["mesh_num_splits"] < 6
    grid_latitudes = args.grid_latitudes or (181 if is_mini else 721)
    grid_longitudes = args.grid_longitudes or (360 if is_mini else 1440)

    config = config_from_fiddle(fiddle, grid_latitudes, grid_longitudes)
    state_dict = convert_state_dict(params, config)
    # Sets `num_grid_to_mesh_edges` and `attention_bandwidth`, so build it before the model.
    state_dict.update(geometry_state_dict(config))

    model = WeatherNext2ForWeatherForecasting(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    print(f"Converted {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters.")

    processor = WeatherNext2FeatureExtractor(
        input_variables=config.input_variables,
        target_variables=config.target_variables,
        forcing_variables=config.forcing_variables,
        atmospheric_variables=config.atmospheric_variables,
        static_variables=config.static_variables,
        global_variables=config.global_variables,
        pressure_levels=config.pressure_levels,
        mean_by_level=statistics_from_fiddle(fiddle, "mean_by_level"),
        stddev_by_level=statistics_from_fiddle(fiddle, "stddev_by_level"),
        diffs_stddev_by_level=statistics_from_fiddle(fiddle, "diffs_stddev_by_level"),
        nan_fill_values=nan_fill_values_from_fiddle(fiddle),
        num_input_timesteps=config.num_input_timesteps,
        time_step_hours=config.time_step_hours,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
    )

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}.")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
