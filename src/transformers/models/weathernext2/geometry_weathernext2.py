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
"""Static geometry for WeatherNext 2, used when converting a checkpoint.

WeatherNext 2 has no learned positional encodings: every position-dependent quantity is a
deterministic function of the icosahedral mesh and of the lat/lon grid. None of it is learned and
none of it ever changes, so it is computed once here, at conversion time, and written into the
checkpoint as buffers. The modeling code only ever loads it, and so needs neither scipy nor trimesh.

The construction mirrors ``weathernext/utils/icosahedral_mesh.py`` and
``weathernext/utils/model_utils.py`` from https://github.com/google-deepmind/weathernext, and must
stay bit-compatible with them: the reverse Cuthill-McKee permutation in particular decides the node
ordering that the attention mask is built from, and the triangle lookup is trimesh's, whose
tie-breaking is not reproduced by a hand-written one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import trimesh
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


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
