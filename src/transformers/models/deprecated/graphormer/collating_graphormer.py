# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ....utils import is_cython_available, requires_backends


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(item, keep_features=True):
    requires_backends(preprocess_item, ["cython"])

    if keep_features and "edge_attr" in item:  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "node_feat" in item:  # input_nodes
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    else:
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # node adj matrix [num_nodes, num_nodes] bool
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)

    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

    # combine
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # we shift all indices by one for padding
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # we shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # for undirected graph
    item["input_edges"] = input_edges + 1  # we shift all indices by one for padding
    if "labels" not in item:
        item["labels"] = item["y"]

    return item


class GraphormerDataCollator:
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        if self.on_the_fly_processing:
            features = [preprocess_item(i) for i in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}

        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        edge_input_size = len(features[0]["input_edges"][0][0][0])
        batch_size = len(features)

        batch["attn_bias"] = torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch["attn_edge_type"] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["spatial_pos"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["in_degree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["input_edges"] = torch.zeros(
            batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long
        )

        for ix, f in enumerate(features):
            for k in ["attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "input_nodes", "input_edges"]:
                f[k] = torch.tensor(f[k])

            if len(f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max]) > 0:
                f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max] = float("-inf")

            batch["attn_bias"][ix, : f["attn_bias"].shape[0], : f["attn_bias"].shape[1]] = f["attn_bias"]
            batch["attn_edge_type"][ix, : f["attn_edge_type"].shape[0], : f["attn_edge_type"].shape[1], :] = f[
                "attn_edge_type"
            ]
            batch["spatial_pos"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]
            batch["in_degree"][ix, : f["in_degree"].shape[0]] = f["in_degree"]
            batch["input_nodes"][ix, : f["input_nodes"].shape[0], :] = f["input_nodes"]
            batch["input_edges"][
                ix, : f["input_edges"].shape[0], : f["input_edges"].shape[1], : f["input_edges"].shape[2], :
            ] = f["input_edges"]

        batch["out_degree"] = batch["in_degree"]

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
            else:  # binary classification
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = torch.from_numpy(np.stack([i["labels"] for i in features], axis=0))

        return batch
