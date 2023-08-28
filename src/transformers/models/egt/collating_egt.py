from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from ...utils import is_dgl_available, requires_backends


if is_dgl_available():
    import dgl


def convert_to_single_node_emb(x, offset: int = 128):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def convert_to_single_edge_emb(x, offset: int = 8):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(item, keep_features=True):
    requires_backends(preprocess_item, ["dgl"])
    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "node_feat" in item.keys():  # input_nodes
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    else:
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = convert_to_single_node_emb(node_feature)
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_edge_emb(edge_attr)

    # convert to dgl graph for computing shortest path distance and svd encodings
    g = dgl.graph((edge_index[0], edge_index[1]))
    shortest_path_result = dgl.shortest_dist(g)
    shortest_path_result = torch.where(shortest_path_result == -1, 510, shortest_path_result)
    svd_pe = dgl.svd_pe(g, k=8, padding=True, random_flip=True)

    # combine
    item["input_nodes"] = input_nodes
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result
    item["svd_pe"] = svd_pe
    if "labels" not in item:
        item["labels"] = item["y"]

    return item


class EGTDataCollator:
    def __init__(self, on_the_fly_processing=False):
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if self.on_the_fly_processing:
            features = [preprocess_item(i) for i in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}

        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        svd_pe_size = len(features[0]["svd_pe"][0]) // 2
        batch_size = len(features)

        batch["featm"] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["dm"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["node_feat"] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["svd_pe"] = torch.zeros(batch_size, max_node_num, svd_pe_size * 2, dtype=torch.float)
        batch["attn_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        for ix, f in enumerate(features):
            for k in ["attn_edge_type", "spatial_pos", "input_nodes", "svd_pe"]:
                f[k] = torch.tensor(f[k])

            batch["featm"][ix, : f["attn_edge_type"].shape[0], : f["attn_edge_type"].shape[1], :] = f["attn_edge_type"]
            batch["dm"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]
            batch["node_feat"][ix, : f["input_nodes"].shape[0], :] = f["input_nodes"]
            batch["svd_pe"][ix, : f["svd_pe"].shape[0], :] = f["svd_pe"]
            batch["attn_mask"][ix, : f["svd_pe"].shape[0]] = 1

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
            else:  # binary classification
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = torch.from_numpy(np.stack([i["labels"] for i in features], dim=0))

        return batch
