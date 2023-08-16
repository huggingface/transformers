from typing import Any, Dict, List, Mapping

import numpy as np
import torch


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(item, keep_features=True):
    # convert to dgl graphs first
    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "node_feat" in item.keys():  # input_nodes
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


class TransformerMDataCollator:
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if self.on_the_fly_processing:
            features = [preprocess_item(i) for i in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}

        num_graphs = len(features)
        graphs = [f["graph"] for f in features]
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # +1 is for the virtual node.
        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -th.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
        )

        for i in range(num_graphs):
            # A binary mask where invalid positions are indicated by True.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(
                th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # Path padding to make all paths to the same length "max_len".
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graphs[i].edata["feat"] + 1
            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = th.cat(
                (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                labels = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
            else:  # binary classification
                labels = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
        else:  # multi task classification, left to float to keep the NaNs
            labels = torch.from_numpy(np.stack([i["labels"] for i in features], axis=0))
        
        return {
            "attn_mask": attn_mask,
            "node_feat": node_feat,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "path_data": path_data,
            "dist": dist,
            "labels": labels,
        }
