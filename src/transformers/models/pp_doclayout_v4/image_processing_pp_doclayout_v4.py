# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

from collections import defaultdict
from typing import ClassVar

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...utils import auto_docstring, is_scipy_available, requires_backends
from ...utils.generic import TensorType


if is_scipy_available():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components


@auto_docstring
class PPDocLayoutV4ImageProcessor(TorchvisionBackend):
    r"""
    Constructs a PP-DocLayoutV4 image processor.

    Images are resized to a fixed 800x800 square with bicubic interpolation and rescaled to `[0, 1]` without further
    normalization, matching the reference `cv2.resize` based preprocessing.

    Unlike the usual `resize` -> `rescale` -> `normalize` order, this processor rescales *before* resizing. The
    reference preprocessing resizes `uint8` with `cv2.resize`, which rounds once; resizing an integer tensor with
    torchvision rounds a second time, and the two roundings compound to ~22/255 on high contrast edges -- enough to
    permute the predicted reading order.

    Post-processing differs from [`PPDocLayoutV3ImageProcessor`], because PP-DocLayoutV4 regresses a four point
    quadrilateral per query instead of predicting a segmentation mask, and emits raw relative/successor order logits
    instead of a decoded reading order.
    """

    resample = PILImageResampling.BICUBIC
    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    size = {"height": 800, "width": 800}
    do_resize = True
    do_rescale = True
    do_normalize = True

    _quad_num_coords: ClassVar[int] = 10

    # We require `self.resize(..., antialias=False)` to approximate the output of `cv2.resize`
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if do_resize:
            # The bicubic overshoot has to be clipped, the way the reference `cv2.resize` bounds the ringing with its
            # saturating cast back to `uint8`: by the range the incoming pixels live in, not by the range of this
            # particular image. `cv2.resize` does not saturate floating point input, so a float tensor is only bounded
            # by 1 when it really is the unit interval that the `do_rescale=False` contract documents.
            is_unit_interval = all(image.is_floating_point() for image in images) and (
                float(max(image.amax() for image in images)) <= 1.0
            )
            upper_bound = (1.0 if is_unit_interval else 255.0) * (rescale_factor if do_rescale else 1.0)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images.to(dtype=torch.float32), rescale_factor)
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample, antialias=False)
                stacked_images = stacked_images.clamp(0, upper_bound)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # `do_rescale` is already applied above, only the normalization is left.
            stacked_images = self.rescale_and_normalize(
                stacked_images, False, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes=None,
    ):
        """
        Converts the raw output of [`PPDocLayoutV4ForObjectDetection`] into final quadrilaterals, enclosing boxes in
        `(top_left_x, top_left_y, bottom_right_x, bottom_right_y)` format and reading order ranks. Only supports
        PyTorch.

        Args:
            outputs ([`PPDocLayoutV4ForObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`(height, width)`) with the target size of each
                image in the batch.

        Returns:
            `list[Dict]`: A list of dictionaries, one per image, each containing the `scores`, `labels`, `boxes`,
            `polygon_points` and `order_seq` predicted by the model. Predictions are sorted by reading order, so
            `order_seq` is non-decreasing (a query selected under several labels keeps a single rank).
        """
        requires_backends(self, ["torch", "scipy"])
        if target_sizes is None:
            raise ValueError("`target_sizes` is required to map the predictions back to the original image size.")

        logits = outputs.logits
        relative_order_logits = outputs.relative_order_logits
        successor_order_logits = outputs.successor_order_logits
        corners = self._quad_to_corners(outputs.pred_boxes)

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        target_sizes = torch.as_tensor(target_sizes, device=logits.device)
        image_heights, image_widths = target_sizes.unbind(1)
        scale = torch.stack([image_widths, image_heights], dim=1).to(corners.dtype)
        corners = corners * scale[:, None, None, :]

        # Flattened top-k over query x class, so one query may show up under several labels. `query_index` keeps the
        # original query so the pairwise order matrices can still be indexed.
        num_top_queries, num_classes = logits.shape[1], logits.shape[2]
        scores = logits.sigmoid()
        scores, flat_index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
        labels = flat_index % num_classes
        query_index = flat_index // num_classes
        corners = corners.gather(
            dim=1, index=query_index[..., None, None].expand(-1, -1, corners.shape[-2], corners.shape[-1])
        )

        results = []
        for image_scores, image_labels, image_corners, image_queries, relative, successor in zip(
            scores, labels, corners, query_index, relative_order_logits, successor_order_logits
        ):
            keep = image_scores >= threshold
            image_scores, image_labels = image_scores[keep], image_labels[keep]
            image_corners, image_queries = image_corners[keep], image_queries[keep]

            unique_queries, inverse = torch.unique(image_queries, return_inverse=True)
            submatrix = np.ix_(unique_queries.cpu().numpy(), unique_queries.cpu().numpy())
            ranks = self._decode_reading_order(
                relative.detach().float().cpu().numpy()[submatrix],
                successor.detach().float().cpu().numpy()[submatrix],
            )
            order_seq = torch.as_tensor(ranks, device=image_scores.device)[inverse]
            order_seq, sorted_index = torch.sort(order_seq, stable=True)

            image_corners = image_corners[sorted_index]
            boxes = torch.cat([image_corners.amin(dim=-2), image_corners.amax(dim=-2)], dim=-1)
            results.append(
                {
                    "scores": image_scores[sorted_index],
                    "labels": image_labels[sorted_index],
                    "boxes": boxes,
                    "polygon_points": image_corners,
                    "order_seq": order_seq,
                }
            )

        return results

    def _quad_to_corners(self, pred_boxes: "torch.Tensor") -> "torch.Tensor":
        """
        Converts the box parameterization predicted by the bbox heads into normalized corner coordinates.

        Args:
            pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
                Boxes as `[center_x, center_y, dx1, dy1, ..., dx4, dy4]` in sigmoid space, with the corner offsets
                shifted by `+0.5`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, 4, 2)`: The four corners in top-left, top-right,
            bottom-right, bottom-left order, normalized to `[0, 1]`.
        """
        num_coords = pred_boxes.shape[-1]
        if num_coords != self._quad_num_coords:
            raise ValueError(
                f"Unsupported num_coords: {num_coords}. PP-DocLayoutV4 only supports quads ({self._quad_num_coords})."
            )
        centers = pred_boxes[..., :2].unsqueeze(-2)
        offsets = pred_boxes[..., 2:].reshape(*pred_boxes.shape[:-1], 4, 2) - 0.5
        return centers + offsets

    def _to_sparse_graph(self, num_nodes, edges):
        """Builds the `scipy.sparse` adjacency matrix of a graph given as an iterable of `(row, column)` pairs."""
        rows = np.fromiter((edge[0] for edge in edges), dtype=np.int64, count=len(edges))
        columns = np.fromiter((edge[1] for edge in edges), dtype=np.int64, count=len(edges))
        data = np.ones(len(edges), dtype=np.int8)
        return coo_matrix((data, (rows, columns)), shape=(num_nodes, num_nodes), dtype=np.int8)

    def _remove_cycles(self, num_nodes, edges):
        """
        Drops the lowest confidence edge of every cyclic component until the graph is a DAG.

        A strongly connected component of more than one node is exactly a set of nodes that lie on a common cycle, so
        every edge inside one is a cycle edge and dropping the weakest of them always breaks at least one cycle. The
        graph is a DAG once every strongly connected component is a single node.
        """
        edges = dict(edges)
        while True:
            _, labels = connected_components(
                self._to_sparse_graph(num_nodes, edges), directed=True, connection="strong"
            )
            # Self loops are excluded from `edges`, so a shared label already implies a component of several nodes.
            weakest = {}
            for edge, confidence in edges.items():
                component = labels[edge[0]]
                if component != labels[edge[1]]:
                    continue
                # Ties keep the first edge in insertion order, which keeps the decode deterministic.
                if component not in weakest or confidence < edges[weakest[component]]:
                    weakest[component] = edge
            if not weakest:
                return list(edges)
            for edge in weakest.values():
                del edges[edge]

    def _find_connected_components(self, num_nodes, edges):
        """Groups nodes into the connected components of the undirected view of the DAG."""
        num_components, labels = connected_components(self._to_sparse_graph(num_nodes, edges), directed=False)
        components = [[] for _ in range(num_components)]
        for node, label in enumerate(labels):
            components[label].append(node)
        return components

    def _topological_sort(self, num_nodes, edges, relative_scores):
        """
        Sorts a DAG topologically, breaking ties with a Borda count over the relative order scores.

        Args:
            num_nodes (`int`):
                Number of nodes in the (sub)graph.
            edges (`list[tuple[int, int]]`):
                DAG edges as `(predecessor, successor)` pairs.
            relative_scores (`np.ndarray` of shape `(num_nodes, num_nodes)`):
                Pairwise relative order scores, where a high `relative_scores[i, j]` means that `i` is likely to be
                read before `j`.

        Returns:
            `list[int]`: Node indices from earliest to latest.
        """
        in_degree = [0] * num_nodes
        graph = defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
            in_degree[j] += 1

        candidates = [node for node in range(num_nodes) if in_degree[node] == 0]
        order = []
        while candidates:
            if len(candidates) == 1:
                best = candidates[0]
            else:
                best, best_score = None, -1.0
                for candidate in candidates:
                    score = sum(relative_scores[candidate][other] for other in candidates if other != candidate)
                    # Ties are broken by the smaller index, which keeps the decode deterministic.
                    if score > best_score or (score == best_score and (best is None or candidate < best)):
                        best_score = score
                        best = candidate
            candidates.remove(best)
            order.append(best)
            for neighbor in graph[best]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidates.append(neighbor)

        # A cycle that survived `_remove_cycles` would strand nodes, so append whatever is left.
        if len(order) < num_nodes:
            missing = set(order)
            order.extend(node for node in range(num_nodes) if node not in missing)
        return order

    def _decode_reading_order(self, relative_logits, successor_logits):
        """
        Decodes the reading order of a single image from the two order heads.

        The successor (ROOR) logits define a soft "directly precedes" graph that is turned into a DAG, split into
        connected components and sorted topologically. The relative order logits break ties inside a component and
        order the components relative to each other.

        Args:
            relative_logits (`np.ndarray` of shape `(num_boxes, num_boxes)`):
                Relative order logits restricted to the kept boxes.
            successor_logits (`np.ndarray` of shape `(num_boxes, num_boxes)`):
                Successor order logits restricted to the kept boxes.

        Returns:
            `np.ndarray` of shape `(num_boxes,)`: The 0-based reading order rank of every box.
        """
        num_boxes = successor_logits.shape[0]
        if num_boxes <= 1:
            return np.zeros(num_boxes, dtype=np.int64)

        edges = {
            (i, j): float(successor_logits[i][j])
            for i in range(num_boxes)
            for j in range(num_boxes)
            if i != j and successor_logits[i][j] > 0
        }
        dag_edges = self._remove_cycles(num_boxes, edges)
        components = self._find_connected_components(num_boxes, dag_edges)

        # Numerically stable sigmoid: the successor head masks its diagonal with -1e4, which overflows `exp(-x)`.
        relative_scores = np.where(
            relative_logits >= 0,
            1.0 / (1.0 + np.exp(-np.abs(relative_logits))),
            np.exp(-np.abs(relative_logits)) / (1.0 + np.exp(-np.abs(relative_logits))),
        )
        np.fill_diagonal(relative_scores, 0.0)
        node_votes = relative_scores.sum(axis=0)

        component_orders = []
        for component in components:
            if len(component) == 1:
                component_orders.append(list(component))
                continue
            nodes = sorted(component)
            local_index = {node: local for local, node in enumerate(nodes)}
            local_edges = [
                (local_index[i], local_index[j]) for i, j in dag_edges if i in local_index and j in local_index
            ]
            local_scores = relative_scores[np.ix_(nodes, nodes)]
            local_order = self._topological_sort(len(nodes), local_edges, local_scores)
            component_orders.append([nodes[local] for local in local_order])

        # Components are laid out by their mean relative-order vote, i.e. earliest reading component first.
        component_orders.sort(key=lambda component: float(np.mean(node_votes[component])))

        ranks = np.zeros(num_boxes, dtype=np.int64)
        for rank, node in enumerate(node for component in component_orders for node in component):
            ranks[node] = rank
        return ranks


__all__ = ["PPDocLayoutV4ImageProcessor"]
