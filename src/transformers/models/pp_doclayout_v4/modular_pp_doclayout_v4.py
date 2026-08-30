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

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging, requires_backends
from ...utils.generic import TensorType, can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..pp_doclayout_v3.image_processing_pp_doclayout_v3 import PPDocLayoutV3ImageProcessor
from ..pp_doclayout_v3.modeling_pp_doclayout_v3 import (
    PPDocLayoutV3Decoder,
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3MLPPredictionHead,
    PPDocLayoutV3Model,
    PPDocLayoutV3MultiscaleDeformableAttention,
    PPDocLayoutV3PreTrainedModel,
)
from ..rt_detr.modeling_rt_detr import (
    RTDetrHybridEncoder,
    get_contrastive_denoising_training_group,
    inverse_sigmoid,
)


logger = logging.get_logger(__name__)

# `[center_x, center_y]` plus four `(dx, dy)` corner offsets.
QUAD_NUM_COORDS = 10


@auto_docstring(checkpoint="PaddlePaddle/PP-DocLayoutV4_safetensors")
@strict
class PPDocLayoutV4Config(PreTrainedConfig):
    r"""
    initializer_bias_prior_prob (`float`, *optional*):
        The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
        If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
    freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
        Whether to freeze the batch normalization layers in the backbone.
    encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
        Multi level features input for encoder.
    feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`):
        Strides used in each feature map.
    encode_proj_layers (`list[int]`, *optional*, defaults to `[2]`):
        Indexes of the projected layers to be used in the encoder.
    positional_encoding_temperature (`int`, *optional*, defaults to 10000):
        The temperature parameter used to create the positional encodings.
    encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    eval_size (`tuple[int, int]`, *optional*):
        Height and width used to computes the effective height and width of the position embeddings after taking
        into account the stride. Must be left as `None` to reproduce the reference implementation, which computes the
        position embeddings dynamically.
    normalize_before (`bool`, *optional*, defaults to `False`):
        Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
        feed-forward modules.
    hidden_expansion (`float`, *optional*, defaults to 1.0):
        Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
    hidden_size (`int`, *optional*, defaults to 256):
        Dimension of the decoder layers, excluding the hybrid encoder. Also readable as `d_model`, the name used by
        the RT-DETR lineage this model descends from.
    label_noise_ratio (`float`, *optional*, defaults to 0.5):
        The fraction of denoising labels to which random noise should be added.
    box_noise_scale (`float`, *optional*, defaults to 1.0):
        Scale or magnitude of noise to be added to the bounding boxes.
    num_queries (`int`, *optional*, defaults to 300):
        Number of object queries.
    decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
        Multi level features dimension for decoder
    decoder_ffn_dim (`int`, *optional*, defaults to 1024):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    num_feature_levels (`int`, *optional*, defaults to 3):
        The number of input feature levels.
    decoder_n_points (`int`, *optional*, defaults to 4):
        The number of sampled keys in each feature level for each attention head in the decoder.
    decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    num_denoising (`int`, *optional*, defaults to 100):
        The total number of denoising tasks or queries to be used for contrastive denoising.
    learn_initial_query (`bool`, *optional*, defaults to `False`):
        Indicates whether the initial query embeddings for the decoder should be learned during training
    anchor_image_size (`tuple[int, int]`, *optional*):
        Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
    disable_custom_kernels (`bool`, *optional*, defaults to `True`):
        Whether to disable custom kernels.
    num_coords (`int`, *optional*, defaults to 10):
        Size of the box parameterization predicted by the bbox heads. PP-DocLayoutV4 regresses a four point
        quadrilateral encoded as `[center_x, center_y, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]` in sigmoid space,
        where the corner offsets are shifted by `+0.5`. Only `10` is supported.
    global_pointer_head_size (`int`, *optional*, defaults to 64):
        The size of the global pointer head.
    gp_dropout_value (`float`, *optional*, defaults to 0.1):
        The dropout probability in the global pointer head.
    use_s2r (`bool`, *optional*, defaults to `True`):
        Whether to fuse the successor (ROOR) head into the relative order head with
        [`PPDocLayoutV4S2RFusion`]. When `False` the relative order logits are used directly.
    s2r_steps (`int`, *optional*, defaults to 3):
        Number of propagation steps used to approximate the transitive closure of the successor matrix.
    s2r_damping (`float`, *optional*, defaults to 0.5):
        Damping factor applied to every additional propagation step of the transitive closure.
    s2r_a_init (`float`, *optional*, defaults to 0.0):
        Initial value of the learnable gate that weights the closure term. Defaults to `0.0` so that an untrained
        fusion module is numerically identical to using the relative order logits alone.
    s2r_b_init (`float`, *optional*, defaults to 1.0):
        Value of the weight applied to the relative order logits. This is only a learnable parameter when
        `s2r_learnable_b=True`, otherwise it stays a plain float and is therefore read from the configuration
        rather than from the checkpoint.
    s2r_learnable_b (`bool`, *optional*, defaults to `False`):
        Whether the weight applied to the relative order logits is learnable.

    Examples:

    ```python
    >>> from transformers import PPDocLayoutV4Config, PPDocLayoutV4ForObjectDetection

    >>> # Initializing a PP-DocLayoutV4 configuration
    >>> configuration = PPDocLayoutV4Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPDocLayoutV4ForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pp_doclayout_v4"
    sub_configs = {"backbone_config": AutoConfig}

    layer_types = ("basic", "bottleneck")
    attribute_map = {
        "d_model": "hidden_size",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    backbone_config: dict | PreTrainedConfig | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: list[int] | tuple[int, int] | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    hidden_size: int = 256
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    learn_initial_query: bool = False
    anchor_image_size: list[int] | tuple[int, int] | None = None
    disable_custom_kernels: bool = True
    is_encoder_decoder: bool = True
    num_coords: int = 10
    global_pointer_head_size: int = 64
    gp_dropout_value: float | int = 0.1
    use_s2r: bool = True
    s2r_steps: int = 3
    s2r_damping: float = 0.5
    s2r_a_init: float = 0.0
    s2r_b_init: float = 1.0
    s2r_learnable_b: bool = False

    def __post_init__(self, **kwargs):
        # The anchor generator, the deformable attention reference points and the corner decode are all written
        # against the quad parameterization, so anything else fails with a shape error deep inside the forward.
        if self.num_coords != QUAD_NUM_COORDS:
            raise ValueError(f"PP-DocLayoutV4 only supports `num_coords={QUAD_NUM_COORDS}`, got {self.num_coords}.")

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                # PP-DocLayoutV4 has no mask branch, so the stride 4 feature is never consumed and the
                # backbone only has to emit the last three stages.
                "return_idx": [1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        self.encoder_in_channels = list(self.encoder_in_channels)
        self.feat_strides = list(self.feat_strides)
        self.encode_proj_layers = list(self.encode_proj_layers)
        self.eval_size = list(self.eval_size) if self.eval_size is not None else None
        self.decoder_in_channels = list(self.decoder_in_channels)
        self.anchor_image_size = list(self.anchor_image_size) if self.anchor_image_size is not None else None
        super().__post_init__(**kwargs)


class PPDocLayoutV4ImageProcessor(PPDocLayoutV3ImageProcessor):
    r"""
    Constructs a PP-DocLayoutV4 image processor.

    Images are resized to a fixed 800x800 square with bicubic interpolation and rescaled to `[0, 1]` without further
    normalization, matching the reference `cv2.resize` based preprocessing.

    Post-processing differs from [`PPDocLayoutV3ImageProcessor`], because PP-DocLayoutV4 regresses a four point
    quadrilateral per query instead of predicting a segmentation mask, and emits raw relative/successor order logits
    instead of a decoded reading order.
    """

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
        # Rescaling happens *before* resizing, unlike in `PPDocLayoutV3ImageProcessor`. The reference preprocessing
        # resizes with `cv2.resize`, which evaluates the bicubic kernel in fixed point and rounds back to `uint8`
        # once. Resizing a `uint8` tensor with torchvision rounds a second time with slightly different weights, and
        # the two roundings compound to ~22/255 on high contrast edges -- enough to permute the predicted reading
        # order. Running the same resize in floating point instead keeps every pixel within one 8-bit step.
        if do_resize:
            # The bicubic overshoot has to be clipped, or the error grows back to ~0.2 instead of staying below
            # 1/255. The reference preprocessing resizes `uint8` with `cv2.resize`, whose saturating cast bounds the
            # ringing by the dtype maximum -- so the bound is the range the incoming pixels live in, not the range of
            # this particular image, which would clip too early. `cv2.resize` does not saturate floating point input
            # at all, so a float tensor is only bounded by 1 when it actually is the unit interval the
            # `do_rescale=False` contract documents; anything else is treated as `[0, 255]` like an integer input.
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

    def _get_order_seqs(self, order_logits):
        raise AttributeError("Not needed for PP-DocLayoutV4")

    def extract_custom_vertices(self, polygon, sharp_angle_thresh=45):
        raise AttributeError("Not needed for PP-DocLayoutV4")

    def _mask2polygon(self, mask, epsilon_ratio=0.004):
        raise AttributeError("Not needed for PP-DocLayoutV4")

    def _extract_polygon_points_by_masks(self, boxes, masks, scale_ratio):
        raise AttributeError("Not needed for PP-DocLayoutV4")

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
        if num_coords != QUAD_NUM_COORDS:
            raise ValueError(f"Unsupported num_coords: {num_coords}. PP-DocLayoutV4 only supports quads (10).")
        centers = pred_boxes[..., :2].unsqueeze(-2)
        offsets = pred_boxes[..., 2:].reshape(*pred_boxes.shape[:-1], 4, 2) - 0.5
        return centers + offsets

    def _find_cycle(self, num_nodes, edges):
        """Detects the first cycle in a directed graph with an iterative depth first search."""
        graph = defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
        visited = [False] * num_nodes
        on_stack = [False] * num_nodes

        def depth_first_search(vertex, path):
            visited[vertex] = True
            on_stack[vertex] = True
            path.append(vertex)
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    cycle = depth_first_search(neighbor, path)
                    if cycle is not None:
                        return cycle
                elif on_stack[neighbor]:
                    return path[path.index(neighbor) :]
            path.pop()
            on_stack[vertex] = False
            return None

        for node in range(num_nodes):
            if not visited[node]:
                cycle = depth_first_search(node, [])
                if cycle:
                    return cycle
        return None

    def _remove_cycles(self, num_nodes, edges):
        """Greedily drops the lowest confidence edge of every cycle until the graph is a DAG."""
        edges = dict(edges)
        while True:
            cycle = self._find_cycle(num_nodes, edges)
            if cycle is None:
                break
            weakest_edge, weakest_confidence = None, float("inf")
            for i, j in zip(cycle, cycle[1:] + cycle[:1]):
                if (i, j) in edges and edges[(i, j)] < weakest_confidence:
                    weakest_edge = (i, j)
                    weakest_confidence = edges[(i, j)]
            if weakest_edge is None:
                break
            del edges[weakest_edge]
        return list(edges.keys())

    def _find_connected_components(self, num_nodes, edges):
        """Groups nodes into the connected components of the undirected view of the DAG."""
        parent = list(range(num_nodes))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for i, j in edges:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        components = defaultdict(list)
        for node in range(num_nodes):
            components[find(node)].append(node)
        return list(components.values())

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
        requires_backends(self, ["torch"])
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


def quad_to_rect(quad: torch.Tensor) -> torch.Tensor:
    """
    Converts the 10 dimensional quad parameterization to `(center_x, center_y, width, height)`.

    `quad` is `[..., 10]` in sigmoid space with the corner offsets shifted by `+0.5`. The returned center is the
    *predicted* `center_x, center_y` rather than the centroid of the corners, and `width, height` are the full extent
    of the offsets. This rect is only used to place the deformable attention sampling grid.
    """
    center = quad[..., :2]
    offsets_x = quad[..., 2::2] - 0.5
    offsets_y = quad[..., 3::2] - 0.5
    width = offsets_x.amax(-1, keepdim=True) - offsets_x.amin(-1, keepdim=True)
    height = offsets_y.amax(-1, keepdim=True) - offsets_y.amin(-1, keepdim=True)
    return torch.cat([center, width, height], dim=-1)


class PPDocLayoutV4MultiscaleDeformableAttention(PPDocLayoutV3MultiscaleDeformableAttention):
    pass


class PPDocLayoutV4MLPPredictionHead(PPDocLayoutV3MLPPredictionHead):
    pass


class PPDocLayoutV4GlobalPointer(nn.Module):
    """
    Pairwise scorer shared by both reading order heads.

    With `antisymmetric=True` (the relative order head) the scores satisfy `logits[i, j] == -logits[j, i]`, so a
    positive score means "read `i` before `j`". With `antisymmetric=False` (the successor head) only self loops are
    suppressed, and a positive score means "`j` directly follows `i`".
    """

    def __init__(self, config: PPDocLayoutV4Config, antisymmetric: bool):
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.antisymmetric = antisymmetric
        self.dense = nn.Linear(config.d_model, self.head_size * 2)
        self.dropout = nn.Dropout(config.gp_dropout_value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        query_key_projection = self.dense(inputs).reshape(batch_size, sequence_length, 2, self.head_size)
        query_key_projection = self.dropout(query_key_projection)
        queries, keys = torch.unbind(query_key_projection, dim=2)

        logits = (queries @ keys.transpose(-2, -1)) / (self.head_size**0.5)
        if self.antisymmetric:
            return logits - logits.transpose(-2, -1)
        eye = torch.eye(sequence_length, device=logits.device, dtype=logits.dtype)
        return logits - eye * 1e4


class PPDocLayoutV4S2RFusion(nn.Module):
    """
    Fuses the transitive closure of the successor matrix into the relative order logits.

    The fused logits are `a * antisymmetrize(closure(successor)) + b * relative`. With `s2r_a_init=0.0` the module
    starts out numerically identical to using the relative order logits alone, and with `s2r_learnable_b=False` the
    weight `b` stays a plain float, which is why a checkpoint only carries `a`.
    """

    def __init__(self, config: PPDocLayoutV4Config):
        super().__init__()
        self.steps = config.s2r_steps
        self.damping = config.s2r_damping
        self.a = nn.Parameter(torch.full((1,), config.s2r_a_init))
        self.learnable_b = config.s2r_learnable_b
        if self.learnable_b:
            self.b = nn.Parameter(torch.full((1,), config.s2r_b_init))
        else:
            self.b = float(config.s2r_b_init)

    def forward(self, relative_logits: torch.Tensor, successor_logits: torch.Tensor) -> torch.Tensor:
        num_queries = successor_logits.shape[-1]
        eye = torch.eye(num_queries, device=successor_logits.device, dtype=successor_logits.dtype)

        # Soft directed adjacency, where adjacency[i, j] approximates P(i directly precedes j).
        adjacency = successor_logits.sigmoid() * (1.0 - eye)
        # Clamping the row sums from below at 1 damps dense rows without amplifying weak edges or terminal nodes.
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1.0)

        # Soft transitive closure: sum over k of damping^(k - 1) * adjacency^k.
        closure = adjacency
        power = adjacency
        for _ in range(self.steps - 1):
            power = self.damping * torch.bmm(adjacency, power)
            closure = closure + power

        return self.a * (closure - closure.transpose(-2, -1)) + self.b * relative_logits


@auto_docstring
class PPDocLayoutV4PreTrainedModel(PPDocLayoutV3PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, PPDocLayoutV4MultiscaleDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1

            init.copy_(module.sampling_offsets.bias, grid_init.view(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            init.xavier_uniform_(module.output_proj.weight)
            init.constant_(module.output_proj.bias, 0.0)

        elif isinstance(module, PPDocLayoutV4Model):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            init.xavier_uniform_(module.enc_score_head.weight)
            init.constant_(module.enc_score_head.bias, bias)
            # The class heads are untied, so every decoder layer gets its own biased initialization.
            for class_embed in module.decoder.class_embed:
                init.xavier_uniform_(class_embed.weight)
                init.constant_(class_embed.bias, bias)

        elif isinstance(module, PPDocLayoutV4S2RFusion):
            init.constant_(module.a, self.config.s2r_a_init)
            if module.learnable_b:
                init.constant_(module.b, self.config.s2r_b_init)

        elif isinstance(module, nn.BatchNorm2d):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)


class PPDocLayoutV4HybridEncoder(RTDetrHybridEncoder):
    """
    PP-DocLayoutV4 uses the plain RT-DETR hybrid encoder: AIFI, a top-down FPN and a bottom-up PAN. Unlike
    [`PPDocLayoutV3HybridEncoder`] there is no mask feature head, so the stride 4 backbone feature is unused.
    """


@auto_docstring(
    custom_intro="""
    Output type of [`PPDocLayoutV4Decoder`].
    """
)
@dataclass
class PPDocLayoutV4DecoderOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_coords)`):
        Stacked intermediate reference points (refined quads of each layer of the decoder).
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
        Classification logits of the last decoder layer. Unlike RT-DETR, PP-DocLayoutV4 only evaluates its
        classification and reading order heads on the last layer.
    relative_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise relative reading order logits of the last decoder layer, after the optional S2R fusion.
    successor_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise direct successor (ROOR) logits of the last decoder layer.
    """

    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    relative_order_logits: torch.FloatTensor | None = None
    successor_order_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None


class PPDocLayoutV4Decoder(PPDocLayoutV3Decoder):
    """
    Main differences to `PPDocLayoutV3Decoder`:
        1. The bbox and class heads are untied and there is one of each per decoder layer, instead of a single head
           shared with the encoder.
        2. Reference points are `config.num_coords` dimensional quads. Deformable attention samples on the enclosing
           rect of the quad, while the query position embedding consumes the full quad.
        3. Classification and reading order are only evaluated on the last layer, and there is no mask branch.
    """

    def __init__(self, config: PPDocLayoutV4Config):
        super().__init__(config)

        self.num_queries = config.num_queries
        self.query_pos_head = PPDocLayoutV4MLPPredictionHead(
            config.num_coords, 2 * config.d_model, config.d_model, num_layers=2
        )
        self.bbox_embed = nn.ModuleList(
            [
                PPDocLayoutV4MLPPredictionHead(config.d_model, config.d_model, config.num_coords, num_layers=3)
                for _ in range(config.decoder_layers)
            ]
        )
        self.class_embed = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(config.decoder_layers)]
        )

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        order_head=None,
        global_pointer=None,
        successor_order_head=None,
        successor_global_pointer=None,
        s2r_fusion=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            The query embeddings that are passed into the decoder.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
            in `[0, 1]`:
            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
        reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
            Quad reference points in inverse sigmoid space.
        spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
            Spatial shapes of the feature maps.
        spatial_shapes_list (`list[tuple[int, int]]`, *optional*):
            Spatial shapes of the feature maps as a list, kept alongside `spatial_shapes` so that the deformable
            attention can index them without a device synchronization.
        level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
            Indexes for the start of each feature level. In range `[0, sequence_length]`.
        order_head (`nn.ModuleList`, *optional*):
            Per-layer projections feeding the relative order global pointer.
        global_pointer (`PPDocLayoutV4GlobalPointer`, *optional*):
            Antisymmetric pairwise scorer producing the relative reading order logits.
        successor_order_head (`nn.ModuleList`, *optional*):
            Per-layer projections feeding the successor global pointer.
        successor_global_pointer (`PPDocLayoutV4GlobalPointer`, *optional*):
            Pairwise scorer producing the direct successor (ROOR) logits.
        s2r_fusion (`PPDocLayoutV4S2RFusion`, *optional*):
            Fuses the successor logits into the relative order logits. When `None` the relative order logits are
            returned unchanged.
        """
        hidden_states = inputs_embeds
        reference_points = F.sigmoid(reference_points)

        intermediate = ()
        intermediate_reference_points = ()
        logits = None
        relative_order_logits = None
        successor_order_logits = None
        last_index = len(self.layers) - 1

        for idx, decoder_layer in enumerate(self.layers):
            # Deformable attention samples on the enclosing rect of the quad.
            reference_points_input = quad_to_rect(reference_points).unsqueeze(2)

            hidden_states = decoder_layer(
                hidden_states,
                object_queries_position_embeddings=self.query_pos_head(reference_points),
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

            reference_points = F.sigmoid(self.bbox_embed[idx](hidden_states) + inverse_sigmoid(reference_points))

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            # Only the last layer's class and order predictions are used, matching the reference implementation.
            if idx != last_index:
                continue

            logits = self.class_embed[idx](hidden_states)
            if order_head is not None and global_pointer is not None:
                valid_query = hidden_states[:, -self.num_queries :] if self.num_queries is not None else hidden_states
                successor_order_logits = successor_global_pointer(successor_order_head[idx](valid_query))
                relative_order_logits = global_pointer(order_head[idx](valid_query))
                if s2r_fusion is not None:
                    relative_order_logits = s2r_fusion(relative_order_logits, successor_order_logits)

        return PPDocLayoutV4DecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=torch.stack(intermediate, dim=1),
            intermediate_reference_points=torch.stack(intermediate_reference_points, dim=1),
            logits=logits,
            relative_order_logits=relative_order_logits,
            successor_order_logits=successor_order_logits,
        )


@auto_docstring(
    custom_intro="""
    Output type of [`PPDocLayoutV4Model`].
    """
)
@dataclass
class PPDocLayoutV4ModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_coords)`):
        Stacked intermediate reference points (refined quads of each layer of the decoder).
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
        Classification logits of the last decoder layer.
    relative_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise relative reading order logits, after the optional S2R fusion.
    successor_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise direct successor (ROOR) logits.
    init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
        Initial quad reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
        Class logits of the encoder proposals that were selected as decoder queries.
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
        Quads of the encoder proposals that were selected as decoder queries.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
        Class logits of every encoder proposal.
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_coords)`):
        Quad logits of every encoder proposal.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values.
    """

    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    relative_order_logits: torch.FloatTensor | None = None
    successor_order_logits: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    init_reference_points: torch.FloatTensor | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


@auto_docstring(
    custom_intro="""
    PP-DocLayoutV4 Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.
    """
)
class PPDocLayoutV4Model(PPDocLayoutV3Model):
    # The bbox and class heads are untied in PP-DocLayoutV4, so nothing is shared with the encoder heads.
    _tied_weights_keys = {}

    def __init__(self, config: PPDocLayoutV4Config):
        super().__init__(config)

        # The backbone emits exactly the three levels the encoder consumes, so unlike PP-DocLayoutV3 there is no
        # leading projection to drop here.
        encoder_input_proj_list = []
        self.encoder_input_proj = nn.ModuleList(encoder_input_proj_list)

        self.enc_bbox_head = PPDocLayoutV4MLPPredictionHead(
            config.d_model, config.d_model, config.num_coords, num_layers=3
        )

        # PP-DocLayoutV4 does not reserve an extra "no object" row in the denoising embedding, so the `num_labels + 1`
        # embedding built by [`PPDocLayoutV3Model`] is overwritten here. The modular converter only deduplicates
        # plain assignments, so both allocations survive into the generated file, the second one winning.
        # CODEPATH: PP-DocLayoutV4_safetensors trains with `num_denoising=100`, so it carries the embedding even
        # though contrastive denoising is training only. Set it to 0 to build an inference-only model.
        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)

        self.decoder = PPDocLayoutV4Decoder(config)
        del self.decoder.class_embed
        del self.decoder.bbox_embed

        self.decoder_order_head = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(config.decoder_layers)]
        )
        self.decoder_global_pointer = PPDocLayoutV4GlobalPointer(config, antisymmetric=True)
        self.decoder_roor_order_head = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(config.decoder_layers)]
        )
        self.decoder_roor_global_pointer = PPDocLayoutV4GlobalPointer(config, antisymmetric=False)
        # CODEPATH: PP-DocLayoutV4_safetensors sets `use_s2r=True`. `False` reproduces the pre-S2R baseline,
        # where the relative order logits are used without the successor closure term.
        self.s2r_fusion = PPDocLayoutV4S2RFusion(config) if config.use_s2r else None

        del self.decoder_norm
        del self.mask_enhanced
        del self.mask_query_head

        self.post_init()

    @staticmethod
    @compile_compatible_method_lru_cache(maxsize=32)
    def _cached_generate_anchors(
        spatial_shapes: tuple[tuple[int, int], ...],
        grid_size: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the 10 dimensional quad anchors in inverse sigmoid space.

        The layout is `[center_x, center_y, x1 + 0.5, y1 + 0.5, ..., x4 + 0.5, y4 + 0.5]` with the corners in
        top-left, top-right, bottom-right, bottom-left order. Only the centers vary spatially, the four offsets are a
        per-level constant, so anchor validity is decided by the centers alone.
        """
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height

            half = grid_size * (2.0**level) / 2.0
            corners = torch.tensor(
                [0.5 - half, 0.5 - half, 0.5 + half, 0.5 - half, 0.5 + half, 0.5 + half, 0.5 - half, 0.5 + half],
                device=device,
                dtype=dtype,
            ).expand(1, height * width, 8)
            anchors.append(torch.concat([grid_xy.reshape(1, height * width, 2), corners], -1))

        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.full((), torch.finfo(dtype).max, dtype=dtype, device=device))

        return anchors, valid_mask

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | PPDocLayoutV4ModelOutput:
        r"""
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Not supported: PP-DocLayoutV4 is inference only in Transformers.
        """
        if labels is not None:
            raise ValueError("PPDocLayoutV4Model does not support training")

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        features = self.backbone(pixel_values, pixel_mask)
        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]

        if encoder_outputs is None:
            encoder_outputs = self.encoder(proj_feats, **kwargs)
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sources = [
            self.decoder_input_proj[level](source) for level, source in enumerate(encoder_outputs.last_hidden_state)
        ]

        # Prepare decoder inputs (by flattening)
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # prepare denoising training
        # CODEPATH: unreachable, `labels` is rejected above. Kept as the scaffolding a future training
        # implementation would build on. Two things have to change before it can run: the helper emits 4-coordinate
        # `(cx, cy, w, h)` boxes, which do not concatenate with V4's `num_coords`-wide reference points, and it fills
        # unused slots with the class index `num_classes`, which is out of range for the `num_labels`-wide denoising
        # embedding below.
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        dtype = source_flatten.dtype
        # CODEPATH: PP-DocLayoutV4_safetensors leaves `anchor_image_size` unset and recomputes the anchors from
        # the input, the cached branch is for configs that pin a single evaluation resolution.
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure lru_cache is working
            anchors, valid_mask = self.generate_anchors(tuple(spatial_shapes_list), device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors.to(device, dtype), self.valid_mask.to(device, dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(dtype) * source_flatten
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        # Class-agnostic top-k over positions
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        # CODEPATH: PP-DocLayoutV4_safetensors sets `learn_initial_query=False` and takes the top-k encoder features
        # as queries. The learned embedding branch is inherited from RT-DETR and unused by released checkpoints.
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        init_reference_points = reference_points_unact.detach()

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            order_head=self.decoder_order_head,
            global_pointer=self.decoder_global_pointer,
            successor_order_head=self.decoder_roor_order_head,
            successor_global_pointer=self.decoder_roor_global_pointer,
            s2r_fusion=self.s2r_fusion,
            **kwargs,
        )

        return PPDocLayoutV4ModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            logits=decoder_outputs.logits,
            relative_order_logits=decoder_outputs.relative_order_logits,
            successor_order_logits=decoder_outputs.successor_order_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


@auto_docstring(
    custom_intro="""
    Output type of [`PPDocLayoutV4ForObjectDetection`].
    """
)
@dataclass
class PPDocLayoutV4ForObjectDetectionOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
        Classification logits (without no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
        Normalized quads for all queries, encoded as `[center_x, center_y, dx1, dy1, ..., dx4, dy4]` where the corner
        offsets are shifted by `+0.5`. Use
        [`~PPDocLayoutV4ImageProcessor.post_process_object_detection`] to retrieve the unnormalized corners and their
        enclosing boxes.
    relative_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise relative reading order logits, after the optional S2R fusion. A positive `relative_order_logits[i, j]`
        means query `i` is read before query `j`.
    successor_order_logits (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`):
        Pairwise direct successor (ROOR) logits. A positive `successor_order_logits[i, j]` means query `j` directly
        follows query `i`.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_coords)`):
        Stacked intermediate reference points (refined quads of each layer of the decoder).
    init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
        Initial quad reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
        Class logits of the encoder proposals that were selected as decoder queries.
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`):
        Quads of the encoder proposals that were selected as decoder queries.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
        Class logits of every encoder proposal.
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_coords)`):
        Quad logits of every encoder proposal.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values.
    """

    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    relative_order_logits: torch.FloatTensor | None = None
    successor_order_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    init_reference_points: torch.FloatTensor | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


@auto_docstring(
    custom_intro="""
    PP-DocLayoutV4 Model (consisting of a backbone and encoder-decoder) outputting quadrilaterals, class logits and
    reading order logits, for tasks such as document layout analysis.
    """
)
class PPDocLayoutV4ForObjectDetection(PPDocLayoutV3ForObjectDetection, PPDocLayoutV4PreTrainedModel):
    _tied_weights_keys = {}
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPDocLayoutV4Config):
        super().__init__(config)

        self.model = PPDocLayoutV4Model(config)
        # `PPDocLayoutV4Model` owns the denoising embedding and only builds it when `config.num_denoising > 0`, so the
        # unconditional one [`PPDocLayoutV3ForObjectDetection`] adds here has to go.
        del self.model.denoising_class_embed
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | PPDocLayoutV4ForObjectDetectionOutput:
        r"""
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Not supported: PP-DocLayoutV4 is inference only in Transformers.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
        >>> image_processor = AutoImageProcessor.from_pretrained(model_path)
        >>> model = AutoModelForObjectDetection.from_pretrained(model_path)

        >>> inputs = image_processor(images=[image], return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # results are already sorted by reading order
        >>> results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
        >>> for result in results:
        ...     for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        ...         box = [round(i, 2) for i in box.tolist()]
        ...         print(f"Order {idx + 1}: {model.config.id2label[label_id.item()]}: {score.item():.2f} {box}")
        Order 1: text: 0.99 [336.07, 182.04, 894.17, 652.62]
        Order 2: paragraph_title: 0.98 [336.45, 681.88, 868.78, 796.91]
        Order 3: text: 0.99 [334.01, 840.84, 889.11, 1452.29]
        Order 4: text: 0.99 [920.65, 183.62, 1476.75, 462.75]
        Order 5: text: 0.99 [919.55, 482.55, 1479.85, 763.41]
        Order 6: text: 0.98 [919.34, 844.48, 1481.28, 1219.44]
        Order 7: text: 0.98 [920.84, 1238.61, 1467.66, 1374.54]
        Order 8: text: 0.90 [334.24, 1612.85, 1478.79, 1730.92]
        Order 9: text: 0.95 [334.25, 1755.98, 1467.34, 1845.6]
        Order 10: text: 0.65 [337.14, 1909.47, 659.78, 1938.2]
        Order 11: footnote: 0.78 [338.73, 2114.87, 1448.16, 2172.06]
        Order 12: number: 0.98 [106.08, 2257.42, 134.76, 2281.3]
        Order 13: footer: 0.93 [339.27, 2255.7, 984.16, 2282.81]
        ```"""
        if labels is not None:
            raise ValueError("PPDocLayoutV4ForObjectDetection does not support training")

        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        return PPDocLayoutV4ForObjectDetectionOutput(
            logits=outputs.logits,
            pred_boxes=outputs.intermediate_reference_points[:, -1],
            relative_order_logits=outputs.relative_order_logits,
            successor_order_logits=outputs.successor_order_logits,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )


__all__ = [
    "PPDocLayoutV4Config",
    "PPDocLayoutV4ForObjectDetection",
    "PPDocLayoutV4ImageProcessor",
    "PPDocLayoutV4Model",
    "PPDocLayoutV4PreTrainedModel",
]
