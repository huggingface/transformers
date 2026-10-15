# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Image processor class for Cellpose."""

import math
from itertools import product

import torch
import torch.nn.functional as F

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring, is_torch_greater_or_equal


class CellposeImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    tile_size (`int`, *optional*, defaults to `self.tile_size`):
        Size of the square tiles the (padded) images are cut into. Must match the model input size.
    tile_overlap (`float`, *optional*, defaults to `self.tile_overlap`):
        Fraction of overlap between neighbouring tiles, clipped to `[0.05, 0.5]`.
    """

    tile_size: int
    tile_overlap: float


def _tile_origins(height: int, width: int, tile_size: int, tile_overlap: float) -> list[tuple[int, int]]:
    """(y, x) of every tile of a padded image, in row-major order."""
    return list(product(_tile_starts(height, tile_size, tile_overlap), _tile_starts(width, tile_size, tile_overlap)))


def _per_label(labels: torch.Tensor, values: torch.Tensor, size: int, reduce: str, init: int = 0) -> torch.Tensor:
    """`reduce` of `values` per label, `init` for labels without pixels."""
    return values.new_full((size,), init).scatter_reduce_(0, labels, values, reduce)


def _mask_sizes(masks: torch.Tensor) -> torch.Tensor:
    """Pixel count of each label, indexed by label. `unique` rather than `bincount`: on MPS, an integer `bincount`
    where most pixels share one value (the background) is ~1000x slower."""
    labels, counts = torch.unique(masks, return_counts=True)
    return torch.zeros(int(labels[-1]) + 1, dtype=counts.dtype, device=masks.device).index_put_((labels,), counts)


# --------------------------------------------------------------------------------------------------------------------
# Vendored from Cellpose (https://github.com/MouseLand/cellpose), ported from numpy/scipy/fastremap to torch.
# Ported code is tagged `Vendored from <cellpose file>::<function>`; masks are identical to Cellpose 4.x.
#
# Copyright © 2020 Howard Hughes Medical Institute
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------------------------------------------


def _normalize_percentile(image: torch.Tensor, lower: float = 1.0, upper: float = 99.0) -> torch.Tensor:
    """Vendored from `cellpose/transforms.py::normalize_img` (defaults, `normalize99`): per-channel 1st/99th
    percentile normalization, flat channels become 0."""
    out = torch.zeros_like(image)
    for c, channel in enumerate(image):
        if channel.max() - channel.min() <= 0:
            continue
        sample = channel
        if channel.numel() > 224**3:  # `normalize99(downsample=True)`
            sample = channel[:: max(1, channel.shape[0] // 224), :: max(1, channel.shape[1] // 224)]
        low, high = torch.quantile(sample.flatten(), torch.tensor([lower / 100, upper / 100], device=image.device))
        if high - low > 1e-3:
            out[c] = (channel - low) / (high - low)
    return out


def _get_padding(height: int, width: int, tile_size: int) -> tuple[int, int, int, int]:
    """Vendored from `cellpose/transforms.py::get_pad_yx` (`div=16, extra=1, min_size=(tile_size, tile_size)`)."""

    def pad(length):
        extra = 16 * math.ceil(length / 16) - length if length >= tile_size else max(0, tile_size - length - 16)
        return 8 + extra // 2, 8 + extra - extra // 2

    return (*pad(height), *pad(width))


def _tile_starts(length: int, tile_size: int, tile_overlap: float) -> list[int]:
    """Vendored from `cellpose/transforms.py::make_tiles` (`augment=False`): tile start offsets along one axis."""
    tile_overlap = min(0.5, max(0.05, tile_overlap))
    num = 1 if length <= tile_size else math.ceil((1.0 + 2 * tile_overlap) * length / tile_size)
    if num == 1:
        return [0]
    step = (length - tile_size) / (num - 1)
    return [int(i * step) for i in range(num - 1)] + [length - tile_size]


def _taper_mask(tile_size: int, sigma: float = 7.5) -> torch.Tensor:
    """Vendored from `cellpose/transforms.py::_taper_mask`, for a square tile."""
    size = max(224, tile_size)
    xm = torch.arange(size, dtype=torch.float64)
    xm = (xm - xm.mean()).abs()
    mask = 1 / (1 + torch.exp((xm - (size / 2 - 20)) / sigma))
    mask = mask[:, None] * mask[None, :]
    start = size // 2 - tile_size // 2
    return mask[start : start + tile_size, start : start + tile_size]


def _follow_flows(flows: torch.Tensor, inds: torch.Tensor, num_iterations: int) -> torch.Tensor:
    """Vendored from `cellpose/dynamics.py::steps_interp` (2D): Euler integration of `flows` (2, H, W) for pixels
    `inds` (2, n)."""
    height, width = flows.shape[1:]
    scale = torch.tensor([width - 1.0, height - 1.0], device=flows.device)  # grid_sample works in (x, y), [-1, 1]
    field = flows.flip(0)[None].float() * (2.0 / scale)[None, :, None, None]
    points = (inds.flip(0).T.float() / scale * 2 - 1)[None, None]  # (1, 1, n, 2)
    for _ in range(num_iterations):
        step = F.grid_sample(field, points, align_corners=False)  # (1, 2, 1, n)
        points = (points + step.permute(0, 2, 3, 1)).clamp_(-1.0, 1.0)
    return ((points[0, 0] + 1) * 0.5 * scale).flip(-1).T  # (2, n) as (y, x)


def _renumber(masks: torch.Tensor) -> torch.Tensor:
    """Relabel to 1..N in order of first appearance (raster order). Written for this port, same output as
    `fastremap.renumber` which Cellpose calls."""
    flat = masks.flatten()
    positions = torch.arange(flat.numel(), device=flat.device)
    first = torch.full((int(flat.max()) + 1,), flat.numel(), device=flat.device)
    first = first.scatter_reduce_(0, flat, positions, "amin").index_fill_(0, flat.new_zeros(1), -1)  # background first
    lookup = torch.empty_like(first)
    lookup[first.argsort()] = torch.arange(len(first), device=flat.device)
    return lookup[masks]


def _get_masks(points: torch.Tensor, inds: torch.Tensor, shape: tuple[int, int], max_size_fraction: float):
    """Vendored from `cellpose/dynamics.py::get_masks_torch` (2D): cluster converged pixel positions into labels."""
    rpad = 20
    height, width = shape[0] + 2 * rpad, shape[1] + 2 * rpad
    points = (points + rpad).clamp(min=0)
    points = torch.stack([points[0].clamp(max=shape[0] + rpad - 1), points[1].clamp(max=shape[1] + rpad - 1)])
    flat_points = points[0] * width + points[1]
    hist = torch.bincount(flat_points, minlength=height * width).to(torch.int32).view(height, width)

    hist_max = F.max_pool2d(hist[None].float(), kernel_size=5, stride=1, padding=2)[0]
    seeds = torch.nonzero((hist.float() - hist_max > -1e-6) & (hist > 10))
    if len(seeds) == 0:
        return torch.zeros(shape, dtype=torch.long, device=points.device)
    seeds = seeds[hist[seeds[:, 0], seeds[:, 1]].argsort()]

    # grow each seed inside its 11x11 window over histogram bins with more than 2 converged pixels
    offsets = torch.arange(-5, 6, device=seeds.device)
    window = (offsets[:, None] * width + offsets[None, :]).flatten()
    hist_windows = hist.view(-1)[window + (seeds[:, 0] * width + seeds[:, 1])[:, None]].view(-1, 11, 11)
    seed_masks = torch.zeros((len(seeds), 11, 11), device=seeds.device)
    seed_masks[:, 5, 5] = 1
    for _ in range(5):
        seed_masks = F.max_pool2d(seed_masks, kernel_size=3, stride=1, padding=1) * (hist_windows > 2)

    pixels = torch.nonzero(seed_masks)
    mask_idx = pixels[:, 0]
    mask_pos = pixels[:, 1:] + seeds[mask_idx] - 5
    label_map = torch.zeros(height * width, dtype=torch.long, device=seeds.device)
    label_map.scatter_reduce_(0, mask_pos[:, 0] * width + mask_pos[:, 1], mask_idx + 1, "amax", include_self=False)

    masks = torch.zeros(shape, dtype=torch.long, device=seeds.device)
    masks[inds[0], inds[1]] = label_map[flat_points]
    # remove masks covering more than `max_size_fraction` of the image
    masks = torch.where(_mask_sizes(masks)[masks] > shape[0] * shape[1] * max_size_fraction, 0, masks)
    return _renumber(masks)


def _fill_holes(masks: torch.Tensor, max_elements: int = 2**25) -> torch.Tensor:
    """
    Vendored from `cellpose/utils.py::fill_holes_and_remove_small_masks` (hole filling): inside each mask's bounding
    box, pixels not reachable from the box border (4-connected, without crossing the mask) are added to the mask.
    Masks are labelled 1..N.
    """
    y, x = torch.nonzero(masks, as_tuple=True)
    labels = masks[y, x]
    num = int(labels.max()) + 1 if len(labels) else 1
    y0, x0 = _per_label(labels, y, num, "amin", 1 << 30), _per_label(labels, x, num, "amin", 1 << 30)
    heights = _per_label(labels, y, num, "amax") - y0 + 3
    widths = _per_label(labels, x, num, "amax") - x0 + 3
    filled = masks.flatten().clone()
    # one padded (num_masks, height, width) canvas per chunk of similarly sized masks, flood the outside from the border
    order = (heights * widths)[1:].argsort() + 1
    start = 0
    while start < len(order):
        size = int(heights[order[start:]].max()) * int(widths[order[start:]].max())  # upper bound for the chunk
        chunk = order[start : start + max(1, max_elements // size)]
        start += len(chunk)
        height, width = int(heights[chunk].max()), int(widths[chunk].max())
        index = torch.full((num,), -1, device=y.device)
        index[chunk] = torch.arange(len(chunk), device=y.device)
        keep = index[labels] >= 0
        mask = torch.zeros(len(chunk), height, width, dtype=torch.bool, device=y.device)
        mask[index[labels[keep]], y[keep] - y0[labels[keep]] + 1, x[keep] - x0[labels[keep]] + 1] = True
        outside = torch.zeros_like(mask)
        outside[:, 0], outside[:, -1], outside[:, :, 0], outside[:, :, -1] = True, True, True, True
        while True:
            grown = outside.clone()
            grown[:, 1:] |= outside[:, :-1]
            grown[:, :-1] |= outside[:, 1:]
            grown[:, :, 1:] |= outside[:, :, :-1]
            grown[:, :, :-1] |= outside[:, :, 1:]
            grown &= ~mask
            if torch.equal(grown, outside):
                break
            outside = grown
        i, hy, hx = torch.nonzero(~outside & ~mask, as_tuple=True)
        if len(i):
            # a hole overwrites the cells it contains, like the original's sequential fill (ponytail: nested holes
            # resolve to the larger label instead of the processing order)
            hy, hx = hy - 1 + y0[chunk[i]], hx - 1 + x0[chunk[i]]
            filled.scatter_reduce_(0, hy * masks.shape[1] + hx, chunk[i], "amax", include_self=False)
    return filled.view_as(masks)


def _remove_bad_flow_masks(masks: torch.Tensor, flows: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Vendored from `cellpose/dynamics.py::remove_bad_flow_masks` (2D, with `masks_to_flows_gpu`, `center_of_mass` and
    `_extend_centers_gpu`): recompute flows from `masks` by heat diffusion from each mask center, then drop masks whose
    mean squared error to the predicted `flows / 5` exceeds `threshold`. Expects masks labelled 1..N.
    """
    padded = F.pad(masks, (1, 1, 1, 1))
    width = padded.shape[1]
    y, x = torch.nonzero(padded, as_tuple=True)
    labels = padded[y, x]
    # (9, n) flat indices of each pixel's neighbourhood: center, up, down, left, right, diagonals
    dy = torch.tensor([0, -1, 1, 0, 0, -1, -1, 1, 1], device=y.device)
    dx = torch.tensor([0, 0, 0, -1, 1, -1, 1, -1, 1], device=y.device)
    neighbors = (y + dy[:, None]) * width + (x + dx[:, None])
    is_neighbor = padded.view(-1)[neighbors] == labels

    # diffusion source: rounded center of mass, or the closest mask pixel if it falls outside (`center_of_mass`)
    counts = torch.bincount(labels)
    num = len(counts)
    y0, x0 = _per_label(labels, y, num, "amin", 1 << 30), _per_label(labels, x, num, "amin", 1 << 30)

    def round_mean(total):  # round-half-to-even of total / count in integers, exact on any device
        count = counts.clamp_min(1)
        quotient, remainder = total // count, total % count
        return quotient + ((2 * remainder > count) | ((2 * remainder == count) & (quotient % 2 == 1))).long()

    # mean relative to the box corner, like `center_of_mass` on the cropped mask
    cy = (round_mean(_per_label(labels, y - y0[labels], num, "sum")) + y0)[labels]
    cx = (round_mean(_per_label(labels, x - x0[labels], num, "sum")) + x0)[labels]
    # argmin of the distance, first pixel in raster order on ties
    distance = ((y - cy) ** 2 + (x - cx) ** 2) * len(y) + torch.arange(len(y), device=y.device)
    centers = (y * width + x)[_per_label(labels, distance, num, "amin", 1 << 62)[1:] % len(y)]
    extent = int((_per_label(labels, y, num, "amax") - y0 + _per_label(labels, x, num, "amax") - x0)[1:].max()) + 4

    dtype = torch.float32 if masks.device.type == "mps" else torch.float64  # MPS has no float64, like cellpose
    heat = torch.zeros(padded.numel(), dtype=dtype, device=y.device)
    for _ in range(2 * extent):
        heat[centers] += 1
        heat[neighbors[0]] = (heat[neighbors] * is_neighbor).sum(dim=0) / 9
    mask_flows = torch.stack([heat[neighbors[2]] - heat[neighbors[1]], heat[neighbors[4]] - heat[neighbors[3]]])
    mask_flows /= mask_flows.norm(dim=0).clamp_min(1e-30)

    error = (mask_flows - (flows[:, y - 1, x - 1] / 5.0).to(dtype)).pow(2).sum(0)
    bad = torch.bincount(labels, weights=error) / counts > threshold  # per label, background is nan -> False
    return torch.where(bad[masks], 0, masks)


# --------------------------------------------------------------------------------------------------------------------
# End of vendored Cellpose code
# --------------------------------------------------------------------------------------------------------------------


@auto_docstring
class CellposeImageProcessor(TorchvisionBackend):
    valid_kwargs = CellposeImageProcessorKwargs
    tile_size = 256
    tile_overlap = 0.1

    def _prepare_images_structure(self, images, expected_ndims: int = 3) -> list:
        """Accept a single image (2D grayscale, or with channels first/last), a 4D batch, or a list of images."""
        images = self.fetch_images(images)
        return list(images) if isinstance(images, (list, tuple)) or getattr(images, "ndim", 0) == 4 else [images]

    def process_image(self, image, input_data_format=None, **kwargs) -> torch.Tensor:
        """Float (3, H, W) tensor: first 3 channels kept, missing channels zero-filled."""
        # cellpose `convert_image` channel guess, which also handles 2-channel (cytoplasm + nuclei) images
        if input_data_format is None and getattr(image, "ndim", 0) == 3:
            input_data_format = ChannelDimension.LAST if image.shape[0] >= image.shape[-1] else ChannelDimension.FIRST
        image = super().process_image(image, input_data_format=input_data_format, **kwargs).float()[:3]
        return F.pad(image, (0, 0, 0, 0, 0, 3 - image.shape[0]))

    def _preprocess(
        self,
        images: list[torch.Tensor],
        tile_size: int,
        tile_overlap: float,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        # Vendored from `cellpose/models.py::CellposeModel._run_net` (normalize, pad, tile), see helpers above
        # ponytail: no diameter rescaling (rescale=1) and no test-time flip augmentation; add a resize + `unaugment_tiles` here if needed
        tiles = []
        for image in images:
            top, bottom, left, right = _get_padding(*image.shape[-2:], tile_size)
            image = F.pad(_normalize_percentile(image), (left, right, top, bottom))
            for y, x in _tile_origins(*image.shape[-2:], tile_size, tile_overlap):
                tiles.append(image[:, y : y + tile_size, x : x + tile_size])
        return BatchFeature(data={"pixel_values": torch.stack(tiles)}, tensor_type=return_tensors)

    def stitch_tiles(
        self,
        logits: torch.Tensor,
        target_sizes: list[tuple[int, int]] | torch.Tensor,
        tile_overlap: float | None = None,
    ) -> list[torch.Tensor]:
        """
        Vendored from `cellpose/transforms.py::average_tiles`: blend per-tile predictions back into full images with the
        tapered mask, then crop the padding.

        Args:
            logits (`torch.Tensor` of shape `(num_tiles, channels, tile_size, tile_size)`):
                Per-tile outputs, in the order produced by `preprocess`.
            target_sizes (`list[tuple[int, int]]` or `torch.Tensor`):
                Original `(height, width)` of each image, in the order they were preprocessed.
            tile_overlap (`float`, *optional*, defaults to `self.tile_overlap`):
                Tile overlap used at preprocessing time.

        Returns:
            `list[torch.Tensor]`: One `(channels, height, width)` tensor per image.
        """
        tile_overlap = self.tile_overlap if tile_overlap is None else tile_overlap
        tile_size = logits.shape[-1]
        taper = _taper_mask(tile_size).to(logits.device, torch.float32)
        results, index = [], 0
        for height, width in [(int(h), int(w)) for h, w in target_sizes]:
            top, bottom, left, right = _get_padding(height, width, tile_size)
            padded_h, padded_w = height + top + bottom, width + left + right
            summed = logits.new_zeros((logits.shape[1], padded_h, padded_w), dtype=torch.float32)
            weight = logits.new_zeros((padded_h, padded_w), dtype=torch.float32)
            for y, x in _tile_origins(padded_h, padded_w, tile_size, tile_overlap):
                summed[:, y : y + tile_size, x : x + tile_size] += logits[index].float() * taper
                weight[y : y + tile_size, x : x + tile_size] += taper
                index += 1
            results.append((summed / weight)[:, top : top + height, left : left + width])
        return results

    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | torch.Tensor,
        cellprob_threshold: float = 0.0,
        flow_threshold: float | None = 0.4,
        num_iterations: int = 200,
        min_size: int = 15,
        max_size_fraction: float = 0.4,
        **kwargs,
    ) -> list[dict]:
        """
        Turn the per-tile flows and cell probabilities predicted by the model into instance masks.

        Args:
            outputs:
                Model outputs with `logits` of shape `(num_tiles, 3, tile_size, tile_size)` holding (flow_y, flow_x,
                cell probability logit) per tile.
            target_sizes (`list[tuple[int, int]]` or `torch.Tensor`):
                Original `(height, width)` of each image, in the order they were preprocessed.
            cellprob_threshold (`float`, *optional*, defaults to 0.0):
                Pixels with a cell probability logit above this threshold are considered part of a cell.
            flow_threshold (`float`, *optional*, defaults to 0.4):
                Masks whose flows, recomputed from the mask shape, differ from the predicted flows by a mean squared
                error above this threshold are removed. Set to 0 or `None` to skip this (slower) quality check.
            num_iterations (`int`, *optional*, defaults to 200):
                Number of Euler steps used to follow the flows.
            min_size (`int`, *optional*, defaults to 15):
                Masks with fewer pixels are removed. Set to 0 to keep all masks.
            max_size_fraction (`float`, *optional*, defaults to 0.4):
                Masks covering more than this fraction of the image are removed.

            kwargs:
                Ignored, accepts the generic thresholds passed by the `image-segmentation` pipeline.

        Returns:
            `list[dict]`: One dict per image with `segmentation` (`(height, width)` long tensor, 0 = background,
            cells labelled 1..N), `segments_info` (one `{"id", "label_id", "score"}` per cell, the score being its
            mean cell probability), `flows` (`(2, height, width)`) and `cell_probability` (`(height, width)` logits).
        """
        # Vendored from `cellpose/dynamics.py::compute_masks` (2D): follow flows, cluster, flow QC, fill holes, min size
        # ponytail: 2D only, no 3D dynamics or z-stitching
        results = []
        logits = outputs.logits
        if logits.device.type == "mps" and not is_torch_greater_or_equal("2.13"):
            logits = logits.cpu()  # int64 `scatter_reduce` is only supported on MPS from torch 2.13
        for stitched in self.stitch_tiles(logits, target_sizes):
            flows, cell_probability = stitched[:2], stitched[2]
            foreground = cell_probability > cellprob_threshold
            masks = torch.zeros(foreground.shape, dtype=torch.long, device=foreground.device)
            if foreground.any():
                inds = torch.nonzero(foreground).T
                points = _follow_flows(flows * foreground / 5.0, inds, num_iterations).int()
                masks = _get_masks(points, inds, tuple(foreground.shape), max_size_fraction)
                if flow_threshold and masks.max() > 0:
                    masks = _remove_bad_flow_masks(masks, flows, flow_threshold)
                masks = torch.where(_mask_sizes(masks)[masks] < min_size, 0, masks)
                masks = _fill_holes(_renumber(masks))
                masks = torch.where(_mask_sizes(masks)[masks] < min_size, 0, masks)
                masks = _renumber(masks)
            counts = _mask_sizes(masks)
            scores = torch.bincount(masks.flatten(), weights=cell_probability.sigmoid().flatten()) / counts
            segments_info = [{"id": i, "label_id": 0, "score": scores[i].item()} for i in range(1, len(counts))]
            results.append(
                {
                    "segmentation": masks,
                    "segments_info": segments_info,
                    "flows": flows,
                    "cell_probability": cell_probability,
                }
            )
        return results


__all__ = ["CellposeImageProcessor"]
