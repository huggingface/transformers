# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Loading optimum-intel's OpenVINO IRs: a stateful causal LM, an encoder-decoder (Whisper's, T5's), an encoder, and a
multimodal model as the multimodal split. Its text embeddings, one graph per modality and language model load as they
are (Gemma 3's, Pixtral's). A tower split into more graphs is fused into one at load, fed what optimum-intel computes
in Python between them: Qwen2-VL's and Qwen2.5-VL's merger, Qwen3-VL's position embeddings, MiniCPM-V 4.5's resampler,
Phi-3.5-vision's and Phi-4-multimodal's projection, and Phi-4-multimodal's audio tower. MiniCPM-V 4.7's vision graph
runs per crop, LFM3-VL's per image on raster patches, and LLaVA-NeXT's crops are packed with the `image_newline` its
config holds. A multimodal export laid out otherwise is refused, saying how."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils.hub import cached_file
from ...utils.import_utils import is_torch_available
from ..metadata import ExportMetadata
from .common import _architecture, _find, _repo_files


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if TYPE_CHECKING:
    import openvino

    from ..base import ExportedComponent


OPENVINO_MODEL_FILE = "openvino_model.xml"
# A multimodal export's: refused, its vision graphs split differently from the components here.
OPENVINO_LANGUAGE_MODEL_FILE = "openvino_language_model.xml"
# An encoder-decoder's two graphs, and the decoder's inputs under the names our decode graph takes (its own cross
# cache, written into state, as `export_for_generation` lays out an OpenVINO encoder-decoder).
OPENVINO_SEQ2SEQ_FILES = {"encoder_model": "openvino_encoder_model.xml", "decoder_model": "openvino_decoder_model.xml"}
_OPENVINO_SEQ2SEQ_DECODER_INPUTS = {
    "input_ids": "decoder_input_ids",
    "encoder_hidden_states": "encoder_outputs.last_hidden_state",
}
_OPENVINO_DTYPES = {
    "i64": "int64",
    "i32": "int32",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "boolean": "bool",
}
# A multimodal export's modality graphs, by the inputs the runtime feeds them: one graph per modality, whose features
# scatter straight into the prompt's placeholder rows
_OPENVINO_MODALITY_INPUTS = {
    "vision_embeddings_model": ("pixel_values",),
    "audio_embeddings_model": ("input_features",),
}
# A vision tower split in two graphs (Qwen2-VL's, Qwen2.5-VL's): the patch embeddings, then a merger taking what the
# image grid gives (rotary embeddings, attention masks, window index), computed here; fused into one graph
OPENVINO_VISION_MERGER = "vision_embeddings_merger_model"
# A vision graph run per crop (MiniCPM-V 4.7's), taking each crop's position, window and merge order
_OPENVINO_CROP_GLUE = {"position_ids", "window_index", "merge_index"}
# Qwen3-VL's third vision graph: the learned position embeddings it interpolates to each image's grid
OPENVINO_VISION_POSITIONS = "vision_embeddings_pos_model"
# MiniCPM-V 4.5's second vision graph: a resampler taking the patch features, fused like a merger; the vision graph takes
# each crop's position ids, the resampler its 2D sin-cos position embeddings, both their padding masks
OPENVINO_VISION_RESAMPLER = "resampler_model"
_OPENVINO_RESAMPLER_GLUE = {"patch_attention_mask", "position_ids", "pos_embed", "key_padding_mask"}
# What its processor returns besides the crops: their patch grids, read by the glue, and the placeholders' spans and
# video frame ids, which the placeholders already give
_OPENVINO_RESAMPLER_INPUTS = {"tgt_sizes", "image_bound", "temporal_ids"}
# Phi-3.5-vision's and Phi-4-multimodal's second vision graph: a projection of the patch features (Phi-3.5's merged
# 2x2 first), fused after them; Phi-4's vision graph also takes each crop's patch mask and position ids
OPENVINO_VISION_PROJECTION = "vision_projection_model"
_OPENVINO_PROJECTION_GLUE = {"patch_attention_mask", "patch_position_ids"}
# Phi-4-multimodal's audio tower, fused into one graph: embeddings, forward embeddings, the encoder on 500-frame chunks
# (fed their mask) and both projections, the vision one used beside an image
OPENVINO_AUDIO_GRAPHS = (
    "audio_embeddings_model",
    "audio_forward_embeddings_model",
    "audio_encoder_model",
    "audio_speech_projection_model",
    "audio_vision_projection_model",
)
_OPENVINO_AUDIO_CHUNK = 500
# What Phi-4-multimodal's processor returns besides its data, which the placeholders and glue already use
_OPENVINO_PHI4_INPUTS = {"image_attention_mask", "audio_embed_sizes", "audio_attention_mask", "input_mode"}
# The placeholder tokens of models whose config doesn't name them, as their processor writes them (Phi-3.5-vision's
# processor writes negative ids, read as its token)
_OPENVINO_PLACEHOLDERS = {
    "minicpmv": {"image_token_id": "<unk>"},
    "phi3_v": {"image_token_id": "<|image|>"},
    "phi4mm": {"image_token_id": "<|endoftext10|>", "audio_token_id": "<|endoftext11|>"},
}
_OPENVINO_MERGER_GLUE = {
    "attention_mask",
    "rotary_pos_emb",
    "window_index",
    "window_attention_mask",
    "position_index",
    "position_weights",
}


def _read_openvino(pretrained_model_name_or_path, file: str, **kwargs) -> openvino.Model:
    import openvino

    xml_file = cached_file(pretrained_model_name_or_path, file, **kwargs)
    cached_file(pretrained_model_name_or_path, file.replace(".xml", ".bin"), **kwargs)
    return openvino.Core().read_model(xml_file)


def _openvino_traced(port) -> dict[str, Any]:
    # A dynamic axis as the length of 2 a decode capture records.
    dims = [dimension.get_length() if dimension.is_static else 2 for dimension in port.get_partial_shape()]
    return {"rank": len(dims), "shape": dims, "dtype": _OPENVINO_DTYPES[port.get_element_type().get_type_name()]}


def _openvino_metadata(config, kwargs: dict[str, Any], outputs: Any, stateful: bool) -> ExportMetadata:
    """The metadata an export would have recorded for an IR: its kwargs, given as their example values (a tensor
    port's from the IR), and `outputs`, the structure it returns."""
    traced = {name: value if isinstance(value, dict) else _openvino_traced(value) for name, value in kwargs.items()}
    return ExportMetadata(
        architecture=_architecture(config),
        export_dtype="float32",
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(kwargs, 0)))),
        out_spec=treespec_dumps(tree_structure(outputs)),
        export_config={"export_format": "openvino", "dynamic": True, "stateful": stateful},
    )


def _rewire(parameter, source) -> None:
    """Feed every consumer of graph input `parameter` from `source` instead: one graph's input fed another's output."""
    for target in parameter.output(0).get_target_inputs():
        target.replace_source_output(source)


def _fuse_openvino_vision(
    embeddings: openvino.Model,
    merger: openvino.Model,
    positions: openvino.Model | None = None,
    hidden: str = "hidden_states",
) -> openvino.Model:
    """One vision graph from a tower split in two or three: the merger's `hidden` input fed the embeddings' output
    (plus, for Qwen3-VL, the position embeddings at the bilinear `position_index`, weighted by `position_weights` and
    summed), and the embeddings' input named `pixel_values`, as the runtime feeds it."""
    import openvino
    import openvino.opset13 as ops

    hidden_states = merger.input(hidden).get_node()
    source = embeddings.get_results()[0].input_value(0)
    parameters = list(embeddings.get_parameters())
    if positions is not None:
        positions.input(0).get_tensor().set_names({"position_index"})
        weights = ops.parameter([4, -1], openvino.Type.f32, name="position_weights")
        weights.output(0).get_tensor().set_names({"position_weights"})
        looked_up = positions.get_results()[0].input_value(0)
        weighted = ops.multiply(looked_up, ops.unsqueeze(weights, ops.constant(-1)))
        source = ops.add(source, ops.reduce_sum(weighted, ops.constant(0), keep_dims=False)).output(0)
        parameters += [*positions.get_parameters(), weights]
    _rewire(hidden_states, source)
    embeddings.input(0).get_tensor().set_names({"pixel_values"})
    parameters += [p for p in merger.get_parameters() if p is not hidden_states]
    return openvino.Model(merger.get_results(), parameters, "vision_embeddings_model")


def _position_interpolation(vision_config, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL's learned position embeddings, bilinearly interpolated to each image's grid: the 4 corner indices and
    weights of every patch, `[4, patches]`, already in the order the merger takes patches (merge blocks)."""
    side = int(vision_config.num_position_embeddings**0.5)
    merge = vision_config.spatial_merge_size
    indices, weights = [], []
    for temporal, height, width in grid_thw.tolist():
        rows, columns = torch.linspace(0, side - 1, height), torch.linspace(0, side - 1, width)
        row_floor, column_floor = rows.int(), columns.int()
        row_ceil, column_ceil = (row_floor + 1).clip(max=side - 1), (column_floor + 1).clip(max=side - 1)
        row_fraction, column_fraction = rows - row_floor, columns - column_floor
        corners = [
            (row_floor, column_floor),
            (row_floor, column_ceil),
            (row_ceil, column_floor),
            (row_ceil, column_ceil),
        ]
        image_indices = torch.stack([(r[:, None] * side + c[None]).flatten() for r, c in corners])
        row_weights = [1 - row_fraction, 1 - row_fraction, row_fraction, row_fraction]
        column_weights = [1 - column_fraction, column_fraction, 1 - column_fraction, column_fraction]
        image_weights = torch.stack([(r[:, None] * c[None]).flatten() for r, c in zip(row_weights, column_weights)])

        def merge_order(values):
            values = values.repeat(1, temporal).view(4, temporal, height // merge, merge, width // merge, merge)
            return values.permute(0, 1, 2, 4, 3, 5).flatten(1)

        indices.append(merge_order(image_indices))
        weights.append(merge_order(image_weights))
    return torch.cat(indices, dim=1).long(), torch.cat(weights, dim=1).float()


def _vision_glue(config, grid_thw: torch.Tensor, traced: dict) -> dict[str, torch.Tensor]:
    """What a fused Qwen-VL merger takes besides the patches, from the image grid: rotary embeddings, the attention
    masks keeping each image (and window) to itself, and the window order, as optimum-intel computes them."""
    from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids, get_vision_window_index

    vision = config.vision_config
    merge_size = vision.spatial_merge_size
    width = getattr(vision, "embed_dim", None) or vision.hidden_size
    dim = width // vision.num_heads // 2
    inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    positions = get_vision_position_ids(grid_thw, merge_size)
    cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, config)
    length = int(cu_seqlens[-1])

    def block_mask(boundaries: torch.Tensor) -> torch.Tensor:
        block = torch.bucketize(torch.arange(length), boundaries[1:], right=True)
        mask = torch.zeros(1, length, length)
        return mask.masked_fill(block[:, None] != block[None, :], float("-inf"))

    glue = {
        "rotary_pos_emb": (positions[..., None].float() * inv_freq).flatten(1),
        "attention_mask": block_mask(cu_seqlens),
    }
    if "position_index" in traced:
        glue["position_index"], glue["position_weights"] = _position_interpolation(vision, grid_thw)
    if "window_index" in traced:
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw, merge_size, vision.window_size, vision.patch_size
        )
        glue["window_index"] = window_index
        glue["window_attention_mask"] = block_mask(torch.unique_consecutive(cu_window_seqlens))
    return {name: value for name, value in glue.items() if name in traced}


def _raster_features(config, runner, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Features from a vision graph taking one image's patches in raster order, `[T, H, W, C]` (LFM3-VL's), run per
    image on the processor's flattened patches, which come in merge-block order."""
    merge = getattr(config, "vision_spatial_merge_size", None) or config.downsample_factor
    features, offset = [], 0
    for temporal, height, width in grid_thw.tolist():
        patches = pixel_values[offset : offset + temporal * height * width]
        offset += temporal * height * width
        patches = patches.view(temporal, height // merge, width // merge, merge, merge, -1)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(temporal, height, width, -1)
        image_features = runner(pixel_values=patches.float(), share_outputs=True)
        features.append(image_features.reshape(-1, image_features.shape[-1]))
    return torch.cat(features)


def _crop_features(config, runner, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
    """Features from a vision graph run per crop (MiniCPM-V 4.7's) on the processor's NaViT-packed patches
    `[1, channels, patch, patches * patch]`, each crop fed its nearest-neighbour position ids, window order and merge
    order, as optimum-intel computes them."""
    from ...vision_utils import get_vision_nearest_position_ids, get_vision_window_index

    vision = config.vision_config
    patch = vision.patch_size
    window_height, window_width = vision.window_kernel_size
    merge_height, merge_width = config.merge_kernel_size
    features, start = [], 0
    for height, width in target_sizes.tolist():
        crop = pixel_values[:1, ..., start * patch : (start + height * width) * patch]
        start += height * width
        size = torch.tensor([[height, width]])
        window_index, _ = get_vision_window_index(
            torch.nn.functional.pad(size, (1, 0), value=1),
            spatial_merge_size=1,
            window_size=window_height,
            patch_size=1,
        )
        # The window merger leaves a `(height / window, width / window)` grid, regrouped so each merge block is
        # contiguous
        rows, columns = height // window_height, width // window_width
        merge_index = torch.arange(rows * columns).view(
            rows // merge_height, merge_height, columns // merge_width, merge_width
        )
        crop_features = runner(
            pixel_values=crop.float(),
            position_ids=get_vision_nearest_position_ids(size, vision.image_size // patch),
            window_index=window_index,
            merge_index=merge_index.permute(0, 2, 1, 3).reshape(-1),
            share_outputs=True,
        )
        features.append(crop_features.reshape(-1, crop_features.shape[-1]))
    return torch.cat(features)


def _fuse_openvino_projection(embeddings: openvino.Model, projection: openvino.Model, config) -> openvino.Model:
    """Phi-3.5-vision's and Phi-4-multimodal's vision graph: each crop's patch features (Phi-3.5's merged 2x2 into the
    projection's input) projected, with the `sub_GN` row separator and `glb_GN` crop separator projected after them as
    the last two rows."""
    import numpy as np
    import openvino
    import openvino.opset13 as ops

    features = embeddings.get_results()[0].input_value(0)
    patches, width = (dim.get_length() for dim in features.get_partial_shape()[1:])
    projected_width = projection.inputs[0].get_partial_shape()[-1].get_length()
    merged = features
    if projected_width == 4 * width:
        side = int(patches**0.5) // 2
        merged = ops.reshape(merged, ops.constant(np.array([-1, side, 2, side, 2, width])), False)
        merged = ops.transpose(merged, ops.constant(np.array([0, 1, 3, 2, 4, 5])))
    merged = ops.reshape(merged, ops.constant(np.array([1, -1, projected_width])), False)
    separators = np.concatenate([np.array(config.sub_GN), np.array(config.glb_GN)[None]], axis=-2)
    merged = ops.concat([merged, ops.constant(separators.reshape(1, 2, projected_width).astype(np.float32))], 1)
    _rewire(projection.inputs[0].get_node(), merged.output(0))
    embeddings.input(0).get_tensor().set_names({"pixel_values"})
    return openvino.Model(projection.get_results(), list(embeddings.get_parameters()), "vision_embeddings_model")


def _masked_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """Phi-4-multimodal's position ids of a crop's patches: those its patch mask keeps, on its patch grid."""
    position_ids = torch.zeros(mask.numel(), dtype=torch.long)
    position_ids[mask.flatten()] = _navit_position_ids(int(mask[:, 0].sum()), int(mask[0].sum()), mask.shape[-1])
    return position_ids


def _hd_features(
    runner, pixel_values: torch.Tensor, image_sizes: torch.Tensor, image_attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Features from Phi-3.5-vision's or Phi-4-multimodal's fused vision graph, run per image on its global view and
    crops: the crops' tokens laid out as the image's grid (cut to the extent `image_attention_mask` keeps) and the
    global view's as its own, each row closed by the `sub_GN` separator, the `glb_GN` separator between them."""
    crop_size = pixel_values.shape[-1]
    features = []
    for index, (crops, (height, width)) in enumerate(zip(pixel_values, image_sizes.tolist())):
        rows, columns = height // crop_size, width // crop_size
        count = 1 + rows * columns
        feed = {"pixel_values": crops[:count].float()}
        if image_attention_mask is not None:
            masks = image_attention_mask[index, :count].bool()
            feed["patch_attention_mask"] = masks
            feed["patch_position_ids"] = torch.stack([_masked_position_ids(mask) for mask in masks])
        output = runner(**feed, share_outputs=True)
        output = output.reshape(-1, output.shape[-1])
        newline, separator, tokens = output[-2], output[-1], output[:-2]
        side = int((tokens.shape[0] // count) ** 0.5)
        grids = tokens.view(count, side, side, -1)
        crops_grid = grids[1:].view(rows, columns, side, side, -1).permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1)
        if image_attention_mask is not None:
            kept = image_attention_mask[index, 1:count, 0::2, 0::2].reshape(rows, columns, side, side)
            kept = kept.transpose(1, 2).reshape(rows * side, columns * side)
            crops_grid = crops_grid[: int(kept[:, 0].sum()), : int(kept[0].sum())]

        def with_newlines(grid):
            return torch.cat([grid, newline.expand(grid.shape[0], 1, -1)], dim=1).flatten(0, 1)

        features += [with_newlines(crops_grid), separator[None], with_newlines(grids[0])]
    return torch.cat(features)


def _fuse_openvino_audio(graphs: dict[str, openvino.Model]) -> openvino.Model:
    """Phi-4-multimodal's audio tower as one graph: its embeddings fed into its forward embeddings, the result padded
    to whole chunks and folded into them for the encoder (fed their `audio_mask`), and both projections, `speech` and
    `vision`, unfolded back to `[batch, chunks * chunk, hidden]`."""
    import numpy as np
    import openvino
    import openvino.opset13 as ops

    def feed(model, source, name=None):
        _rewire((model.input(name) if name else model.inputs[0]).get_node(), source)

    def output(model):
        return model.get_results()[0].input_value(0)

    embeddings, forward, encoder = (graphs[name] for name in OPENVINO_AUDIO_GRAPHS[:3])
    feed(forward, output(embeddings))
    hidden = output(forward)
    shape = ops.shape_of(hidden, "i64")
    chunk = ops.constant(np.array([_OPENVINO_AUDIO_CHUNK]))
    batch = ops.gather(shape, ops.constant(np.array([0])), ops.constant(0))
    length = ops.gather(shape, ops.constant(np.array([1])), ops.constant(0))
    pad = ops.mod(ops.subtract(chunk, ops.mod(length, chunk)), chunk)
    zero = ops.constant(np.array([0]))
    padded = ops.pad(hidden, ops.constant(np.array([0, 0, 0])), ops.concat([zero, pad, zero], 0), "constant")
    width = hidden.get_partial_shape()[-1].get_length()
    # Folded by the chunk count, the chunk axis left dynamic: a static one breaks the CPU plugin's kernel fusion
    chunks = ops.multiply(batch, ops.divide(ops.add(length, pad), chunk))
    target = ops.concat([chunks, ops.constant(np.array([-1, width]))], 0)
    folded = ops.reshape(padded, target, False)
    feed(encoder, folded.output(0), "audio_feature")
    results = []
    for name, projection in (
        ("speech", graphs[OPENVINO_AUDIO_GRAPHS[3]]),
        ("vision", graphs[OPENVINO_AUDIO_GRAPHS[4]]),
    ):
        feed(projection, output(encoder))
        projected = output(projection)
        hidden_size = projected.get_partial_shape()[-1].get_length()
        unfolded = ops.reshape(projected, ops.concat([batch, ops.constant(np.array([-1, hidden_size]))], 0), False)
        unfolded.output(0).get_tensor().set_names({name})
        results.append(ops.result(unfolded))
    embeddings.input(0).get_tensor().set_names({"input_features"})
    parameters = [*embeddings.get_parameters(), encoder.input("audio_mask").get_node()]
    return openvino.Model(results, parameters, "audio_embeddings_model")


def _phi4_audio_features(config, runner, input_features: torch.Tensor, sizes: torch.Tensor, vision: bool):
    """Features from Phi-4-multimodal's fused audio graph: each 500-frame chunk masked to its own frames and to the
    streaming window its config sets, the vision projection's beside an image, each clip cut to its `sizes`."""
    import numpy as np

    from ...models.phi4_multimodal.modeling_phi4_multimodal import adaptive_enc_mask

    audio = config.audio_processor["config"]
    length = -(-input_features.shape[1] // audio["time_reduction"])
    chunks = -(-length // _OPENVINO_AUDIO_CHUNK)
    starts = np.arange(0, _OPENVINO_AUDIO_CHUNK, audio["chunk_size"])
    streaming = adaptive_enc_mask(_OPENVINO_AUDIO_CHUNK, starts, left_window=audio["left_chunk"])
    kept = (torch.arange(chunks * _OPENVINO_AUDIO_CHUNK) < length).view(chunks, 1, _OPENVINO_AUDIO_CHUNK)
    mask = (streaming[None] & kept).repeat(input_features.shape[0], 1, 1)
    outputs = runner(input_features=input_features.float(), audio_mask=mask, share_outputs=True)
    features = outputs["vision" if vision else "speech"][:, :length]
    return torch.cat([clip[: int(size)] for clip, size in zip(features, sizes)])


def _sincos(dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    angles = positions[..., None].double() * omega
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _navit_position_ids(height: int, width: int, side: int) -> torch.Tensor:
    """MiniCPM-V 4.5's position ids: each patch's bucket on a `side` grid, from coordinates stepped by `arange` as its
    vision tower computes them (`i / height` rounds differently at bucket edges)."""
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    rows = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / height), boundaries, right=True)
    columns = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / width), boundaries, right=True)
    return (rows[:, None] * side + columns).flatten()


def _resampled_features(config, runner, pixel_values: list, tgt_sizes: list) -> torch.Tensor:
    """Features from a vision graph fused with its resampler (MiniCPM-V 4.5's), run per crop on the processor's crops
    `[channels, patch, patches * patch]`, each fed its position ids and the 2D sin-cos position embeddings of its
    patch grid, as optimum-intel computes them."""
    crops = [crop for image_crops in pixel_values for crop in image_crops]
    sizes = torch.cat([sizes for sizes in tgt_sizes if isinstance(sizes, torch.Tensor)])
    dim = config.hidden_size
    side = config.vision_config.image_size // config.patch_size
    features = []
    for crop, (height, width) in zip(crops, sizes.tolist()):
        rows, columns = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        # The column coordinate's half first, as the resampler was trained
        pos_embed = torch.cat([_sincos(dim // 2, columns), _sincos(dim // 2, rows)], dim=-1).flatten(0, 1)
        patches = height * width
        crop_features = runner(
            pixel_values=crop[None].float(),
            patch_attention_mask=torch.ones(1, 1, patches, dtype=torch.bool),
            position_ids=_navit_position_ids(height, width, side)[None],
            pos_embed=pos_embed[:, None].float(),
            key_padding_mask=torch.zeros(1, patches, dtype=torch.bool),
            share_outputs=True,
        )
        features.append(crop_features.reshape(-1, crop_features.shape[-1]))
    return torch.cat(features)


def _load_openvino_multimodal(
    pretrained_model_name_or_path, config, **kwargs
) -> tuple[dict[str, ExportedComponent], bool]:
    """A multimodal export laid out as this runtime's: its text embeddings, one graph per modality taking only that
    modality's data, and its stateful language model on `inputs_embeds` (Gemma 3's).

    Raises:
        `ValueError`: for one laid out otherwise, saying how.
    """
    from ...modeling_outputs import CausalLMOutputWithPast
    from ..base import ExportedComponent

    graphs = {
        file.removeprefix("openvino_").removesuffix(".xml")
        for file in _repo_files(pretrained_model_name_or_path, **kwargs)
        if file.startswith("openvino_") and file.endswith("_model.xml")
    }

    def read(name):
        return _read_openvino(pretrained_model_name_or_path, f"openvino_{name}.xml", **kwargs)

    reason = None
    merger = positions = projection = None
    hidden, merger_glue = "hidden_states", _OPENVINO_MERGER_GLUE
    if OPENVINO_VISION_RESAMPLER in graphs:
        merger = read(OPENVINO_VISION_RESAMPLER)
        hidden, merger_glue = "image_feature", _OPENVINO_RESAMPLER_GLUE
        graphs.discard(OPENVINO_VISION_RESAMPLER)
    elif OPENVINO_VISION_PROJECTION in graphs and getattr(config, "sub_GN", None) is not None:
        projection = read(OPENVINO_VISION_PROJECTION)
        graphs.discard(OPENVINO_VISION_PROJECTION)
    elif OPENVINO_VISION_MERGER in graphs:
        merger = read(OPENVINO_VISION_MERGER)
        if {port.get_any_name() for port in merger.inputs} - {"hidden_states"} <= _OPENVINO_MERGER_GLUE:
            graphs.discard(OPENVINO_VISION_MERGER)
            if OPENVINO_VISION_POSITIONS in graphs:
                positions = read(OPENVINO_VISION_POSITIONS)
                graphs.discard(OPENVINO_VISION_POSITIONS)
        else:
            merger = None
    audio = None
    if set(OPENVINO_AUDIO_GRAPHS) <= graphs:
        audio = _fuse_openvino_audio({name: read(name) for name in OPENVINO_AUDIO_GRAPHS})
        graphs -= set(OPENVINO_AUDIO_GRAPHS[1:])
    placeholders = {
        field: token
        for field, token in _OPENVINO_PLACEHOLDERS.get(getattr(config, "model_type", None), {}).items()
        if getattr(config, field, None) is None
    }
    if placeholders:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(cached_file(pretrained_model_name_or_path, "tokenizer.json", **kwargs))
        for field, token in placeholders.items():
            setattr(config, field, tokenizer.token_to_id(token))
    if extra := sorted(graphs - {"text_embeddings_model", "language_model", *_OPENVINO_MODALITY_INPUTS}):
        reason = (
            f"its vision side is split into more graphs than one per modality ({', '.join(extra)}), with what flows "
            "between them (rotary embeddings, attention masks, window indices from the image grid) computed in Python "
            "by code specific to the model, while this runtime takes one graph per modality, whose features scatter "
            "straight into the prompt"
        )
    elif getattr(config, "image_grid_pinpoints", None) is not None and getattr(config, "image_newline", None) is None:
        reason = "its image crops are packed with an `image_newline` weight neither its graphs nor its config hold"
    elif getattr(config, "image_token_id", None) is None:
        reason = "its config names no image placeholder token for the features to fill"
    components = {}
    for name, input_names in _OPENVINO_MODALITY_INPUTS.items():
        if reason is not None or name not in graphs:
            continue
        program = audio if name == "audio_embeddings_model" and audio is not None else read(name)
        if name == "vision_embeddings_model" and merger is not None:
            program = _fuse_openvino_vision(program, merger, positions, hidden)
        if name == "vision_embeddings_model" and projection is not None:
            program = _fuse_openvino_projection(program, projection, config)
        for port in program.inputs:
            # The image grid, which the runtime feeds a modality graph as `grid_thw`
            if port.get_any_name() == "image_grid_thw":
                port.get_tensor().set_names({"grid_thw"})
        port_names = [port.get_any_name() for port in program.inputs]
        glue_ports = {"grid_thw", "audio_mask"} | (merger_glue if merger is not None else set())
        if projection is not None:
            glue_ports |= _OPENVINO_PROJECTION_GLUE
        if set(port_names) >= _OPENVINO_CROP_GLUE:
            glue_ports |= _OPENVINO_CROP_GLUE
        data_ports = [port for port in port_names if port not in glue_ports]
        if data_ports != list(input_names):
            reason = f"its {name} takes {port_names}, not just {list(input_names)}"
            break
        # A merger returning deepstack features too (Qwen3-VL's) returns them as the runtime reads them
        outputs = [port.get_any_name() for port in program.outputs]
        if len(outputs) == 1:
            returns = torch.zeros(1)
        elif name == "audio_embeddings_model":
            # Phi-4-multimodal's fused audio graph, returning each projection
            returns = dict.fromkeys(outputs, torch.zeros(1))
        else:
            returns = {"features": torch.zeros(1), "deepstack_features": torch.zeros(1)}
        metadata = _openvino_metadata(config, {port: program.input(port) for port in port_names}, returns, False)
        components[name] = ExportedComponent(program, metadata)
    if reason is not None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-intel multimodal export this runtime can't run: {reason}. "
            "Re-export it with `OpenVINOExporter().export_for_generation`, which splits it into one graph per modality."
        )
    text = read("text_embeddings_model")
    text.input(0).get_tensor().set_names({"input_ids"})
    components["text_embeddings_model"] = ExportedComponent(
        text, _openvino_metadata(config, {"input_ids": text.input(0)}, torch.zeros(1), False)
    )
    language = _read_openvino(pretrained_model_name_or_path, OPENVINO_LANGUAGE_MODEL_FILE, **kwargs)
    inputs = {port.get_any_name(): port for port in language.inputs if port.get_any_name() != "beam_idx"}
    inputs["cache"] = {"container": "DynamicCache"}
    if "deepstack_visual_embeds" in inputs:
        # By layer, as the runtime builds them; `_OpenVINOGenerationMixin._call` stacks them as the graph takes them
        layers = len(getattr(config.vision_config, "deepstack_visual_indexes", ()))
        port = _openvino_traced(inputs["deepstack_visual_embeds"])
        inputs["deepstack_visual_embeds"] = {"items": [{**port, "rank": 2, "shape": port["shape"][1:]}] * layers}
    metadata = _openvino_metadata(config, inputs, CausalLMOutputWithPast(logits=torch.zeros(1)), stateful=True)
    components["language_model"] = ExportedComponent(language, metadata)
    return components, True


def _load_openvino(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A stateful causal LM (inputs `input_ids`, `attention_mask`, `position_ids`, `beam_idx`; output `logits`; the
    cache as state), an encoder-decoder (an encoder graph, and a decoder that writes its own cross cache into state), or
    an encoder."""
    from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
    from ..base import ExportedComponent

    if _find(pretrained_model_name_or_path, (OPENVINO_LANGUAGE_MODEL_FILE,), **kwargs) is not None:
        return _load_openvino_multimodal(pretrained_model_name_or_path, config, **kwargs)
    if _find(pretrained_model_name_or_path, (OPENVINO_SEQ2SEQ_FILES["decoder_model"],), **kwargs) is not None:
        encoder = _read_openvino(pretrained_model_name_or_path, OPENVINO_SEQ2SEQ_FILES["encoder_model"], **kwargs)
        decoder = _read_openvino(pretrained_model_name_or_path, OPENVINO_SEQ2SEQ_FILES["decoder_model"], **kwargs)
        for port in decoder.inputs:
            if (name := _OPENVINO_SEQ2SEQ_DECODER_INPUTS.get(port.get_any_name())) is not None:
                port.get_tensor().set_names({name})
        hidden = BaseModelOutput(last_hidden_state=torch.zeros(1))
        encoder_kwargs = {port.get_any_name(): port for port in encoder.inputs}
        decoder_kwargs = {
            "decoder_input_ids": decoder.input("decoder_input_ids"),
            "encoder_outputs": {"container": "BaseModelOutput"},
            "cache": {"container": "EncoderDecoderCache"},
        }
        encoder_metadata = _openvino_metadata(config, encoder_kwargs, hidden, stateful=False)
        decoder_metadata = _openvino_metadata(config, decoder_kwargs, Seq2SeqLMOutput(logits=torch.zeros(1)), True)
        # The output class of the encoder, which the runtime reads off the decode graph's input spec.
        decoder_metadata.in_spec = treespec_dumps(
            tree_structure(((), {"decoder_input_ids": 0, "encoder_outputs": hidden, "cache": 0}))
        )
        components = {
            "encoder_model": ExportedComponent(encoder, encoder_metadata),
            "decoder_model": ExportedComponent(decoder, decoder_metadata),
        }
        return components, True

    program = _read_openvino(pretrained_model_name_or_path, OPENVINO_MODEL_FILE, **kwargs)
    inputs = {port.get_any_name(): port for port in program.inputs if port.get_any_name() != "beam_idx"}
    output_names = [port.get_any_name() for port in program.outputs]
    if len(inputs) == len(program.inputs):
        outputs = dict.fromkeys(output_names, torch.zeros(1))
        return {"model": ExportedComponent(program, _openvino_metadata(config, inputs, outputs, False))}, False
    if output_names[:1] != ["logits"]:
        raise ValueError(
            f"{pretrained_model_name_or_path} keeps a state but doesn't return logits (outputs {output_names}): only "
            "a stateful causal LM loads from optimum-intel. Re-export it with `OpenVINOExporter`."
        )
    inputs["cache"] = {"container": "DynamicCache"}
    metadata = _openvino_metadata(config, inputs, CausalLMOutputWithPast(logits=torch.zeros(1)), stateful=True)
    return {"model": ExportedComponent(program, metadata)}, True


class _OpenVINOGenerationMixin:
    """What optimum-intel's multimodal graphs leave around them: the glue between a fused vision or audio tower's
    graphs, its processors' inputs under the names the graphs take, and deepstack features stacked as its language
    model takes them."""

    def _features(self, spec, runner, kwargs: dict):
        """What an optimum-intel vision graph leaves around it: a fused Qwen-VL one also takes what its image grid gives
        (`_vision_glue`), LFM3-VL's runs per image on raster patches (`_raster_features`), and LLaVA-NeXT's crops are
        packed with the `image_newline` its config holds."""
        traced = runner.metadata.kwargs
        grid = kwargs.get(spec.grid_key) if spec.grid_key is not None else None
        if "rotary_pos_emb" in traced and grid is not None:
            kwargs = {**kwargs, **_vision_glue(self.config, grid, traced)}
        pixel_values = kwargs.get("pixel_values")
        if "audio_mask" in traced:
            vision = kwargs.get("pixel_values") is not None
            features = _phi4_audio_features(
                self.config, runner, kwargs["input_features"], kwargs["audio_embed_sizes"], vision
            )
            return features, None
        if getattr(self.config, "sub_GN", None) is not None and kwargs.get("image_sizes") is not None:
            image_attention_mask = kwargs.get("image_attention_mask") if "patch_attention_mask" in traced else None
            return _hd_features(runner, pixel_values, kwargs["image_sizes"], image_attention_mask), None
        if "pos_embed" in traced and kwargs.get("tgt_sizes") is not None:
            return _resampled_features(self.config, runner, pixel_values, kwargs["tgt_sizes"]), None
        if "merge_index" in traced and kwargs.get("target_sizes") is not None:
            return _crop_features(self.config, runner, pixel_values, kwargs["target_sizes"]), None
        if traced.get("pixel_values", {}).get("rank") == 4 and grid is not None and pixel_values.dim() == 2:
            return _raster_features(self.config, runner, pixel_values, grid), None
        features, layers = super()._features(spec, runner, kwargs)
        newline = getattr(self.config, "image_newline", None)
        if newline is not None and kwargs.get("image_sizes") is not None and features.dim() == 3:
            from ..decompose import pack_anyres_features

            features = pack_anyres_features(self.config, features, kwargs["image_sizes"], torch.tensor(newline))
        return features, layers

    def embed(self, **kwargs) -> tuple[dict, dict]:
        """Also read negative ids (Phi-3.5-vision's processor's image placeholders) as the image placeholder token, and
        Phi-4-multimodal's processor's data under the names the modality graphs take (dropping the empty ones)."""
        for given, name in (("input_image_embeds", "pixel_values"), ("input_audio_embeds", "input_features")):
            if given in kwargs:
                value = kwargs.pop(given)
                if value is not None and value.numel() > 0:
                    kwargs[name] = value
        input_ids = kwargs.get("input_ids")
        if (
            input_ids is not None
            and getattr(self.config, "image_token_id", None) is not None
            and (input_ids < 0).any()
        ):
            kwargs["input_ids"] = input_ids.masked_fill(input_ids < 0, self.config.image_token_id)
        return super().embed(**kwargs)

    def _call(self, **kwargs) -> Any:
        deepstack = kwargs.get("deepstack_visual_embeds")
        if isinstance(deepstack, (list, tuple)):
            # optimum-intel's language model takes the layers stacked
            kwargs["deepstack_visual_embeds"] = torch.stack(list(deepstack))
        return super()._call(**kwargs)

    def _validate_model_kwargs(self, model_kwargs: dict):
        """Also accept MiniCPM-V 4.5's crop inputs, for a vision graph fused with its resampler, and
        Phi-4-multimodal's processor's."""
        accepted = set()
        vision = self.components.get("vision_embeddings_model")
        if vision is not None and "pos_embed" in vision.metadata.kwargs:
            accepted |= _OPENVINO_RESAMPLER_INPUTS
        if getattr(self.config, "model_type", None) == "phi4mm":
            accepted |= _OPENVINO_PHI4_INPUTS | {"input_image_embeds", "input_audio_embeds"}
        super()._validate_model_kwargs({key: value for key, value in model_kwargs.items() if key not in accepted})
