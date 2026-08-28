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
"""
Convert a PP-DocLayoutV4 PaddlePaddle checkpoint to the Transformers format.

The `.pdparams` file written by PaddleDetection is a plain pickle of numpy arrays, so this script only needs `torch`
and `numpy` -- do not import `paddle` here, it shadows parts of the `torch` C extension when both are loaded in the
same process.

Example:

```bash
python src/transformers/models/pp_doclayout_v4/convert_pp_doclayout_v4_to_hf.py \
    --pdparams_path 126.pdparams \
    --output_dir PP-DocLayoutV4_safetensors
```
"""

import argparse
import pickle
import re

import torch

from transformers import PPDocLayoutV4Config, PPDocLayoutV4ForObjectDetection, PPDocLayoutV4ImageProcessor


# The class order is fixed by the checkpoint, it matches `label_list` in the exported `inference.yml`.
LABELS = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]

# The HGNetV2 backbone, `conv` -> `convolution` and `bn` -> `normalization`.
BACKBONE_MAPPING = [
    (r"^backbone\.stem\.(\w+)\.conv\.", r"backbone.model.embedder.\1.convolution."),
    (r"^backbone\.stem\.(\w+)\.bn\.", r"backbone.model.embedder.\1.normalization."),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.conv\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.convolution.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.bn\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.normalization.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.conv\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.convolution.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.bn\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.normalization.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.conv\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.bn\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.",
        r"backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.",
    ),
    (r"^backbone\.stages\.(\d+)\.downsample\.conv\.", r"backbone.model.encoder.stages.\1.downsample.convolution."),
    (r"^backbone\.stages\.(\d+)\.downsample\.bn\.", r"backbone.model.encoder.stages.\1.downsample.normalization."),
]

# The `neck` becomes the hybrid encoder plus its input projections.
ENCODER_MAPPING = [
    (r"^neck\.input_proj\.", r"encoder_input_proj."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.linear1\.", r"encoder.aifi.\1.layers.\2.mlp.fc1."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.linear2\.", r"encoder.aifi.\1.layers.\2.mlp.fc2."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.norm1\.", r"encoder.aifi.\1.layers.\2.self_attn_layer_norm."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.norm2\.", r"encoder.aifi.\1.layers.\2.final_layer_norm."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.self_attn\.out_proj\.", r"encoder.aifi.\1.layers.\2.self_attn.o_proj."),
    (r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.self_attn\.", r"encoder.aifi.\1.layers.\2.self_attn."),
    (r"^neck\.(lateral_convs|downsample_convs)\.(\d+)\.bn\.", r"encoder.\1.\2.norm."),
    (r"^neck\.(lateral_convs|downsample_convs)\.", r"encoder.\1."),
    (r"^neck\.(fpn_blocks|pan_blocks)\.(\d+)\.((?:bottlenecks\.\d+\.)?conv\d)\.bn\.", r"encoder.\1.\2.\3.norm."),
    (r"^neck\.(fpn_blocks|pan_blocks)\.", r"encoder.\1."),
]

# The `transformer` becomes the decoder, its input projections and all prediction heads.
DECODER_MAPPING = [
    (r"^transformer\.input_proj\.(\d+)\.conv\.", r"decoder_input_proj.\1.0."),
    (r"^transformer\.input_proj\.(\d+)\.norm\.", r"decoder_input_proj.\1.1."),
    (r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.", r"decoder.layers.\1.encoder_attn."),
    (r"^transformer\.decoder\.layers\.(\d+)\.linear1\.", r"decoder.layers.\1.mlp.fc1."),
    (r"^transformer\.decoder\.layers\.(\d+)\.linear2\.", r"decoder.layers.\1.mlp.fc2."),
    (r"^transformer\.decoder\.layers\.(\d+)\.norm1\.", r"decoder.layers.\1.self_attn_layer_norm."),
    (r"^transformer\.decoder\.layers\.(\d+)\.norm2\.", r"decoder.layers.\1.encoder_attn_layer_norm."),
    (r"^transformer\.decoder\.layers\.(\d+)\.norm3\.", r"decoder.layers.\1.final_layer_norm."),
    (r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.", r"decoder.layers.\1.self_attn.o_proj."),
    (r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.", r"decoder.layers.\1.self_attn."),
    (r"^transformer\.query_pos_head\.", r"decoder.query_pos_head."),
    (r"^transformer\.dec_bbox_head\.", r"decoder.bbox_embed."),
    (r"^transformer\.dec_score_head\.", r"decoder.class_embed."),
    # The ROOR rules must come before the plain relative order ones, `dec_order_head` is not a suffix of
    # `dec_roor_order_head` but a naive substring match on the shorter name would still be wrong here.
    (r"^transformer\.dec_roor_order_head\.", r"decoder_roor_order_head."),
    (r"^transformer\.dec_roor_global_pointer\.", r"decoder_roor_global_pointer."),
    (r"^transformer\.dec_order_head\.", r"decoder_order_head."),
    (r"^transformer\.dec_global_pointer\.", r"decoder_global_pointer."),
    (r"^transformer\.s2r_fusion\.", r"s2r_fusion."),
    (r"^transformer\.(enc_bbox_head|enc_score_head|enc_output|denoising_class_embed)\.", r"\1."),
]

MAPPING = BACKBONE_MAPPING + ENCODER_MAPPING + DECODER_MAPPING


def rename_key(key: str) -> str:
    """Maps one PaddleDetection parameter name onto its Transformers counterpart."""
    for pattern, replacement in MAPPING:
        renamed, num_substitutions = re.subn(pattern, replacement, key)
        if num_substitutions:
            key = renamed
            break
    # Paddle names the batch norm running statistics `_mean` and `_variance`.
    key = key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
    return "model." + key


def convert_state_dict(paddle_state_dict: dict, model: PPDocLayoutV4ForObjectDetection) -> dict:
    """
    Renames, transposes and splits the Paddle parameters into a Transformers state dict.

    Whether a 2D weight has to be transposed is decided by looking up the owning module in `model`: Paddle stores
    `nn.Linear` weights as `(in_features, out_features)` while torch uses `(out_features, in_features)`. Deciding this
    from the module type rather than from a list of name substrings matters because several of the heads are square,
    where a missed transpose is silently wrong instead of a shape error.
    """
    expected_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    linear_modules = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

    state_dict = {}
    for key, value in paddle_state_dict.items():
        new_key = rename_key(key)
        tensor = torch.from_numpy(value.copy())

        # Paddle fuses the self-attention projections into a single `in_proj` parameter.
        if new_key.endswith(("in_proj_weight", "in_proj_bias")):
            prefix, suffix = new_key.rsplit(".", 1)
            suffix = "weight" if suffix == "in_proj_weight" else "bias"
            if suffix == "weight":
                tensor = tensor.T
            for name, chunk in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                state_dict[f"{prefix}.{name}.{suffix}"] = chunk.contiguous()
            continue

        if new_key not in expected_shapes:
            raise ValueError(f"{key} was renamed to {new_key}, which is not a parameter of the Transformers model.")
        if new_key.endswith(".weight") and new_key.rsplit(".", 1)[0] in linear_modules:
            tensor = tensor.T.contiguous()
        if tuple(tensor.shape) != expected_shapes[new_key]:
            raise ValueError(
                f"{key} -> {new_key}: got shape {tuple(tensor.shape)}, expected {expected_shapes[new_key]}."
            )
        state_dict[new_key] = tensor

    # Paddle does not track the number of batch norm updates, so the counters start from zero.
    for name, shape in expected_shapes.items():
        if name.endswith("num_batches_tracked") and name not in state_dict:
            state_dict[name] = torch.zeros(shape, dtype=torch.long)

    missing = sorted(set(expected_shapes) - set(state_dict))
    if missing:
        raise ValueError(f"{len(missing)} parameters were not covered by the conversion, e.g. {missing[:5]}.")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdparams_path", required=True, help="Path to the PaddleDetection `.pdparams` checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Where to write the converted model.")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repository id to push the result to.")
    args = parser.parse_args()

    with open(args.pdparams_path, "rb") as checkpoint:
        paddle_state_dict = pickle.load(checkpoint)
    # PaddleDetection stores a structured-name to parameter-name table next to the weights.
    paddle_state_dict.pop("StructuredToParameterName@@", None)

    config = PPDocLayoutV4Config(
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    model = PPDocLayoutV4ForObjectDetection(config)
    state_dict = convert_state_dict(paddle_state_dict, model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image_processor = PPDocLayoutV4ImageProcessor()

    model.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print(f"Converted {len(paddle_state_dict)} Paddle parameters into {len(state_dict)} tensors in {args.output_dir}.")

    if args.push_to_hub is not None:
        model.push_to_hub(args.push_to_hub)
        image_processor.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
