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
"""Convert the original FlexGlue checkpoint (https://github.com/cvg/FlexGlue) to the Hugging Face format."""

import argparse
import re
from pathlib import Path

import torch

from transformers import FlexGlueConfig, FlexGlueForKeypointMatching, FlexGlueImageProcessor


DEFAULT_CHECKPOINT_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth"

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Backbone: the DINOv3 ConvNeXt
    r"extractor\.backbone\.downsample_layers\.(\d+)\.(\d+)\.":           r"flexglue.backbone.model.stages.\1.downsample_layers.\2.",
    r"extractor\.backbone\.stages\.(\d+)\.(\d+)\.dwconv\.":              r"flexglue.backbone.model.stages.\1.layers.\2.depthwise_conv.",
    r"extractor\.backbone\.stages\.(\d+)\.(\d+)\.norm\.":                r"flexglue.backbone.model.stages.\1.layers.\2.layer_norm.",
    r"extractor\.backbone\.stages\.(\d+)\.(\d+)\.pwconv(\d)\.":          r"flexglue.backbone.model.stages.\1.layers.\2.pointwise_conv\3.",
    r"extractor\.backbone\.stages\.(\d+)\.(\d+)\.gamma":                 r"flexglue.backbone.model.stages.\1.layers.\2.gamma",
    # Neck: the original LinearConvHead
    r"extractor\.head\.stem\.0\.":                                       r"flexglue.neck.stem_input_norm.",
    r"extractor\.head\.stem\.1\.":                                       r"flexglue.neck.stem_conv.",
    r"extractor\.head\.stem\.2\.":                                       r"flexglue.neck.stem_output_norm.",
    r"extractor\.head\.norms\.(\d+)\.":                                  r"flexglue.neck.coarse_norms.\1.",
    r"extractor\.head\.projs\.(\d+)\.":                                  r"flexglue.neck.coarse_projections.\1.",
    r"extractor\.head\.fine_proj\.":                                     r"flexglue.neck.fine_projection.",
    r"extractor\.head\.mlp\.0\.":                                        r"flexglue.neck.mlp.layer_norm.",
    r"extractor\.head\.mlp\.1\.":                                        r"flexglue.neck.mlp.fc1.",
    r"extractor\.head\.mlp\.3\.":                                        r"flexglue.neck.mlp.fc2.",
    r"extractor\.head\.down\.":                                          r"flexglue.neck.output_projection.",
    r"extractor\.head\.out_norm\.":                                      r"flexglue.neck.output_norm.",
    # Transformer
    r"matcher\.input_proj\.":                                            r"flexglue.encoder.input_projection.",
    r"matcher\.posenc\.Wr\.":                                            r"flexglue.encoder.positional_encoder.projector.",
    r"matcher\.transformers\.(\d+)\.self_attn\.(q_norm|k_norm)\.":       r"flexglue.encoder.layers.\1.self_attention.\2.",
    r"matcher\.transformers\.(\d+)\.self_attn\.out_proj\.":              r"flexglue.encoder.layers.\1.self_attention.o_proj.",
    r"matcher\.transformers\.(\d+)\.self_attn\.message_norm\.":          r"flexglue.encoder.layers.\1.self_message_norm.",
    r"matcher\.transformers\.(\d+)\.self_attn\.ffn\.0\.":                r"flexglue.encoder.layers.\1.self_mlp.fc1.",
    r"matcher\.transformers\.(\d+)\.self_attn\.ffn\.1\.":                r"flexglue.encoder.layers.\1.self_mlp.layer_norm.",
    r"matcher\.transformers\.(\d+)\.self_attn\.ffn\.3\.":                r"flexglue.encoder.layers.\1.self_mlp.fc2.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.to_qk\.":                r"flexglue.encoder.layers.\1.cross_attention.qk_proj.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.to_v\.":                 r"flexglue.encoder.layers.\1.cross_attention.v_proj.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.to_out\.":               r"flexglue.encoder.layers.\1.cross_attention.o_proj.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.qk_norm\.":              r"flexglue.encoder.layers.\1.cross_attention.qk_norm.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.message_norm\.":         r"flexglue.encoder.layers.\1.cross_message_norm.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.ffn\.0\.":               r"flexglue.encoder.layers.\1.cross_mlp.fc1.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.ffn\.1\.":               r"flexglue.encoder.layers.\1.cross_mlp.layer_norm.",
    r"matcher\.transformers\.(\d+)\.cross_attn\.ffn\.3\.":               r"flexglue.encoder.layers.\1.cross_mlp.fc2.",
    # Match heads, one per layer
    r"matcher\.heads\.(\d+)\.to_search\.":                               r"heads.\1.search_projection.",
    r"matcher\.heads\.(\d+)\.disambiguate\.":                            r"heads.\1.covisibility.",
    r"matcher\.heads\.(\d+)\.offset_head\.(\w+)\.0\.":                   r"heads.\1.offset_head.\2.fc1.",
    r"matcher\.heads\.(\d+)\.offset_head\.(\w+)\.1\.":                   r"heads.\1.offset_head.\2.layer_norm.",
    r"matcher\.heads\.(\d+)\.offset_head\.(\w+)\.3\.":                   r"heads.\1.offset_head.\2.fc2.",
    r"matcher\.heads\.(\d+)\.(temperature|matchability)":                r"heads.\1.\2",
}
# fmt: on

# Not used by the Hugging Face model: the descriptor head only describes keypoints for other matchers.
IGNORED_KEYS = (r"extractor\.descriptor_head\.",)


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """Map each original key to its new key, `None` for ignored keys."""
    output_dict = {}
    for old_key in state_dict_keys:
        if any(re.match(pattern, old_key) for pattern in IGNORED_KEYS):
            output_dict[old_key] = None
            continue
        new_key = old_key
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_key, num_replacements = re.subn(f"^{pattern}", replacement, new_key)
            if num_replacements:
                break
        output_dict[old_key] = new_key
    return output_dict


def split_qkv(state_dict: dict[str, torch.Tensor], num_heads: int) -> dict[str, torch.Tensor]:
    """Split the fused self-attention projections, whose rows are ordered as (heads, head_dim, qkv)."""
    for key in [k for k in state_dict if ".self_attn.Wqkv." in k]:
        tensor = state_dict.pop(key)
        layer, kind = re.match(r"matcher\.transformers\.(\d+)\.self_attn\.Wqkv\.(weight|bias)", key).groups()
        tensor = tensor.reshape(num_heads, -1, 3, *tensor.shape[1:])
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            new_key = f"flexglue.encoder.layers.{layer}.self_attention.{name}.{kind}"
            state_dict[new_key] = tensor[:, :, i].reshape(-1, *tensor.shape[3:]).contiguous()
    return state_dict


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], config: FlexGlueConfig
) -> dict[str, torch.Tensor]:
    original_state_dict = dict(original_state_dict)
    state_dict = split_qkv(
        {k: v for k, v in original_state_dict.items() if ".self_attn.Wqkv." in k}, config.num_attention_heads
    )
    key_mapping = convert_old_keys_to_new_keys([k for k in original_state_dict if ".self_attn.Wqkv." not in k])
    for old_key, new_key in key_mapping.items():
        if new_key is not None:
            state_dict[new_key] = original_state_dict[old_key]
    return state_dict


@torch.no_grad()
def write_model(
    checkpoint_url: str,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    if Path(checkpoint_url).exists():
        original_state_dict = torch.load(checkpoint_url, map_location="cpu", weights_only=True)
    else:
        original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    config = FlexGlueConfig()
    state_dict = convert_state_dict(original_state_dict, config)

    print("Loading the checkpoint in a FlexGlue model.")
    with torch.device("meta"):
        model = FlexGlueForKeypointMatching(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    print("Checkpoint loaded successfully.")

    image_processor = FlexGlueImageProcessor()
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"Model and image processor saved to {output_dir}")

    if push_to_hub:
        model.push_to_hub(repo_id)
        image_processor.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default=DEFAULT_CHECKPOINT_URL,
        type=str,
        help="URL or local path of the original FlexGlue checkpoint.",
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the converted model to.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the converted model to the hub.")
    parser.add_argument("--repo_id", default="ETH-CVG/flexglue", type=str, help="Hub repository to push to.")
    args = parser.parse_args()
    write_model(args.checkpoint_url, args.output_dir, push_to_hub=args.push_to_hub, repo_id=args.repo_id)
