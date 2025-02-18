# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import glob
import re

import torch
from huggingface_hub import file_exists, hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    JanusConfig,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/janus/convert_janus_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/janus-v1.5-7b-conv --old_state_dict_id liuhaotian/janus-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from janus.model.language_model.janus_llama import JanusLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = JanusLlamaForCausalLM.from_pretrained("liuhaotian/janus-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/janus-v1.5-7b/model_state_dict.bin")
"""

# Refactor later
vision_mappings = {
    "vision_model.vision_tower.blocks": "vision_model.vision_tower.layers",
    "vision_model.vision_tower.pos_embed": "vision_model.embeddings.position_embeddings",
    "vision_model.vision_tower.patch_embed.proj": "vision_model.embeddings.patch_embeddings.projection",
    "vision_model.vision_tower.norm": "vision_model.post_layernorm",
    "vision_model.vision_tower.attn_pool": "vision_model.head",
    "proj":"projection_layer",
    "norm":"layer_norm",
    "norm1":"layer_norm1",
    "norm2":"layer_norm2",
}


other_mappings = {
    # VQ Model prefix conversion
    "res": "block",
    "mid.0": "mid.block_1",
    "mid.1": "mid.attn_1",
    "mid.2": "mid.block_2",

    # Aligner module changes
    "layers.0": "fc1",
    "layers.2": "hidden_layers.0",
    "gen_head.output_mlp_projector": "gen_head.proj_out",
}


def convert_key(old_key, vision_mappings, other_mappings):

    # No conversion as language model is same.
    if "language_model" in old_key:
        return old_key

    new_key = old_key
    new_key = new_key.replace("gen_vision_model", "vqmodel")

    if "vision_model" in new_key:
        for old, new in vision_mappings.items():
            if re.search(rf'\b{re.escape(old)}\b', new_key):
                new_key = new_key.replace(old, new)
        new_key = new_key.replace("vision_tower", "encoder")
    else:
        for old, new in other_mappings.items():
            new_key = new_key.replace(old, new)

        if "encoder" in new_key:
            new_key = new_key.replace("conv_blocks", "down")
        elif "decoder" in new_key:
            new_key = new_key.replace("conv_blocks", "up")

    return new_key


def convert_state_dict(old_state_dict, vision_mappings, other_mappings):
    return {convert_key(k, vision_mappings, other_mappings): v for k, v in old_state_dict.items()}

old_weights = torch.hub.load_state_dict_from_url("https://huggingface.co/deepseek-ai/Janus-Pro-1B/resolve/main/pytorch_model.bin", map_location="cpu")

new_weights = convert_state_dict(old_weights,vision_mappings, other_mappings )
torch.save(new_weights, "temp/full_model.pth")


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()

if __name__ == "__main__":
    main()
