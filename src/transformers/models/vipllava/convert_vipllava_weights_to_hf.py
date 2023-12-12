# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
    VipLlavaConfig,
    VipLlavaForConditionalGeneration,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "final_linear.0": "linear_1",
    "final_linear.2": "linear_2",
    "multi_modal_projector.clip_layernorm": "multi_modal_projector.projector_layernorm",
}


# Copied from transformers.models.llava.convert_llava_weights_to_hf.convert_state_dict_to_hf
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict


def convert_vipllava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False))
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = VipLlavaConfig(text_config=text_config)
    config.pad_token_id = 32001

    with torch.device("meta"):
        model = VipLlavaForConditionalGeneration(config)

    # Pad to 64 for performance reasons
    pad_shape = 64

    state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict_7b.bin")

    state_dict = torch.load(state_dict_path, map_location="cpu")
    state_dict = convert_state_dict_to_hf(state_dict)

    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )
    model.config.vocab_size = model.config.vocab_size + pad_shape
    model.config.text_config.vocab_size = model.config.text_config.vocab_size + pad_shape

    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)


def main():
    parser = argparse.ArgumentParser()
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
    convert_vipllava_llama_to_hf(
        args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id
    )


if __name__ == "__main__":
    main()
