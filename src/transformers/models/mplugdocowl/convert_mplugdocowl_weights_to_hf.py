# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    MPLUGDocOwlConfig,
    MPLUGDocOwlForConditionalGeneration,
    MPLUGDocOwlProcessor,
)
from transformers.models.mplugdocowl.image_processing_mplugdocowl import MPLUGDocOwlImageProcessor


KEYS_TO_MODIFY_MAPPING = {
    r"model\.vision_model\.embeddings\.position_embedding": r"vision_tower.vision_model.embeddings.position_embedding",
    r"model\.vision_model\.encoder\.layers\.(\d+)\.input_layernorm": r"vision_tower.vision_model.encoder.layers.\1.layer_norm1",
    r"model\.vision_model\.encoder\.layers\.(\d+)\.post_attention_layernorm": r"vision_tower.vision_model.encoder.layers.\1.layer_norm2",
    r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn.dense": r"vision_tower.vision_model.encoder.layers.\1.self_attn.out_proj",
    r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn.query_key_value": r"vision_tower.vision_model.encoder.layers.\1.self_attn.q_v_k_proj",
    r"model\.vision_model\.embeddings\.pre_layernorm": r"vision_tower.vision_model.embeddings.pre_layernorm",
    r"model\.vision_model\.embeddings\.patch_embed": r"vision_tower.vision_model.embeddings.patch_embedding",
    r"model\.vision_model\.embeddings\.cls_token": r"vision_tower.vision_model.embeddings.class_embedding",
    r"model\.vision_model\.": r"vision_tower.vision_model.",
    r"model\.layers\.": r"language_model.model.layers.",
    r"model\.mm_projector": r"multi_modal_projector",
    r"lm_head": r"language_model.lm_head",
    r"model\.norm\.": r"language_model.model.norm.",
    r"model\.embed_tokens": r"language_model.model.embed_tokens",
    r"model\.vision2text": r"multi_modal_projector",
    r"ln_q": r"layer_norm",
}


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        original_key = key
        for pattern, replacement in KEYS_TO_MODIFY_MAPPING.items():
            if re.search(pattern, key):
                key = re.sub(pattern, replacement, key)

        new_state_dict[key] = value
        print(f"Converted {original_key} to {key}")
    return new_state_dict


def convert_mplugdocowl_llama_to_hf(
    text_model_id, vision_model_id, output_hub_path, old_state_dict_id, pretrained=True
):
    if not pretrained:
        torch.set_default_dtype(torch.float16)
        text_config = AutoConfig.from_pretrained(text_model_id)

        tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=False)
        tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

        image_processor = MPLUGDocOwlImageProcessor()
        processor = MPLUGDocOwlProcessor(tokenizer=tokenizer, image_processor=image_processor)
        config = MPLUGDocOwlConfig(text_config=text_config)
        config.pad_token_id = 32001

        with torch.device("cuda:0"):
            model = MPLUGDocOwlForConditionalGeneration(config).eval()

        # Pad to 64 for performance reasons
        pad_shape = 64

        state_dict_path = hf_hub_download(old_state_dict_id, "pytorch_model.bin")

        state_dict = torch.load(state_dict_path, map_location="cpu")

        state_dict = convert_state_dict_to_hf(state_dict)

        state_dict["multi_modal_projector.reducer_before.0.weight"] = state_dict[
            "multi_modal_projector.reducer_before.0.weight"
        ].contiguous()
        state_dict["multi_modal_projector.reducer.weight"] = state_dict[
            "multi_modal_projector.reducer.weight"
        ].contiguous()

        model.load_state_dict(state_dict, strict=True, assign=True)

        pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we resize the model
        model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
        model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
            tuple(
                (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))
            ),
            dim=0,
        )
        model.language_model.lm_head.weight.data[32000:] = torch.stack(
            tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
            dim=0,
        )
        model.to(torch.float16)
        model.save_pretrained("tmp/hf_models/mplugdocowl1.5-Chat-hf/")
        processor.save_pretrained("tmp/hf_models/mplugdocowl1.5-Chat-hf")
    else:
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("tmp/hf_models/mplugdocowl1.5-Chat-hf")
        model.to(torch.float16)
        processor = MPLUGDocOwlProcessor.from_pretrained("tmp/hf_models/mplugdocowl1.5-Chat-hf")
    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)


def main():
    parser = argparse.ArgumentParser(
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
    convert_mplugdocowl_llama_to_hf(
        args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id
    )


if __name__ == "__main__":
    main()
