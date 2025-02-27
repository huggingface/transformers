# Copyright 2025 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
import os
import re

import torch
from accelerate import init_empty_weights

from transformers import (
    CosmosConfig,
    CosmosForConditionalGeneration,
    CosmosVideoProcessor,
    GenerationConfig,
)


"""
Sample usage:

```
python src/transformers/models/cosmos/convert_cosmos_weights_to_hf.py \
    --vq_model_id BAAI/-VisionTokenizer --llm_model_id BAAI/-Chat --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import ForConditionalGeneration, Emu3Processor

model = ForConditionalGeneration.from_pretrained("/output/path")
processor = Emu3Processor.from_pretrained("/output/path")
```

"""


KEYS_TO_MODIFY_MAPPING = {
    # rename LM keys
    r"model\.output": "lm_head",
    r"^model\.": "model.language_model.",
    "tok_embeddings": "embed_tokens",
    r"attention\.wq": "self_attn.q_proj",
    r"attention\.wk": "self_attn.k_proj",
    r"attention\.wv": "self_attn.v_proj",
    r"attention\.wo": "self_attn.o_proj",
    r"attention\.q_norm": "self_attn.q_norm",
    r"attention\.k_norm": "self_attn.k_norm",
    "attention_norm": "post_attention_layernorm",
    "ffn_norm": "input_layernorm",
    r"feed_forward\.w1": "mlp.up_proj",
    r"feed_forward\.w2": "mlp.down_proj",
    r"feed_forward\.w3": "mlp.gate_proj",
    # rename autoencoder keys
    "^encoder": "model.vqmodel.encoder",
    "^decoder": "model.vqmodel.decoder",
    "^post_quant_conv": "model.vqmodel.post_quant_conv",
    "^quant_conv": "model.vqmodel.quant_conv",
    "^quantize": "model.vqmodel.quantize",
    "patcher3d": "patch",
    # isolate down/mid/up into separate classes for readability
    r"\.down\.": ".down_block.down.",
    r"\.up\.": ".up_block.up.",
    r"\.mid\.": ".middle_block.",
    # move the attention norms outside of attention modules
    r"attn_1\.0\.norm": "attn.attn_norm",
    r"attn_1\.1\.norm": "attn.attn_norm",
    # rename QKV proj for the VQ-VAE model because we use SiglipAttention
    r"attn_1\.0\.k\.conv3d": "attn.attn_1.k_proj",
    r"attn_1\.0\.q\.conv3d": "attn.attn_1.q_proj",
    r"attn_1\.0\.v\.conv3d": "attn.attn_1.v_proj",
    r"attn_1\.0\.proj_out\.conv3d": "attn.attn_1.out_proj",
    r"attn_1\.1\.k\.conv3d": "attn.attn_2.k_proj",
    r"attn_1\.1\.q\.conv3d": "attn.attn_2.q_proj",
    r"attn_1\.1\.v\.conv3d": "attn.attn_2.v_proj",
    r"attn_1\.1\.proj_out\.conv3d": "attn.attn_2.out_proj",
}


def convert_state_dict_to_hf(old_state_dict, new_state_dict):
    for key, value in old_state_dict.items():
        # convert conv layers in attn to linear
        if any(name in key for name in [".q.conv3d", ".k.conv3d", ".v.conv3d", ".proj_out.conv3d"]):
            value = value.squeeze()

        for old_pattern, new_pattern in KEYS_TO_MODIFY_MAPPING.items():
            key = re.sub(old_pattern, new_pattern, key)

        if any(name in key for name in [".quantizer.", ".patch.", "unpatch"]):
            continue

        new_state_dict[key] = value
    return new_state_dict


def convert_model(autoencoder_weights_path, llm_weights_path, output_dir, hub_model_id=None, test_inference=False):
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Convert and save processor for text conditioned video generation
    # extra_special_tokens = extra_special_tokens = {
    #     "image_token": "<image>",
    #     "boi_token": "<|image start|>",
    #     "eoi_token": "<|image end|>",
    #     "video_token": "<video>",
    # }

    video_processor = CosmosVideoProcessor()
    video_processor.save_pretrained(output_dir)

    # Load models
    llm_state_dict = torch.load(f"{llm_weights_path}/model.pt", weights_only=True)
    autoencoder_state_dict = torch.jit.load(f"{autoencoder_weights_path}/autoencoder.jit").state_dict()
    config = CosmosConfig()

    with init_empty_weights():
        model = CosmosForConditionalGeneration(config=config)
        model.generation_config = GenerationConfig(
            do_sample=True,
            top_k=2048,
            # pad_token_id=processor.tokenizer.pad_token_id,
            # eos_token_id=processor.tokenizer.eos_token_id,
        )

    state_dict = {}
    state_dict = convert_state_dict_to_hf(llm_state_dict, state_dict)
    state_dict = convert_state_dict_to_hf(autoencoder_state_dict, state_dict)

    model.load_state_dict(state_dict, assign=True, strict=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    # if hub_model_id is not None:
    #     model.push_to_hub(hub_model_id)
    #     processor.push_to_hub(hub_model_id)

    if test_inference:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autoencoder_weights_path",
        help="Path to the directoory where autoencoder model was downloaded",
        default="/raid/raushan/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16",
    )
    parser.add_argument(
        "--llm_weights_path",
        help="Path to the directoory where LLM model was downloaded",
        default="/raid/raushan/Cosmos/checkpoints/Cosmos-1.0-Autoregressive-4B",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument(
        "--hub_model_id",
        help="Model ID in the hub where to push the model.",
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Whether to load the model for generation to test it's converted correctly.",
    )
    args = parser.parse_args()
    convert_model(
        autoencoder_weights_path=args.autoencoder_weights_path,
        llm_weights_path=args.llm_weights_path,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        test_inference=args.test_inference,
    )


if __name__ == "__main__":
    main()
