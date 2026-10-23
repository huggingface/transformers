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
"""Converts the original Xiaomi-Lens-0 checkpoint (https://huggingface.co/SeerRay-Lab/Xiaomi-Lens-0) to the
`xiaomi_lens0` format.

The weights keep their original names: `conversion_mapping.py` renames the mask decoder weights on the fly. Only the
config (stored as a `qwen3_5` config with an extra `mask_decoder` dict) and the processor are rewritten.

Usage:

```bash
python src/transformers/models/xiaomi_lens0/convert_xiaomi_lens0_weights_to_hf.py \
    --input_path SeerRay-Lab/Xiaomi-Lens-0 --output_path Xiaomi-Lens-0-hf
```
"""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
    XiaomiLens0Config,
    XiaomiLens0ForConditionalGeneration,
    XiaomiLens0Processor,
)


# Keys of the original `mask_decoder` config dict -> `XiaomiLens0MaskDecoderConfig` attributes
MASK_DECODER_CONFIG_MAPPING = {
    "llm_hidden_size": "hidden_size",
    "llm_intermediate_size": "intermediate_size",
    "llm_num_heads": "num_attention_heads",
    "vision_hidden_size": "vision_hidden_size",
    "mask_decoder_channels": "upscale_channels",
    "mask_decoder_embed_dim": "mask_embed_dim",
}


def convert_config(original_config: dict, mask_token_id: int) -> XiaomiLens0Config:
    text_config = {key: value for key, value in original_config["text_config"].items() if key != "model_type"}
    vision_config = {key: value for key, value in original_config["vision_config"].items() if key != "model_type"}
    mask_decoder_config = {
        MASK_DECODER_CONFIG_MAPPING[key]: value for key, value in original_config["mask_decoder"].items()
    }
    return XiaomiLens0Config(
        text_config=text_config,
        vision_config=vision_config,
        mask_decoder_config=mask_decoder_config,
        image_token_id=original_config["image_token_id"],
        video_token_id=original_config["video_token_id"],
        vision_start_token_id=original_config["vision_start_token_id"],
        vision_end_token_id=original_config["vision_end_token_id"],
        mask_token_id=mask_token_id,
        tie_word_embeddings=original_config["tie_word_embeddings"],
    )


def convert_checkpoint(input_path: str, output_path: str, push_to_hub: str | None = None):
    with open(hf_hub_download(input_path, "config.json"), encoding="utf-8") as config_file:
        original_config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    config = convert_config(original_config, mask_token_id=tokenizer.convert_tokens_to_ids("<|mask|>"))

    processor = XiaomiLens0Processor(
        image_processor=Qwen2VLImageProcessor.from_pretrained(input_path),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor.from_pretrained(input_path),
        chat_template=tokenizer.chat_template,
    )

    model, loading_info = XiaomiLens0ForConditionalGeneration.from_pretrained(
        input_path, config=config, dtype=torch.bfloat16, output_loading_info=True
    )
    if any(loading_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"The checkpoint does not match the model: {loading_info}")
    model.generation_config = GenerationConfig.from_pretrained(input_path)

    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    if push_to_hub is not None:
        model.push_to_hub(push_to_hub, private=True)
        processor.push_to_hub(push_to_hub, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="SeerRay-Lab/Xiaomi-Lens-0", help="Original checkpoint repo or path.")
    parser.add_argument("--output_path", required=True, help="Directory to save the converted checkpoint to.")
    parser.add_argument("--push_to_hub", default=None, help="Optional repo id to push the converted checkpoint to.")
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path, args.push_to_hub)
