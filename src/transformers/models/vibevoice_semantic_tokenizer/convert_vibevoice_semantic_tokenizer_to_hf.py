# coding=utf-8
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
import torch
import json
from transformers import VibeVoiceSemanticTokenizerConfig, VibeVoiceSemanticTokenizerModel, AutoModel


def convert_checkpoint(vibevoice_model_id, config_path, push_to_hub, bfloat16):

    # 1) load original model
    full_model = AutoModel.from_pretrained(vibevoice_model_id)

    # 2) extract semantic tokenizer configuration
    if config_path is None:
        semantic_tokenizer_config = full_model.config.semantic_tokenizer_config
        semantic_tokenizer_config = semantic_tokenizer_config.to_dict()
    else:
        # load config
        with open(config_path, "r") as f:
            config = json.load(f)
        # extract semantic tokenizer configuration
        semantic_tokenizer_config = config["semantic_tokenizer_config"]

    # -- cleanup
    semantic_tokenizer_config["encoder_depths"] = list(map(int, semantic_tokenizer_config["encoder_depths"].split("-")))

    # 3) create config
    model_config = VibeVoiceSemanticTokenizerConfig(**semantic_tokenizer_config)

    # 4) create model
    model = VibeVoiceSemanticTokenizerModel(model_config)
    # -- to bfloat16
    if bfloat16:
        model = model.to(torch.bfloat16)

    # 5) load state dict of semantic tokenizer from original VibeVoice model
    original_state_dict = full_model.semantic_tokenizer.state_dict()
    missing, unexpected = model.load_state_dict(original_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")

    # TODO create audio feature extractor here??

    # push to hub
    if push_to_hub is not None:
        model.push_to_hub(push_to_hub)

"""
Can directly use VibeVoice model checkpoint
```bash
python src/transformers/models/vibevoice_semantic_tokenizer/convert_vibevoice_semantic_tokenizer_to_hf.py \
    --vibevoice_model_id microsoft/VibeVoice-1.5B \
    --push_to_hub bezzam/VibeVoiceSemanticTokenizer
```

Using config:
```bash
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P /raid/eric/vibevoice_original

python src/transformers/models/vibevoice_semantic_tokenizer/convert_vibevoice_semantic_tokenizer_to_hf.py \
    --vibevoice_model_id microsoft/VibeVoice-1.5B \
    --config_path /raid/eric/vibevoice_original/config.json \
    --push_to_hub bezzam/VibeVoiceSemanticTokenizer
```
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vibevoice_model_id", required=True, default=None, type=str, help="ID of the VibeVoice model to extract the acoustic tokenizer from.")
    parser.add_argument(
        "--config_path", default=None, type=str, help="Path to hf config.yaml of model to convert"
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.vibevoice_model_id,
        args.config_path,
        args.push_to_hub,
        bfloat16=not args.float32,
    )
