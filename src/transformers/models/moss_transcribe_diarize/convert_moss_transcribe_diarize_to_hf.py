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

import argparse
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    MossTranscribeDiarizeConfig,
    MossTranscribeDiarizeForConditionalGeneration,
    MossTranscribeDiarizeProcessor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Whisper audio encoder, renamed to match `MossTranscribeDiarizeModel.audio_tower`.
    r"^model\.whisper_encoder\.":     r"model.audio_tower.",
    # VQAdaptor `nn.Sequential` (Linear, SiLU, Linear, LayerNorm), renamed to match
    # `MossTranscribeDiarizeMultiModalProjector` (linear_1, act, linear_2, norm).
    r"^model\.vq_adaptor\.layers\.0\.": r"model.multi_modal_projector.linear_1.",
    r"^model\.vq_adaptor\.layers\.2\.": r"model.multi_modal_projector.linear_2.",
    r"^model\.vq_adaptor\.layers\.3\.": r"model.multi_modal_projector.norm.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key = re.sub(pattern, replacement, new_key)
    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}
    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")
    return new_state_dict


def load_original_state_dict(checkpoint_dir: Path) -> dict[str, Any]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    state_dict = {}
    for shard_file in shard_files:
        state_dict.update(load_file(checkpoint_dir / shard_file))
    return state_dict


def convert_checkpoint(checkpoint_dir, push_to_hub, bfloat16):
    dtype = torch.bfloat16 if bfloat16 else torch.float32
    checkpoint_dir = Path(checkpoint_dir)

    # `adaptor_input_dim` is derived.
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "r") as f:
        raw_config_dict = json.load(f)
    raw_config_dict.pop("adaptor_input_dim", None)
    with open(config_path, "w") as f:
        json.dump(raw_config_dict, f, indent=2)

    # 1) Load original state dict, config, generation config and processor.
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    original_state_dict = load_original_state_dict(checkpoint_dir)
    config = MossTranscribeDiarizeConfig.from_pretrained(checkpoint_dir)
    processor = MossTranscribeDiarizeProcessor.from_pretrained(checkpoint_dir)

    processor.tokenizer.padding_side = "left"
    processor.tokenizer.init_kwargs["padding_side"] = "left"

    # 2) Convert state dict to match HF model structure
    logger.info("Converting state dict")
    converted_state_dict = convert_state_dict(original_state_dict)

    # 3) Create model and load weights
    logger.info("Creating MossTranscribeDiarizeForConditionalGeneration model")
    model = MossTranscribeDiarizeForConditionalGeneration(config).to(dtype)

    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    # `lm_head.weight` is tied to `model.language_model.embed_tokens.weight` and isn't saved in the checkpoint.
    missing = [key for key in missing if key != "lm_head.weight"]
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"Missing keys: {missing}")
    model.tie_weights()

    generation_config_path = checkpoint_dir / "generation_config.json"
    if generation_config_path.exists():
        model.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)

    if push_to_hub:
        logger.info(f"Pushing to hub as {push_to_hub}")
        processor.push_to_hub(push_to_hub)
        model.push_to_hub(push_to_hub)

        gc.collect()
        logger.info("Verifying conversion by reloading model")
        AutoProcessor.from_pretrained(push_to_hub)
        AutoModelForCausalLM.from_pretrained(push_to_hub, dtype=torch.bfloat16, device_map="auto")
        logger.info("Model reloaded successfully!")
        logger.info("Conversion complete!")


"""
Conversion script to convert the original MOSS-Transcribe-Diarize checkpoint (Whisper encoder + VQAdaptor + Qwen3,
using the original repo's `trust_remote_code` module names) into an `MossTranscribeDiarizeForConditionalGeneration`
checkpoint using the natively-supported 🤗 Transformers modeling code.

1) download the original checkpoint (config, weights, tokenizer and processor files) locally, e.g.:
```bash
huggingface-cli download itazap/MOSS-Transcribe-Diarize-HF --local-dir /raid/moss_transcribe_diarize/original
```

2) run conversion with:
```bash
python src/transformers/models/moss_transcribe_diarize/convert_moss_transcribe_diarize_to_hf.py \
    --checkpoint_dir /raid/moss_transcribe_diarize/original \
    --push_to_hub itazap/MOSS-Transcribe-Diarize-HF
```

A checkpoint will be pushed to `itazap/MOSS-Transcribe-Diarize-HF` on the HF Hub.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        default=None,
        type=str,
        help="Path to a local directory with the original MOSS-Transcribe-Diarize checkpoint (config, safetensors "
        "weights, tokenizer and processor files).",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_dir,
        args.push_to_hub,
        bfloat16=not args.float32,
    )
