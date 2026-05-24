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

"""
Conversion script to convert the original Qwen3-TTS-Tokenizer-12Hz checkpoint to Hugging Face format.

Usage:

1) Download the original Qwen3-TTS-Tokenizer-12Hz model checkpoint:
```bash
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir /path/to/qwen3-tts-tokenizer
```

2) Run conversion script:
```bash
python src/transformers/models/qwen3_tts_tokenizer_multi_codebook/convert_qwen3_tts_tokenizer_multi_codebook_to_hf.py \\
    --checkpoint_path /path/to/qwen3-tts-tokenizer \\
    --output_dir ./qwen3_tts_tokenizer_mc_hf \\
    --push_to_hub your-username/Qwen3-TTS-Tokenizer-12Hz-HF
```
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers import Qwen3TTSTokenizerMultiCodebookModel
from transformers.models.qwen3_tts_tokenizer_multi_codebook.configuration_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookConfig,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_original_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load original checkpoint weights from safetensors file(s)."""
    single_file = checkpoint_path / "model.safetensors"
    if single_file.exists():
        logger.info(f"Loading checkpoint from {single_file}")
        return load_file(str(single_file))

    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        logger.info(f"Loading sharded checkpoint from {checkpoint_path}")
        with open(index_path, "r") as f:
            index = json.load(f)

        state_dict = {}
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in shard_files:
            shard_path = checkpoint_path / shard_file
            logger.info(f"Loading shard: {shard_file}")
            state_dict.update(load_file(str(shard_path)))
        return state_dict

    raise FileNotFoundError(
        f"Could not find 'model.safetensors' or 'model.safetensors.index.json' in {checkpoint_path}"
    )


def remap_keys(state_dict: dict) -> dict:
    """Remap original Qwen3-TTS-Tokenizer-12Hz keys to HF key names."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(
            "decoder.quantizer.rvq_first.",
            "decoder.quantizer.semantic_residual_vector_quantizer.",
        )
        new_key = new_key.replace(
            "decoder.quantizer.rvq_rest.",
            "decoder.quantizer.acoustic_residual_vector_quantizer.",
        )
        new_key = new_key.replace(".vq.layers.", ".layers.")
        new_key = new_key.replace("._codebook.", ".codebook.")
        new_key = new_key.replace(".embedding_sum", ".embed_sum")
        new_state_dict[new_key] = value
    return new_state_dict


def convert(checkpoint_path, output_dir, push_to_hub, bfloat16, max_shard_size):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(Path(checkpoint_path))
    logger.info(f"Original keys: {len(original_state_dict)}")

    logger.info("Remapping keys")
    converted_state_dict = remap_keys(original_state_dict)

    config = Qwen3TTSTokenizerMultiCodebookConfig()
    config.save_pretrained(str(output_path))

    dtype = torch.bfloat16 if bfloat16 else torch.float32
    model = Qwen3TTSTokenizerMultiCodebookModel(config).to(dtype)

    load_result = model.load_state_dict(converted_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}")

    # Mark codebook entries as initialized (not stored in original checkpoint)
    for module in model.modules():
        if hasattr(module, "initialized"):
            module.initialized.fill_(1.0)

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)

    if push_to_hub:
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)

    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS-Tokenizer-12Hz checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 precision. Default is bfloat16.",
    )
    parser.add_argument("--max_shard_size", type=str, default="2.5GB")

    args = parser.parse_args()
    convert(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        bfloat16=not args.float32,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
