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
Conversion script to convert the original Qwen3-TTS-Tokenizer-25Hz checkpoint to Hugging Face format.

The key mapping follows the reference implementation in `qwen_tts/core/tokenizer_25hz/` of
https://github.com/QwenLM/Qwen3-TTS. The 25 Hz weights have not been published yet
(https://github.com/QwenLM/Qwen3-TTS/issues/34); until they are, the script can be exercised on a randomly
initialised reference model saved with `save_pretrained`.

Usage:

1) Download the original checkpoint:
```bash
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-25Hz --local-dir /path/to/qwen3-tts-tokenizer-25hz
```

2) Run conversion script:
```bash
python src/transformers/models/qwen3_tts_tokenizer_single_codebook/convert_qwen3_tts_tokenizer_single_codebook_to_hf.py \\
    --checkpoint_path /path/to/qwen3-tts-tokenizer-25hz \\
    --output_dir ./qwen3_tts_tokenizer_sc_hf \\
    --push_to_hub your-username/Qwen3-TTS-Tokenizer-25Hz-HF
```
"""

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers import Qwen3TTSTokenizerSingleCodebookFeatureExtractor, Qwen3TTSTokenizerSingleCodebookModel
from transformers.models.qwen3_tts_tokenizer_single_codebook.configuration_qwen3_tts_tokenizer_single_codebook import (
    Qwen3TTSTokenizerSingleCodebookConfig,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Whisper log-mel hop length in samples at 16 kHz, times the stride-2 convolution of the encoder.
ENCODER_SAMPLES_PER_FRAME = 160 * 2

ENCODER_LAYER_RENAMES = (
    ("attn_ln.", "self_attn_layer_norm."),
    ("mlp_ln.", "final_layer_norm."),
    ("attn.query.", "self_attn.q_proj."),
    ("attn.key.", "self_attn.k_proj."),
    ("attn.value.", "self_attn.v_proj."),
    ("attn.out.", "self_attn.out_proj."),
    ("mlp.0.", "fc1."),
    ("mlp.2.", "fc2."),
)

# Encoder tensors that only serve the LM path of the original model and play no role in tokenization.
ENCODER_KEYS_TO_DROP = (
    "positional_embedding",
    "ln_post.",
    "proj.",
    "audio_bos_eos_token.",
    "audio_vq_upsample.",
    "project_after_vq_pe.",
)


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


def convert_config(original_config: dict) -> Qwen3TTSTokenizerSingleCodebookConfig:
    """Build the HF config from the original `config.json`."""
    encoder = original_config["encoder_config"]
    decoder = original_config["decoder_config"]
    dit = dict(decoder["dit_config"])
    bigvgan = dict(decoder["bigvgan_config"])

    if encoder.get("audio_vq_type", "GRVQ") != "GRVQ":
        raise ValueError(f"Unsupported audio_vq_type: {encoder['audio_vq_type']}")

    num_encoder_layers = encoder["audio_vq_layers"]
    encoder_config = {
        "num_mel_bins": encoder["n_mels"],
        "hidden_size": encoder["n_state"],
        "encoder_attention_heads": encoder["n_head"],
        "encoder_ffn_dim": 4 * encoder["n_state"],
        "encoder_layers": num_encoder_layers,
        "max_source_positions": encoder["n_ctx"],
        "n_window": encoder["n_window"],
    }
    quantizer_config = {
        "hidden_size": encoder["n_state"],
        "codebook_size": encoder["audio_vq_codebook_size"],
        "codebook_dim": encoder.get("audio_vq_codebook_dim") or encoder["n_state"],
        "downsample_rate": encoder["audio_vq_ds_rate"],
    }

    rope_theta = dit.pop("rope_theta", 10000.0)
    dit["rope_parameters"] = {"rope_type": "default", "rope_theta": rope_theta}
    for key in ("model_type", "transformers_version"):
        dit.pop(key, None)
        bigvgan.pop(key, None)
    # The original vocoder uses fully causal residual blocks for the first two upsample stages.
    bigvgan["resblock_causal_modes"] = [
        "full_causal" if layer_idx <= 1 else "hybrid" for layer_idx in range(len(bigvgan["upsample_rates"]))
    ]

    return Qwen3TTSTokenizerSingleCodebookConfig(
        encoder_config=encoder_config,
        quantizer_config=quantizer_config,
        decoder_config={"dit_config": dit, "bigvgan_config": bigvgan},
        input_sample_rate=16000,
        output_sample_rate=original_config.get("output_sample_rate", 24000),
        encode_downsample_rate=ENCODER_SAMPLES_PER_FRAME * encoder["audio_vq_ds_rate"],
        decode_upsample_rate=dit["repeats"] * math.prod(bigvgan["upsample_rates"]),
    )


def remap_keys(state_dict: dict, num_encoder_layers: int) -> dict:
    """Remap original Qwen3-TTS-Tokenizer-25Hz keys to HF key names."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.tokenizer."):
            key = key.removeprefix("encoder.tokenizer.")
            if key.startswith(ENCODER_KEYS_TO_DROP):
                continue
            if key.startswith("blocks."):
                layer_idx = int(key.split(".")[1])
                if layer_idx >= num_encoder_layers:
                    continue
                for old, new in ENCODER_LAYER_RENAMES:
                    key = key.replace(old, new)
                new_state_dict["encoder." + key.replace("blocks.", "layers.")] = value
            elif key.startswith("audio_vq_downsample."):
                new_state_dict["quantizer." + key.replace("audio_vq_downsample.", "downsample.")] = value
            elif key.startswith("audio_quantizer.rvqs.0.layers.0."):
                new_state_dict["quantizer.vq." + key.removeprefix("audio_quantizer.rvqs.0.layers.0.")] = value
            elif key.startswith("audio_quantizer.rvqs.0."):
                # The original stacks the codebook buffers over its (single) quantizer.
                new_state_dict["quantizer.vq.codebook." + key.removeprefix("audio_quantizer.rvqs.0.")] = value[0]
            else:
                new_state_dict["encoder." + key] = value
        elif key == "decoder.dit.rotary_embed.inv_freq":
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def convert(checkpoint_path, output_dir, push_to_hub, max_shard_size):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path / "config.json", "r") as f:
        original_config = json.load(f)
    config = convert_config(original_config)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(checkpoint_path)
    logger.info(f"Original keys: {len(original_state_dict)}")

    logger.info("Remapping keys")
    converted_state_dict = remap_keys(original_state_dict, config.encoder_config.encoder_layers)

    model = Qwen3TTSTokenizerSingleCodebookModel(config)
    model.load_state_dict(converted_state_dict, strict=True)
    logger.info(f"Loaded {len(converted_state_dict)} tensors, no missing or unexpected keys")

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)
    feature_extractor = Qwen3TTSTokenizerSingleCodebookFeatureExtractor(
        audio_vq_ds_rate=config.quantizer_config.downsample_rate
    )
    feature_extractor.save_pretrained(str(output_path))

    if push_to_hub:
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)
        feature_extractor.push_to_hub(push_to_hub)

    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS-Tokenizer-25Hz checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted model")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Hub repo id to push the converted model to")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="Maximum shard size for saving")
    args = parser.parse_args()

    convert(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
