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
Conversion script to convert the original Qwen3-TTS checkpoint to Hugging Face format.

Usage:

1) Download the original Qwen3-TTS model checkpoint:
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir /path/to/qwen3-tts
```

2) Run conversion script (with optional `push_to_hub` argument):
```bash
python src/transformers/models/qwen3_tts/convert_qwen3_tts_to_hf.py \
    --checkpoint_path /path/to/qwen3-tts \
    --output_dir ./qwen3_tts_hf \
    --push_to_hub your-username/Qwen3-TTS-12Hz-0.6B-Base-HF
```
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)
from transformers.models.qwen3_tts_tokenizer_multi_codebook.configuration_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookConfig,
)
from transformers.models.qwen3_tts_tokenizer_multi_codebook.modeling_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookModel,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_original_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load original checkpoint weights from safetensors file(s)."""
    # Single file checkpoint
    single_file = checkpoint_path / "model.safetensors"
    if single_file.exists():
        logger.info(f"Loading checkpoint from {single_file}")
        return load_file(str(single_file))

    # Sharded checkpoint
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
            shard_dict = load_file(str(shard_path))
            state_dict.update(shard_dict)
        return state_dict

    raise FileNotFoundError(
        f"Could not find 'model.safetensors' or 'model.safetensors.index.json' in {checkpoint_path}"
    )


def create_config_from_checkpoint(checkpoint_path: Path) -> Qwen3TTSConfig:
    """Create HF config from the original checkpoint's config.json."""
    config_path = checkpoint_path / "config.json"

    if not config_path.exists():
        logger.warning("No config.json found, using default configuration")
        return Qwen3TTSConfig()

    with open(config_path, "r") as f:
        original_config = json.load(f)

    talker_dict = original_config.get("talker_config", {})
    code_predictor_dict = talker_dict.pop("code_predictor_config", {})

    # Clean up keys not in our config
    keys_to_keep_talker = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "attention_bias",
        "use_sliding_window",
        "sliding_window",
        "attention_dropout",
        "num_code_groups",
        "text_hidden_size",
        "codec_eos_token_id",
        "codec_think_id",
        "codec_nothink_id",
        "codec_think_bos_id",
        "codec_think_eos_id",
        "codec_pad_id",
        "codec_bos_id",
        "spk_id",
        "spk_is_dialect",
        "codec_language_id",
        "text_vocab_size",
    }
    keys_to_keep_code_predictor = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "attention_bias",
        "use_sliding_window",
        "sliding_window",
        "attention_dropout",
        "num_code_groups",
        "layer_types",
    }

    talker_filtered = {k: v for k, v in talker_dict.items() if k in keys_to_keep_talker}
    code_predictor_filtered = {k: v for k, v in code_predictor_dict.items() if k in keys_to_keep_code_predictor}

    # Handle rope_scaling -> rope_parameters conversion
    # rope_theta is a separate key in the original config, needs to be merged into rope_parameters
    rope_scaling = talker_dict.get("rope_scaling")
    if rope_scaling is not None:
        rope_params = dict(rope_scaling)
        rope_theta = talker_dict.get("rope_theta")
        if rope_theta is not None:
            rope_params["rope_theta"] = rope_theta
        talker_filtered["rope_parameters"] = rope_params

    code_predictor_rope = code_predictor_dict.get("rope_scaling")
    if code_predictor_rope is not None:
        cp_rope_params = dict(code_predictor_rope)
        cp_rope_theta = code_predictor_dict.get("rope_theta")
        if cp_rope_theta is not None:
            cp_rope_params["rope_theta"] = cp_rope_theta
        code_predictor_filtered["rope_parameters"] = cp_rope_params

    # Pass code_predictor as a dict inside talker_dict, since configs unpack with **
    talker_filtered["code_predictor_config"] = code_predictor_filtered

    speaker_encoder_dict = original_config.get("speaker_encoder_config", {})

    config = Qwen3TTSConfig(
        talker_config=talker_filtered,
        speaker_encoder_config=speaker_encoder_dict,
        tokenizer_type=original_config.get("tokenizer_type"),
        tts_model_size=original_config.get("tts_model_size"),
        tts_model_type=original_config.get("tts_model_type"),
        im_start_token_id=original_config.get("im_start_token_id", 151644),
        im_end_token_id=original_config.get("im_end_token_id", 151645),
        tts_pad_token_id=original_config.get("tts_pad_token_id", 151671),
        tts_bos_token_id=original_config.get("tts_bos_token_id", 151672),
        tts_eos_token_id=original_config.get("tts_eos_token_id", 151673),
    )

    return config


def remap_tokenizer_multi_codebook_keys(state_dict: dict) -> dict:
    """Remap original Qwen3-TTS-Tokenizer-12Hz keys to HF key names."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # rvq_first → semantic_residual_vector_quantizer
        new_key = new_key.replace(
            "decoder.quantizer.rvq_first.",
            "decoder.quantizer.semantic_residual_vector_quantizer.",
        )
        # rvq_rest → acoustic_residual_vector_quantizer
        new_key = new_key.replace(
            "decoder.quantizer.rvq_rest.",
            "decoder.quantizer.acoustic_residual_vector_quantizer.",
        )
        # Remove the .vq. level: .vq.layers.N. → .layers.N.
        new_key = new_key.replace(".vq.layers.", ".layers.")
        # ._codebook. → .codebook.
        new_key = new_key.replace("._codebook.", ".codebook.")
        # .embedding_sum → .embed_sum
        new_key = new_key.replace(".embedding_sum", ".embed_sum")

        new_state_dict[new_key] = value

    return new_state_dict


def convert_tokenizer_multi_codebook(checkpoint_path, output_dir, push_to_hub, bfloat16, max_shard_size):
    """Convert Qwen3-TTS-Tokenizer-12Hz (multi-codebook) checkpoint to HF format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(Path(checkpoint_path))
    logger.info(f"Original keys: {len(original_state_dict)}")

    logger.info("Remapping keys")
    converted_state_dict = remap_tokenizer_multi_codebook_keys(original_state_dict)

    config = Qwen3TTSTokenizerMultiCodebookConfig()
    config.save_pretrained(str(output_path))

    dtype = torch.bfloat16 if bfloat16 else torch.float32
    model = Qwen3TTSTokenizerMultiCodebookModel(config).to(dtype)

    load_result = model.load_state_dict(converted_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}")

    # Set initialized buffers to True (not saved in original checkpoint)
    for module in model.modules():
        if hasattr(module, "initialized"):
            module.initialized.fill_(1.0)

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)

    if push_to_hub:
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)

    logger.info("Tokenizer conversion complete!")


def convert_checkpoint(checkpoint_path, output_dir, push_to_hub, bfloat16, max_shard_size):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating config from checkpoint")
    config = create_config_from_checkpoint(Path(checkpoint_path))
    config.save_pretrained(str(output_path))

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(Path(checkpoint_path))
    logger.info(f"Number of parameters in original state dict: {len(original_state_dict)}")

    # No key renaming needed — original and HF key names match
    converted_state_dict = original_state_dict

    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    logger.info(f"Creating model with dtype {dtype}")
    model = Qwen3TTSForConditionalGeneration(config).to(dtype)
    logger.info(f"Number of parameters in model state dict: {len(model.state_dict())}")

    logger.info("Loading weights into model")
    load_result = model.load_state_dict(converted_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"{len(load_result.missing_keys)} missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"{len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys}")

    logger.info(f"Saving model to {output_path}")
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)

    if push_to_hub:
        logger.info(f"Pushing to Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)

    logger.info("Verifying conversion by reloading model")
    del model, converted_state_dict, original_state_dict
    gc.collect()
    reloaded = Qwen3TTSForConditionalGeneration.from_pretrained(str(output_path))
    logger.info(f"Model reloaded successfully with {sum(p.numel() for p in reloaded.parameters())} parameters")
    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS checkpoint to Hugging Face format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original Qwen3-TTS checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted checkpoint",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'username/Qwen3-TTS-12Hz-0.6B-Base-HF')",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Whether to use float32 precision. Default is bfloat16.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2.5GB",
        help="Maximum shard size for safetensors files.",
    )
    parser.add_argument(
        "--tokenizer_12hz",
        action="store_true",
        help="Convert Qwen3-TTS-Tokenizer-12Hz (multi-codebook) instead of the main model.",
    )

    args = parser.parse_args()

    if args.tokenizer_12hz:
        convert_tokenizer_multi_codebook(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            bfloat16=not args.float32,
            max_shard_size=args.max_shard_size,
        )
        return

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        bfloat16=not args.float32,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
