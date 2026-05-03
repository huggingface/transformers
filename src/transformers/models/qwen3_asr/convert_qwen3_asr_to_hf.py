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
Convert Qwen3 ASR or Qwen3 Forced Aligner checkpoints to Hugging Face format.

The script auto-detects the model type from the source checkpoint's config.json
(by looking for a ``classify_num`` field inside ``thinker_config``).  You can
also force the type with ``--model_type asr`` or ``--model_type forced_aligner``.

Reproducible Usage
==================

1) Convert a Qwen3 ASR model:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --model_id Qwen/Qwen3-ASR-0.6B \
  --dst_dir qwen3-asr-hf \
  --push_to_hub <username-or-org>/Qwen3-ASR-0.6B
```

2) Convert a Qwen3 Forced Aligner model:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --model_id Qwen/Qwen3-ForcedAligner-0.6B \
  --dst_dir qwen3-forced-aligner-hf \
  --push_to_hub <username-or-org>/Qwen3-ForcedAligner-0.6B
```

3) Convert from a local directory with explicit model type:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --src_dir /path/to/local/model \
  --dst_dir output-hf \
  --model_type forced_aligner
```
"""

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen3ASRConfig,
    Qwen3ASRFeatureExtractor,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRForForcedAlignment,
    Qwen3ASRProcessor,
    Qwen3ForcedAlignerConfig,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# fmt: off
STATE_DICT_MAPPING_ASR = {
    r"^thinker\.audio_tower\.": r"model.audio_tower.",
    r"^thinker\.lm_head\.": r"lm_head.",
    r"^thinker\.model\.": r"model.language_model.",
}

STATE_DICT_MAPPING_FORCED_ALIGNER = {
    r"^thinker\.audio_tower\.": r"model.audio_tower.",
    r"^thinker\.lm_head\.": r"classifier.",
    r"^thinker\.model\.": r"model.language_model.",
}
# fmt: on


def map_old_key_to_new(old_key: str, mapping: dict[str, str]) -> str:
    """Map checkpoint keys to transformers model keys."""
    new_key = old_key
    for pattern, replacement in mapping.items():
        new_key, n = re.subn(pattern, replacement, new_key)
        if n > 0:
            break
    return new_key


def convert_state_dict(original_state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Convert checkpoint state dict to transformers format."""
    new_state_dict = {}
    # `Qwen3ASRAudioAttention` inherits from `WhisperAttention`, which hardcodes `bias=False` on
    # `k_proj` — drop the k_proj bias entries from the source checkpoint (they're mathematically
    # redundant for softmax attention: a per-query constant that cancels out during softmax).
    k_proj_bias_re = re.compile(r"audio_tower\.layers\.\d+\.self_attn\.k_proj\.bias$")
    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key, mapping)
        if k_proj_bias_re.search(new_key):
            logger.debug(f"Dropping redundant k_proj bias: {old_key}")
            continue
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")
    return new_state_dict


def detect_model_type(src_root: Path) -> str:
    """Auto-detect model type from the source checkpoint's config.json."""
    config_path = src_root / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    thinker = config.get("thinker_config", {})
    if "classify_num" in thinker:
        logger.info("Auto-detected model type: forced_aligner (found classify_num in thinker_config)")
        return "forced_aligner"

    logger.info("Auto-detected model type: asr (no classify_num in thinker_config)")
    return "asr"


def clean_config(src_root: Path, model_type: str) -> dict:
    """Load and clean up the source config for transformers compatibility."""
    config_path = src_root / "config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)

    config_dict = model_config.copy()

    # fmt: off
    # Remove unused top-level keys
    for key in ["support_languages"]:
        config_dict.pop(key, None)

    # Flatten thinker_config structure
    if "thinker_config" in config_dict:
        thinker_config = config_dict.pop("thinker_config")
        if "audio_config" in thinker_config:
            config_dict["audio_config"] = thinker_config["audio_config"]
        if "text_config" in thinker_config:
            config_dict["text_config"] = thinker_config["text_config"]
        if "audio_token_id" in thinker_config:
            config_dict["audio_token_id"] = thinker_config["audio_token_id"]
        if "initializer_range" in thinker_config:
            config_dict["initializer_range"] = thinker_config["initializer_range"]
        # Forced aligner specific
        if model_type == "forced_aligner" and "classify_num" in thinker_config:
            config_dict["num_timestamp_bins"] = thinker_config["classify_num"]

    # Audio config: strip non-standard fields
    if "audio_config" in config_dict:
        audio_unused = [
            "_name_or_path", "architectures", "dtype", "model_type", "use_bfloat16", "add_cross_attention",
            "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
            "finetuning_task", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
            "output_attentions", "output_hidden_states", "pad_token_id", "bos_token_id", "eos_token_id",
            "prefix", "problem_type", "pruned_heads", "return_dict", "sep_token_id", "task_specific_params",
            "tf_legacy_loss", "tie_encoder_decoder", "tie_word_embeddings", "tokenizer_class", "torchscript",
        ]
        for key in audio_unused:
            config_dict["audio_config"].pop(key, None)

    # Text config: strip non-standard fields + MoE fields + M-RoPE fields
    if "text_config" in config_dict:
        text_unused = [
            "_name_or_path", "architectures", "dtype", "model_type", "use_bfloat16", "add_cross_attention",
            "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
            "finetuning_task", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
            "output_attentions", "output_hidden_states", "prefix", "problem_type", "pruned_heads",
            "return_dict", "sep_token_id", "task_specific_params", "tf_legacy_loss", "tie_encoder_decoder",
            "tokenizer_class", "torchscript",
            # MoE-specific fields
            "decoder_sparse_step", "moe_intermediate_size", "num_experts_per_tok", "num_experts",
            "norm_topk_prob", "output_router_logits", "router_aux_loss_coef", "mlp_only_layers",
        ]
        for key in text_unused:
            config_dict["text_config"].pop(key, None)

        # Strip M-RoPE fields from rope_scaling
        rope_cfg = config_dict["text_config"].get("rope_scaling")
        if isinstance(rope_cfg, dict):
            for mrope_key in ["mrope_interleaved", "interleaved", "mrope_section", "type"]:
                rope_cfg.pop(mrope_key, None)
    # fmt: on

    return config_dict


# fmt: off
FORCED_ALIGNER_CHAT_TEMPLATE = (
    "{%- set ns = namespace(audio_tokens='', words=[]) -%}"
    "{%- for m in messages -%}"
    "{%- if m.content is not string -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}"
    "{%- set ns.audio_tokens = ns.audio_tokens + '<|audio_start|><|audio_pad|><|audio_end|>' -%}"
    "{%- endif -%}"
    "{%- if c.type == 'text' and (c.text is defined) -%}"
    "{%- set ns.words = ns.words + [c.text] -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{{- ns.audio_tokens + ns.words | join('<timestamp><timestamp>') + '<timestamp><timestamp>' -}}"
)
# fmt: on


def write_processor(src_root: Path, dst_root: Path, model_type: str):
    """Write processor (shared by both ASR and Forced Aligner)."""
    tokenizer = AutoTokenizer.from_pretrained(src_root)

    if model_type == "forced_aligner":
        chat_template = FORCED_ALIGNER_CHAT_TEMPLATE
    else:
        # Load chat template from separate file if it exists
        chat_template_file = src_root / "chat_template.json"
        chat_template = None
        if chat_template_file.exists():
            logger.info("Loading chat template from %s", chat_template_file)
            with open(chat_template_file, "r", encoding="utf-8") as f:
                chat_template_data = json.load(f)
                chat_template = chat_template_data.get("chat_template")

    processor = Qwen3ASRProcessor(
        feature_extractor=Qwen3ASRFeatureExtractor(),
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.save_pretrained(str(dst_root))
    logger.info("Processor saved to %s", dst_root)
    return processor


def load_state_dict(src_root: Path) -> dict[str, torch.Tensor]:
    """Load safetensors state dict from source directory."""
    state = {}
    shard_files = sorted(src_root.glob("model-*.safetensors"))
    single_file = src_root / "model.safetensors"

    if shard_files:
        logger.info("Found %d sharded safetensor files", len(shard_files))
        safetensor_paths = shard_files
    elif single_file.exists():
        safetensor_paths = [single_file]
    else:
        raise FileNotFoundError(f"No safetensor files found in {src_root}")

    for path in safetensor_paths:
        logger.info("Loading %s", path.name)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)

    return state


def write_asr_model(src_root: Path, dst_root: Path):
    """Convert and write a Qwen3 ASR model."""
    config_dict = clean_config(src_root, "asr")
    config = Qwen3ASRConfig(**config_dict)
    model = Qwen3ASRForConditionalGeneration(config).to(torch.bfloat16)

    state = load_state_dict(src_root)
    state = convert_state_dict(state, STATE_DICT_MAPPING_ASR)

    load_res = model.load_state_dict(state, strict=True)
    if load_res.missing_keys:
        raise ValueError(f"Missing keys: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        raise ValueError(f"Unexpected keys: {load_res.unexpected_keys}")

    model.to(torch.bfloat16)
    model.generation_config = GenerationConfig(
        eos_token_id=(151643, 151645),
        pad_token_id=151645,
        do_sample=False,
    )
    model.save_pretrained(str(dst_root))
    logger.info("ASR model saved to %s", dst_root)
    return model


def write_forced_aligner_model(src_root: Path, dst_root: Path):
    """Convert and write a Qwen3 Forced Aligner model."""
    config_dict = clean_config(src_root, "forced_aligner")
    config = Qwen3ForcedAlignerConfig(**config_dict)
    model = Qwen3ASRForForcedAlignment(config).to(torch.bfloat16)

    state = load_state_dict(src_root)
    state = convert_state_dict(state, STATE_DICT_MAPPING_FORCED_ALIGNER)

    load_res = model.load_state_dict(state, strict=True)
    if load_res.missing_keys:
        raise ValueError(f"Missing keys: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        raise ValueError(f"Unexpected keys: {load_res.unexpected_keys}")

    model.to(torch.bfloat16)
    model.save_pretrained(str(dst_root))
    logger.info("Forced Aligner model saved to %s", dst_root)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Qwen3 ASR or Qwen3 Forced Aligner checkpoints to Hugging Face format."
    )
    ap.add_argument("--model_id", default=None, type=str, help="Hugging Face model ID")
    ap.add_argument("--src_dir", default=None, help="Source model root directory (alternative to --model_id)")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for converted model")
    ap.add_argument(
        "--model_type",
        default=None,
        choices=["asr", "forced_aligner"],
        help="Model type to convert. If not specified, auto-detected from the source config.",
    )
    ap.add_argument("--push_to_hub", default=None, type=str, help="Push to Hub repo ID")
    args = ap.parse_args()

    # Determine source directory
    if args.model_id:
        logger.info("Downloading model from Hugging Face Hub: %s", args.model_id)
        src_root = Path(tempfile.mkdtemp())
        src_root = Path(snapshot_download(args.model_id, cache_dir=str(src_root)))
        logger.info("Model downloaded to: %s", src_root)
    elif args.src_dir:
        src_root = Path(args.src_dir).resolve()
    else:
        raise ValueError("Either --model_id or --src_dir must be provided")

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    # Auto-detect or use provided model type
    model_type = args.model_type or detect_model_type(src_root)
    logger.info("Converting model type: %s", model_type)

    dst_root = Path(args.dst_dir).resolve()
    if dst_root.exists():
        logger.info("Removing existing destination directory: %s", dst_root)
        shutil.rmtree(dst_root)

    # Write processor (shared class, model-type-specific chat template)
    processor = write_processor(src_root, dst_root, model_type)

    # Write model
    if model_type == "asr":
        model = write_asr_model(src_root, dst_root)
    else:
        model = write_forced_aligner_model(src_root, dst_root)

    # Optionally push to Hub
    if args.push_to_hub:
        logger.info("Pushing processor to the Hub ...")
        processor.push_to_hub(args.push_to_hub)
        logger.info("Pushing model to the Hub ...")
        model.push_to_hub(args.push_to_hub)

        # Verify upload
        logger.info("Verifying upload by loading from Hub: %s", args.push_to_hub)
        _ = Qwen3ASRProcessor.from_pretrained(args.push_to_hub)
        if model_type == "asr":
            _ = Qwen3ASRForConditionalGeneration.from_pretrained(args.push_to_hub)
        else:
            _ = Qwen3ASRForForcedAlignment.from_pretrained(args.push_to_hub)
        logger.info("Verification successful!")


if __name__ == "__main__":
    main()
