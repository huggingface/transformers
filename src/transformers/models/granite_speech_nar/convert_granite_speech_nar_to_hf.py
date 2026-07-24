# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Convert an original Granite Speech NAR checkpoint (e.g. `ibm-granite/granite-speech-4.1-2b-nar`)
to the native HuggingFace `GraniteSpeechNarForCTC` format.

The upstream checkpoint ships as a remote-code model (its `config.json` carries an `auto_map`
pointing at `GraniteSpeechNarForASR`) and stores its weights under the pre-refactor namespace:
  * the language model lives under `language_model.model.*` / `language_model.lm_head.*`,
  * the conformer encoder, projector and BPE CTC head are flat top-level `encoder.*` /
    `projector.*` (with `encoder.out_bpe.*` for the BPE head), and
  * the projector's Q-Former uses a fused `cross_attention.{q,k,v,o}_proj` attention.

The native model nests everything under `base_model_prefix = "model"` and reuses Blip2's
multi-head attention in the Q-Former, so the cross-attention projections become
`cross_attention.{query,key,value}` with the output projection lifted to the layer (`o_proj`).
These are exactly the renamings that used to live in `conversion_mapping.py` under the
`granite_speech_nar` / `GraniteSpeechNarModel` entries; applying them here at conversion time lets
us drop that runtime mapping entirely (mirroring `convert_nemotron_asr_streaming_to_hf.py`).
"""

import argparse
import gc
import re

import torch
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    GraniteSpeechNarConfig,
    GraniteSpeechNarFeatureExtractor,
    GraniteSpeechNarForCTC,
    GraniteSpeechNarProcessor,
)
from transformers.utils.hub import cached_file, get_checkpoint_shard_files


# Ordered regex renamings from the original checkpoint namespace to the native HF layout. Order
# matters: `encoder.out_bpe` must precede the generic `encoder` prefix so the BPE head is lifted to
# the model level instead of being buried under `model.encoder`, and the Q-Former cross-attention
# renames run last, after `projector` has already been prefixed with `model.`.
ORIGINAL_TO_HF_WEIGHT_MAPPING = {
    r"^language_model\.model": "model.language_model",
    r"^language_model\.lm_head": "lm_head",
    r"^encoder\.out_bpe": "model.out_bpe",
    r"^encoder": "model.encoder",
    r"^projector": "model.projector",
    # `window_positions` lives on the Q-Former (it is applied inside its forward), so it moves under
    # the `qformer` submodule rather than sitting directly on the projector.
    r"projector\.window_positions": "projector.qformer.window_positions",
    # The projector reuses the parent `GraniteSpeechEncoderProjector.linear` as its output head.
    r"projector\.out_linear": "projector.linear",
    # The projector's Q-Former reuses Blip2's multi-head attention (`query`/`key`/`value`) instead
    # of the fused `q_proj`/`k_proj`/`v_proj`, and applies the output projection at the layer level
    # (`o_proj`) rather than inside `cross_attention`.
    r"qformer\.layers\.(\d+)\.cross_attention\.q_proj": r"qformer.layers.\1.cross_attention.query",
    r"qformer\.layers\.(\d+)\.cross_attention\.k_proj": r"qformer.layers.\1.cross_attention.key",
    r"qformer\.layers\.(\d+)\.cross_attention\.v_proj": r"qformer.layers.\1.cross_attention.value",
    r"qformer\.layers\.(\d+)\.cross_attention\.o_proj": r"qformer.layers.\1.o_proj",
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def load_original_state_dict(hf_repo_id, revision=None):
    """Load the original (unconverted) weights from the Hub, handling sharded checkpoints."""
    try:
        index_file = cached_file(hf_repo_id, "model.safetensors.index.json", revision=revision)
        shard_files, _ = get_checkpoint_shard_files(hf_repo_id, index_file, revision=revision)
    except OSError:
        # Single-shard checkpoint: fall back to the lone safetensors file.
        shard_files = [cached_file(hf_repo_id, "model.safetensors", revision=revision)]

    state_dict = {}
    for shard_file in shard_files:
        state_dict.update(load_file(shard_file))
    return state_dict


def convert_state_dict(state_dict):
    """Rename every key from the original namespace onto the native `GraniteSpeechNarForCTC` layout."""
    converted = {}
    for key, value in state_dict.items():
        # Every projector/Q-Former LayerNorm is affine-free in the native model (its checkpoint affine
        # params are exactly identity), so drop those weights/biases entirely.
        if key.startswith("projector.") and re.search(r"\.(layer_norms\.\d+|out_norm|attn_norm|mlp_norm)\.", key):
            continue
        converted[convert_key(key, ORIGINAL_TO_HF_WEIGHT_MAPPING)] = value
    return converted


def write_processor(hf_repo_id, output_dir, revision=None):
    """Rebuild the processor from the native feature extractor + tokenizer and save it."""
    feature_extractor_dict, _ = GraniteSpeechNarFeatureExtractor.get_feature_extractor_dict(
        hf_repo_id, revision=revision
    )
    # The native feature extractor follows the `SequenceFeatureExtractor` convention and names the
    # number of mel bins `feature_size` (the original checkpoint calls it `n_mels`).
    if "n_mels" in feature_extractor_dict:
        feature_extractor_dict["feature_size"] = feature_extractor_dict.pop("n_mels")
    feature_extractor = GraniteSpeechNarFeatureExtractor.from_dict(feature_extractor_dict)
    # Drop the remote-code pointer so the output loads as a native feature extractor.
    feature_extractor.auto_map = {}
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id, revision=revision)
    processor = GraniteSpeechNarProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)


# Keys present in the original checkpoint's config that the slim native config no longer declares
# (unused fields, or renamed like `num_heads` -> `num_attention_heads`). `@strict` configs absorb
# unknown keys and re-emit them from `to_dict`, so drop them to keep the saved config clean.
STALE_CONFIG_KEYS = {
    "": ["min_edit_sequence_length", "scale_projected_embeddings", "audio_token_id", "encoder_layer_indices"],
    "encoder_config": ["pred_dropout", "blank_token_id"],
    "projector_config": ["llm_dim", "mlp_bias", "mlp_ratio", "num_heads"],
}


def _strip_stale_config_keys(config):
    for attr in STALE_CONFIG_KEYS[""]:
        config.__dict__.pop(attr, None)
    for sub, attrs in STALE_CONFIG_KEYS.items():
        if sub:
            for attr in attrs:
                getattr(config, sub).__dict__.pop(attr, None)


def write_model(hf_repo_id, output_dir, revision=None):
    """Convert the weights, load them into a native model, and save the model."""
    config_dict, _ = GraniteSpeechNarConfig.get_config_dict(hf_repo_id, revision=revision)
    # The original checkpoint stores `text_config` as a plain `granite` config; retarget it to the
    # dedicated NAR `model_type` so it resolves to the non-causal language-model config.
    if isinstance(config_dict.get("text_config"), dict) and config_dict["text_config"].get("model_type") == "granite":
        config_dict["text_config"]["model_type"] = "granite_speech_nar_text"
    config = GraniteSpeechNarConfig.from_dict(config_dict)
    # Drop remote-code pointers and any stale checkpoint path so the output is a pure native model.
    config.auto_map = {}
    if hasattr(config, "_name_or_path"):
        config._name_or_path = ""
    _strip_stale_config_keys(config)
    print(f"Loaded config: {config}")

    print("Loading and converting the original checkpoint.")
    original_state_dict = load_original_state_dict(hf_repo_id, revision=revision)
    converted_state_dict = convert_state_dict(original_state_dict)

    print("Loading the converted weights into a native GraniteSpeechNarForCTC model.")
    with torch.device("meta"):
        model = GraniteSpeechNarForCTC(config)

    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False, assign=True)
    # The LM head is tied to the input embeddings, so it is legitimately absent from the checkpoint.
    tied_keys = set(model._tied_weights_keys or {})
    missing_keys = [key for key in missing_keys if key not in tied_keys]

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        print("All weights loaded successfully!")

    model.tie_weights()

    print("Saving the model.")
    model.save_pretrained(output_dir)

    del model, original_state_dict, converted_state_dict
    gc.collect()

    print("Reloading the model to check it saved correctly.")
    GraniteSpeechNarForCTC.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def main(hf_repo_id, output_dir, revision=None):
    write_processor(hf_repo_id, output_dir, revision=revision)
    write_model(hf_repo_id, output_dir, revision=revision)


"""
Conversion example:
```bash
python src/transformers/models/granite_speech_nar/convert_granite_speech_nar_to_hf.py \
    --hf_repo_id ibm-granite/granite-speech-4.1-2b-nar \
    --revision a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2 \
    --output_dir /raid/eustache/granite-speech-4.1-2b-nar
```
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="ibm-granite/granite-speech-4.1-2b-nar",
        help="Model repo on huggingface.co holding the original (unconverted) checkpoint.",
    )
    parser.add_argument(
        "--revision",
        default="a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2",
        help="Git revision (commit hash / branch / tag) of the source repo to convert.",
    )
    parser.add_argument(
        "--output_dir",
        default="/raid/eustache/granite-speech-4.1-2b-nar",
        help="Output directory for the converted native HuggingFace model.",
    )
    args = parser.parse_args()
    main(args.hf_repo_id, args.output_dir, revision=args.revision)
