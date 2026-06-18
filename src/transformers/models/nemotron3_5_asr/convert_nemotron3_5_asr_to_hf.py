# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert a NeMo multilingual, prompt-conditioned cache-aware streaming RNN-T checkpoint
(e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b`) to the HuggingFace `Nemotron3_5Asr` format.

Adapted from `convert_nemotron_asr_to_hf.py`. `Nemotron3_5Asr` is the multilingual extension of
`NemotronAsr`: the encoder / decoder / joint module layout (and therefore the weight key mappings) is
identical, plus a single new top-level module — `prompt_kernel`, the language-ID prompt-fusion MLP —
whose NeMo keys (`prompt_kernel.0/.2.*`) match the HF names verbatim and so pass through unchanged.

The multilingual differences handled here are:
  * the target classes are `Nemotron3_5Asr*`,
  * `num_prompts` and the `prompt_dictionary` (locale -> prompt index) are read from NeMo's
    `model_defaults` and written into the config and processor,
  * `prompt_intermediate_size` is derived from the `prompt_kernel` shapes.
"""

import argparse
import gc
import os
import re
import tarfile

import torch
import yaml
from tokenizers import AddedToken

from transformers import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrEncoderConfig,
    Nemotron3_5AsrFeatureExtractor,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
    ParakeetTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter
from transformers.utils.hub import cached_file


# Encoder / decoder / joint submodule layout matches NemotronAsr (and Parakeet), so these mappings are
# reused verbatim. The new `prompt_kernel.*` keys are not matched by any pattern and pass through as-is.
NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    # NeMo stores the conformer conv norm under `conv.batch_norm` regardless of whether it is a
    # BatchNorm or (for cache-aware checkpoints) a LayerNorm; HF names it `conv.norm`.
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

# RNN-T decoder (prediction network) and joint network.
NEMO_RNNT_WEIGHT_MAPPING = {
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict[str, str]:
    """Extract .nemo file (tar archive) and return paths to important files."""
    print(f"Extracting NeMo archive: {nemo_file_path}")

    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
        tar.extractall(extract_dir)

    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    print(f"All extracted files: {[os.path.basename(f) for f in all_files]}")

    model_files = {}
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()

            if (
                file.endswith(".pt")
                or file.endswith(".pth")
                or file.endswith(".ckpt")
                or file.endswith(".bin")
                or "model" in file_lower
                and ("weight" in file_lower or "state" in file_lower)
                or file_lower == "model.pt"
                or file_lower == "pytorch_model.bin"
                or file_lower == "model_weights.ckpt"
            ):
                model_files["model_weights"] = file_path
                print(f"Found model weights: {file}")

            elif (
                file == "model_config.yaml"
                or file == "config.yaml"
                or (file.endswith(".yaml") and "config" in file_lower)
            ):
                if "model_config" not in model_files:
                    model_files["model_config"] = file_path
                    print(f"Found config file: {file}")
                if file == "model_config.yaml":
                    model_files["model_config"] = file_path

            elif (
                file.endswith(".vocab")
                or file.endswith(".model")
                or file.endswith(".txt")
                or ("tokenizer" in file_lower and (file.endswith(".vocab") or file.endswith(".model")))
            ):
                if "tokenizer_model_file" not in model_files or file.endswith(".model"):
                    model_files["tokenizer_model_file"] = file_path
                    print(f"Found tokenizer model file: {file}")
                else:
                    print(f"Found additional vocabulary file (using existing): {file}")

    print(f"Found model files: {list(model_files.keys())}")

    if "model_weights" not in model_files:
        raise FileNotFoundError(
            f"Could not find model weights file in {nemo_file_path}. "
            f"Expected files with extensions: .pt, .pth, .ckpt, .bin. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    if "model_config" not in model_files:
        raise FileNotFoundError(
            f"Could not find model config file in {nemo_file_path}. "
            f"Expected: model_config.yaml or config.yaml. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    return model_files


def _resolve_prompt_conditioning(nemo_config: dict) -> tuple[int, dict]:
    """Return (num_prompts, prompt_dictionary) from NeMo's `model_defaults`."""
    model_defaults = nemo_config.get("model_defaults", {})
    num_prompts = model_defaults.get("num_prompts")
    prompt_dictionary = model_defaults.get("prompt_dictionary")
    if num_prompts is None or prompt_dictionary is None:
        raise ValueError(
            "Could not find `num_prompts` / `prompt_dictionary` in NeMo `model_defaults`. This converter "
            "targets the multilingual, prompt-conditioned Nemotron 3.5 ASR checkpoints; for the English "
            "(non-prompted) model use `convert_nemotron_asr_to_hf.py` instead."
        )
    # NeMo stores keys like `'no'` (Norwegian) that YAML may parse as the boolean False; normalize to str.
    prompt_dictionary = {str(k): int(v) for k, v in prompt_dictionary.items()}
    return int(num_prompts), prompt_dictionary


def write_processor(nemo_config: dict, model_files, output_dir, push_to_repo_id=None, create_pr=True, revision=None):
    tokenizer_converted = ParakeetConverter(model_files["tokenizer_model_file"]).converted()
    tokenizer_converted_fast = ParakeetTokenizer(
        tokenizer_object=tokenizer_converted,
        clean_up_tokenization_spaces=False,
    )

    if tokenizer_converted_fast.convert_tokens_to_ids("<unk>") is None:
        tokenizer_converted_fast.add_tokens([AddedToken("<unk>", normalized=False, special=True)])
        print(f"Added <unk> token at ID: {tokenizer_converted_fast.convert_tokens_to_ids('<unk>')}")
    if tokenizer_converted_fast.convert_tokens_to_ids("<pad>") is None:
        tokenizer_converted_fast.add_tokens([AddedToken("<pad>", normalized=False, special=True)])
        print(f"Added <pad> token at ID: {tokenizer_converted_fast.convert_tokens_to_ids('<pad>')}")
    # Transducer models need a separate blank token at the end of the vocab.
    tokenizer_converted_fast.add_tokens([AddedToken("<blank>", normalized=False, special=True)])
    print(f"Added <blank> token at ID: {tokenizer_converted_fast.convert_tokens_to_ids('<blank>')}")
    tokenizer_converted_fast.add_special_tokens(
        {
            "pad_token": AddedToken("<pad>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
        }
    )
    # The multilingual model emits a language tag (e.g. `<en-US>`) in automatic-detection mode. These tags
    # are regular BPE vocab tokens in the NeMo checkpoint; mark them as special tokens (keeping their ids)
    # so `batch_decode(..., skip_special_tokens=True)` strips them, Whisper-style.
    language_tag_pattern = re.compile(r"^<[a-z]{2,3}-[A-Za-z]{2}>$")
    language_tags = sorted(t for t in tokenizer_converted_fast.get_vocab() if language_tag_pattern.match(t))
    tokenizer_converted_fast.add_special_tokens(
        {"additional_special_tokens": [AddedToken(t, normalized=False, special=True) for t in language_tags]}
    )
    print(f"Marked {len(language_tags)} language-tag tokens as special: {language_tags}")

    # Nemotron3_5AsrFeatureExtractor (like NemotronAsrFeatureExtractor) has no normalization step at all,
    # so NeMo's `preprocessor.normalize` is dropped rather than translated.
    feature_extractor_config_keys_mapping = {
        "sample_rate": "sampling_rate",
        "window_size": "win_length",
        "window_stride": "hop_length",
        "n_fft": "n_fft",
        "features": "feature_size",
        "pad_value": "padding_value",
        "preemphasis": "preemphasis",
    }
    feature_extractor_keys_to_ignore = [
        "_target_",
        "normalize",
        "dither",
        "window",
        "log",
        "pad_to",
        "frame_splicing",
        "nb_augmentation_prob",
    ]
    converted_feature_extractor_config = {}

    for key, value in nemo_config["preprocessor"].items():
        if key in feature_extractor_keys_to_ignore:
            continue
        if key in feature_extractor_config_keys_mapping:
            # NeMo stores window size/stride in seconds; the feature extractor wants samples.
            if key in ["window_size", "window_stride"]:
                value = int(value * nemo_config["preprocessor"]["sample_rate"])
            converted_feature_extractor_config[feature_extractor_config_keys_mapping[key]] = value
        else:
            raise ValueError(f"Key {key} not found in feature_extractor_config_keys_mapping")

    feature_extractor = Nemotron3_5AsrFeatureExtractor(**converted_feature_extractor_config)

    # Carry the model's supported right attention contexts onto the processor so it can validate
    # `streaming_latency_ms` and emit `num_lookahead_tokens`. NeMo stores a single [left, right] pair, a list
    # of pairs (multi-lookahead), or [-1, -1] for full (offline) context.
    att_context_size = nemo_config["encoder"].get("att_context_size")
    supported_num_lookahead_tokens = default_num_lookahead_tokens = None
    if att_context_size is not None and att_context_size not in ([-1, -1], [[-1, -1]]):
        pairs = att_context_size if isinstance(att_context_size[0], (list, tuple)) else [att_context_size]
        supported_num_lookahead_tokens = [right for _, right in pairs]
        default_num_lookahead_tokens = pairs[0][1]

    num_prompts, prompt_dictionary = _resolve_prompt_conditioning(nemo_config)

    processor = Nemotron3_5AsrProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_converted_fast,
        supported_num_lookahead_tokens=supported_num_lookahead_tokens,
        default_num_lookahead_tokens=default_num_lookahead_tokens,
        prompt_dictionary=prompt_dictionary,
        num_prompts=num_prompts,
    )
    processor.save_pretrained(output_dir)

    if push_to_repo_id:
        commit_info = processor.push_to_hub(push_to_repo_id, create_pr=create_pr, revision=revision)
        if create_pr and hasattr(commit_info, "pr_url") and commit_info.pr_url:
            pr_num = commit_info.pr_url.rstrip("/").split("/")[-1]
            return f"refs/pr/{pr_num}"

    return revision


def convert_encoder_config(nemo_config):
    """Convert NeMo encoder config to a `Nemotron3_5AsrEncoderConfig`."""
    encoder_keys_to_ignore = [
        "stochastic_depth_start_layer",
        "feat_out",
        "stochastic_depth_drop_prob",
        "_target_",
        "untie_biases",
        "self_attention_model",
        "subsampling",
        "stochastic_depth_mode",
        "dropout_pre_encoder",
        "reduction",
        "reduction_factor",
        "reduction_position",
        # Multi-lookahead training-only sampling probs; inference uses the first context only.
        "att_context_probs",
        # These are inherent to the cache-aware architecture and not represented in the HF config: the
        # conv module is always layer-norm + fully causal, and the subsampling is always causal.
        "att_context_style",
        "conv_norm_type",
        "conv_context_size",
        "causal_downsampling",
    ]
    # ff_expansion_factor combines with d_model to give intermediate_size in HF.
    ff_expansion = nemo_config["encoder"].get("ff_expansion_factor")
    d_model = nemo_config["encoder"].get("d_model")
    if ff_expansion is not None and d_model is not None:
        nemo_config = {**nemo_config, "encoder": {**nemo_config["encoder"]}}
        nemo_config["encoder"].pop("ff_expansion_factor")
        nemo_config["encoder"]["__intermediate_size__"] = d_model * ff_expansion
    encoder_config_keys_mapping = {
        "d_model": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
        "feat_in": "num_mel_bins",
        "conv_kernel_size": "conv_kernel_size",
        "subsampling_factor": "subsampling_factor",
        "subsampling_conv_channels": "subsampling_conv_channels",
        "pos_emb_max_len": "max_position_embeddings",
        "dropout": "dropout",
        "dropout_emb": "dropout_positions",
        "dropout_att": "attention_dropout",
        "xscaling": "scale_input",
        "use_bias": "attention_bias",
        # Derived from ff_expansion_factor * d_model in NeMo; consumed as hidden_size * factor here.
        "__intermediate_size__": "intermediate_size",
        # Cache-aware (streaming-trained) field.
        "att_context_size": "att_context_size",
    }
    converted_encoder_config = {}

    for key, value in nemo_config["encoder"].items():
        if key in encoder_keys_to_ignore:
            continue
        if key in encoder_config_keys_mapping:
            if key == "att_context_size":
                # NeMo stores a single [left, right] pair, a list of pairs (multi-lookahead), or [-1, -1]
                # for full (offline) context. The shared left context becomes `sliding_window` (= left + 1)
                # and the per-lookahead rights become `supported_num_lookahead_tokens`.
                if value in ([-1, -1], [[-1, -1]]):
                    continue
                pairs = value if isinstance(value[0], (list, tuple)) else [value]
                converted_encoder_config["sliding_window"] = pairs[0][0] + 1
                converted_encoder_config["supported_num_lookahead_tokens"] = [right for _, right in pairs]
                converted_encoder_config["default_num_lookahead_tokens"] = pairs[0][1]
                continue
            converted_encoder_config[encoder_config_keys_mapping[key]] = value
            if key == "use_bias":
                converted_encoder_config["convolution_bias"] = value
        else:
            raise ValueError(f"Key {key} not found in encoder_config_keys_mapping")

    return Nemotron3_5AsrEncoderConfig(**converted_encoder_config)


def _resolve_transducer_labels(nemo_config: dict) -> tuple[list, int]:
    """
    Return (labels, pad_token_id) for the RNN-T model.

    Cache-aware NeMo RNN-T configs keep the vocab under `joint.vocabulary` instead of the top-level
    `labels` field. Falls back to that when needed.
    """
    labels = nemo_config.get("labels")
    if not labels:
        labels = nemo_config.get("joint", {}).get("vocabulary") or []
    if not labels:
        raise ValueError(
            "Could not find vocabulary in NeMo config. Looked under top-level `labels` and `joint.vocabulary`."
        )
    pad_token_id = labels.index("<pad>") if "<pad>" in labels else 0
    return list(labels), pad_token_id


def convert_rnnt_config(nemo_config, encoder_config, prompt_intermediate_size):
    """Convert NeMo RNN-T config to a `Nemotron3_5AsrConfig`."""
    decoder_config = nemo_config["decoder"]
    joint_config = nemo_config["joint"]
    labels, pad_token_id = _resolve_transducer_labels(nemo_config)
    blank_token_id = len(labels)
    vocab_size = len(labels) + 1  # +1 for blank token, matches NeMo's joint output dim

    prednet = decoder_config.get("prednet", {})
    decoder_hidden_size = prednet.get("pred_hidden", 640)
    num_decoder_layers = prednet.get("pred_rnn_layers", 2)
    jointnet = joint_config.get("jointnet", {})
    joint_hidden_size = jointnet.get("joint_hidden", decoder_hidden_size)
    activation = jointnet.get("activation", "relu")
    max_symbols_per_step = nemo_config.get("decoding", {}).get("greedy", {}).get("max_symbols", 10)

    num_prompts, _ = _resolve_prompt_conditioning(nemo_config)

    print(
        f"RNN-T config: vocab_size={vocab_size} (including blank token), "
        f"decoder_hidden={decoder_hidden_size}, joint_hidden={joint_hidden_size}, "
        f"decoder_layers={num_decoder_layers}, max_symbols_per_step={max_symbols_per_step}, "
        f"num_prompts={num_prompts}, prompt_intermediate_size={prompt_intermediate_size}"
    )

    return Nemotron3_5AsrConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=decoder_hidden_size,
        joint_hidden_size=joint_hidden_size,
        num_decoder_layers=num_decoder_layers,
        hidden_act=activation,
        max_symbols_per_step=max_symbols_per_step,
        encoder_config=encoder_config.to_dict(),
        pad_token_id=pad_token_id,
        blank_token_id=blank_token_id,
        num_prompts=num_prompts,
        prompt_intermediate_size=prompt_intermediate_size,
    )


def load_and_convert_rnnt_state_dict(model_files):
    """Load NeMo RNN-T state dict and convert keys to HF format."""
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted_state_dict = {}
    all_mappings = {**NEMO_TO_HF_WEIGHT_MAPPING, **NEMO_RNNT_WEIGHT_MAPPING}

    for key, value in state_dict.items():
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            print(f"Skipping preprocessing weight: {key}")
            continue
        # Skip the auxiliary CTC head weights present in some hybrid checkpoints.
        if key.startswith("ctc_decoder.") or "ctc_loss" in key:
            print(f"Skipping auxiliary CTC weight: {key}")
            continue
        # `prompt_kernel.*` (the language-ID fusion MLP) has identical names in NeMo and HF.
        converted_key = convert_key(key, all_mappings)
        converted_state_dict[converted_key] = value

    return converted_state_dict


def write_rnnt_model(nemo_config, encoder_config, model_files, output_dir, push_to_repo_id=None, revision=None):
    """Write the RNN-T model using the encoder config, RNN-T config, and converted state dict."""
    converted_state_dict = load_and_convert_rnnt_state_dict(model_files)

    # The prompt-fusion MLP hidden size is the output dim of its first linear (`prompt_kernel.0`).
    if "prompt_kernel.0.weight" not in converted_state_dict:
        raise ValueError(
            "Checkpoint has no `prompt_kernel.0.weight`; this does not look like a prompt-conditioned "
            "Nemotron 3.5 ASR checkpoint. Use `convert_nemotron_asr_to_hf.py` for the English model."
        )
    prompt_intermediate_size = converted_state_dict["prompt_kernel.0.weight"].shape[0]

    model_config = convert_rnnt_config(nemo_config, encoder_config, prompt_intermediate_size)
    print(f"Converted RNN-T config: {model_config}")

    print("Loading the checkpoint in a Nemotron3_5Asr RNN-T model.")
    with torch.device("meta"):
        model = Nemotron3_5AsrForRNNT(model_config)

    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False, assign=True)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        print("All weights loaded successfully!")

    del model.config._name_or_path

    model.generation_config.decoder_start_token_id = model.config.blank_token_id

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id, revision=revision)

    del model

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    Nemotron3_5AsrForRNNT.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def main(
    hf_repo_id,
    output_dir,
    push_to_repo_id=None,
    create_pr=True,
    revision=None,
    nemo_file=None,
):
    if nemo_file is not None:
        filepath = nemo_file
        extract_dir = os.path.join(output_dir, "_nemo_extract")
        os.makedirs(extract_dir, exist_ok=True)
    else:
        nemo_filename = f"{hf_repo_id.split('/')[-1]}.nemo"
        filepath = cached_file(hf_repo_id, nemo_filename)
        extract_dir = os.path.dirname(filepath)

    model_files = extract_nemo_archive(filepath, extract_dir)
    nemo_config = yaml.load(open(model_files["model_config"], "r"), Loader=yaml.FullLoader)

    # When revision is given (e.g. "refs/pr/3"), both pushes target that existing PR branch.
    # Otherwise, write_processor creates a new PR and returns its revision for the model push.
    pr_revision = write_processor(
        nemo_config,
        model_files,
        output_dir,
        push_to_repo_id,
        create_pr=create_pr if revision is None else False,
        revision=revision,
    )
    encoder_config = convert_encoder_config(nemo_config)
    print(f"Converted encoder config: {encoder_config}")
    write_rnnt_model(nemo_config, encoder_config, model_files, output_dir, push_to_repo_id, pr_revision)


"""
Conversion example (Hub):
```bash
python src/transformers/models/nemotron3_5_asr/convert_nemotron3_5_asr_to_hf.py \
    --hf_repo_id nvidia/nemotron-3.5-asr-streaming-0.6b \
    --output_dir OUTPUT_DIR
```

Conversion example (local .nemo file):
```bash
python src/transformers/models/nemotron3_5_asr/convert_nemotron3_5_asr_to_hf.py \
    --nemo_file /path/to/nemotron-3.5-asr-streaming-0.6b.nemo \
    --hf_repo_id nvidia/nemotron-3.5-asr-streaming-0.6b \
    --output_dir OUTPUT_DIR
```
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
        help="Model repo on huggingface.co (or any identifier when --nemo_file is set)",
    )
    parser.add_argument(
        "--nemo_file",
        default=None,
        help="Path to a local .nemo archive. When set, --hf_repo_id is only used for naming/output.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for the HuggingFace model")
    parser.add_argument("--push_to_repo_id", help="Repository ID to push the model to on the Hub")
    parser.add_argument(
        "--create_pr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Create a PR when pushing to the Hub (default: True). Use --no-create_pr to push directly.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help='Push to an existing Hub PR branch (e.g. "refs/pr/3"). Overrides --create_pr.',
    )
    args = parser.parse_args()
    main(
        args.hf_repo_id,
        args.output_dir,
        args.push_to_repo_id,
        args.create_pr,
        args.revision,
        nemo_file=args.nemo_file,
    )
