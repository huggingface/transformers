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
"""Adds the transformers format to the Gradium Phonon TTS runtime package `kmhf/phonon-40m-en` (config.json,
model.safetensors, tokenizer.json, default-voice.safetensors), next to the runtime's files, and converts its audio
tokenizer (the codec, decoder only: the package ships no encoder) to a checkpoint of its own."""

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file
from tokenizers import Regex, Tokenizer, normalizers

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    GradiumPhononTTSConfig,
    GradiumPhononTTSFlowHeadConfig,
    GradiumPhononTTSForConditionalGeneration,
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerModel,
    PocketTTSFeatureExtractor,
    PocketTTSProcessor,
)


# The chat template lays out the prompt: the voice placeholder (when the user turn has an `audio` item: an empty one for
# precomputed latents), the text and, as the generation prompt, the first audio position (`<|audio_bos|>`, the BOS
# latent). The normalizer of the tokenizer prepares the text the way Gradium's runtime does (`build_normalizer`).
# With an `assistant` turn holding the target audio (training), the text may be empty, and the target is the first audio
# position and one placeholder, which the processor expands to the target's frames.
CHAT_TEMPLATE = """
{%- set training = messages | length == 2 -%}
{%- if messages | length > 2 or messages[0]['role'] != 'user' or (training and messages[1]['role'] != 'assistant') -%}
    {{- raise_exception('Gradium Phonon TTS takes one `user` message per conversation, followed by the `assistant` target audio to train on.') -}}
{%- endif -%}
{%- if training -%}
    {%- set target = messages[1]['content'] -%}
    {%- if target is string or target | length != 1 or target[0]['type'] != 'audio' -%}
        {{- raise_exception('The `assistant` turn holds one `audio` item, the target.') -}}
    {%- endif -%}
{%- endif -%}
{%- set content = messages[0]['content'] -%}
{%- if content is string -%}
    {%- set texts = [content] -%}
    {%- set num_audio = 0 -%}
{%- else -%}
    {%- set texts = content | selectattr('type', 'equalto', 'text') | map(attribute='text') | list -%}
    {%- set num_audio = content | selectattr('type', 'equalto', 'audio') | list | length -%}
{%- endif -%}
{%- if texts | length > 1 or (texts | length == 0 and not training) or num_audio > 1 -%}
    {{- raise_exception('The `user` turn holds one `text` item and at most one `audio` item (the voice).') -}}
{%- endif -%}
{%- set voice = '<|audio_prompt|>' if num_audio else '' -%}
{%- set text = (texts[0] if texts else '') | trim -%}
{%- if training -%}
    {{- voice ~ text ~ '<|audio_bos|><|audio_prompt|>' -}}
{%- else -%}
    {{- voice ~ text -}}
    {%- if add_generation_prompt -%}
        {{- '<|audio_bos|>' -}}
    {%- endif -%}
{%- endif -%}
"""

AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_PROMPT_TOKEN = "<|audio_prompt|>"
AUDIO_LATENT_TOKEN = "<|audio_latent|>"
# Gradium's runtime: frames generated after the end-of-speech frame (short texts: at most 4 words), and no fade-in
PROCESSOR_KWARGS = {
    "short_text_max_words": 4,
    "frames_after_eos_short_text": 3,
    "frames_after_eos": 1,
    "fade_in_duration": 0.0,
}
# The placeholders take the tokenizer's first reserved control tokens (special pieces that text never produces, unused
# by the runtime), renamed: the first audio position, the voice frames and the audio latents.
RESERVED_TOKENS = {
    "\x00<|c000|>": AUDIO_BOS_TOKEN,
    "\x00<|c001|>": AUDIO_PROMPT_TOKEN,
    "\x00<|c002|>": AUDIO_LATENT_TOKEN,
}

# The transformers format sits next to Gradium's runtime package: the runtime's `config.json` gets the transformers
# fields and its `tokenizer.json` the placeholder names, its weights (`model.safetensors`, renamed at load by
# `conversion_mapping.py`) are read as they are, and these files are added. The runtime normalizes the text itself before
# its `tokenizer.json`: transformers reads the same tokenizer with the runtime's text preparation as its normalizer, from
# a versioned tokenizer file (`fast_tokenizer_files`).
TRANSFORMERS_TOKENIZER_FILE = "tokenizer.5.0.0.json"
TRANSFORMERS_FILES = ["config.json", "tokenizer.json", TRANSFORMERS_TOKENIZER_FILE, "tokenizer_config.json"]
TRANSFORMERS_FILES += ["chat_template.jinja"]
TRANSFORMERS_FILES += ["processor_config.json", "generation_config.json"]

# Sentinels marking the parts of a number while it is rewritten: double quotes, which the character pass folds to `"`, so
# the text holds none of them by then.
START, END, YEAR_END = "«", "»", "‟"
SCALES = {"“": "billion", "”": "million", "„": "thousand"}


def build_normalizer() -> list:
    """
    Gradium's runtime text preparation in English, as tokenizer normalizers. Its `normalize_text` (`ptts/src/preprocess.rs`):
    typographic quotes and dashes are folded to ASCII, bullets, arrows, emoji and control characters become spaces, `…` a
    period, `@`, `+` and `=` their spoken forms, `;`, `(`, `)` and a `:` not between two characters a comma, punctuation
    does not pile up before a `.` or `,`, and its `numbers` rewrite (`preprocess/rewrite.rs`) spells out the sign, the
    decimal point and the scales of the numbers ("-1,234.5" -> "minus 1 thousand 234 point 5"; values under a thousand and
    years keep their digits, a leading zero marks a code). Then its `prepare_text_prompt`: collapse the whitespace, end
    text whose last character is alphanumeric with a period (`PocketTTSTokenizer` upper-cases the first character).
    """
    billion, million, thousand = SCALES
    # a word that is a number then punctuation: a sign, digits and group commas (at most 40, ending with a digit, below
    # 10**12 without the leading zeros), decimals
    rest, word_end = r"(?:\.[0-9]+)?[!?.:,;…]*", "(?= |$)"
    integer = "(?=[0-9,]{1,40}(?![0-9,]))(?:[0,]*0|[0,]*[1-9](?:,*[0-9]){0,11})"
    steps = [
        normalizers.Replace(Regex(r"(?<!\S):|:(?!\S)|[;()]"), ", "),
        normalizers.Replace(Regex("[»«“”„‟″‶⹂❝❞❠〝〞〟＂]"), '"'),
        normalizers.Replace(Regex("[‘’‚‛′‵❛❜❟＇ʼʻʹ´`]"), "'"),
        normalizers.Replace(Regex("[‐‑‒―]"), "-"),
        normalizers.Replace("…", "."),
        normalizers.Replace("@", " at "),
        normalizers.Replace("+", " plus "),
        normalizers.Replace("=", " equals "),
        normalizers.Replace(
            Regex(r"[–*—\[\]{}•‣◦·→←↑↓➡➜\p{Cc}\x{1F600}-\x{1FAFF}\x{2600}-\x{27BF}\x{1F1E6}-\x{1F1FF}]"), " "
        ),
        normalizers.Replace(Regex(r"[\s!#-&(-/:-@\[-`{-~]+(?=[.,])"), ""),
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
        # numbers: mark the word and the end of its integer part, drop the group commas and the leading zeros
        normalizers.Replace(Regex(rf"(?<![^ ])(?=(?!0[0-9])-?{integer}{rest}{word_end})"), START),
        normalizers.Replace(Regex(rf"(?<={START}-?[0-9,]{{0,40}}[0-9])(?={rest}{word_end})"), END),
        normalizers.Replace(Regex(rf"(?<={START}-?[0-9,]{{0,40}}),(?=[0-9,]*{END})"), ""),
        normalizers.Replace(Regex(f"(?<={START}-?)0+(?=[0-9])"), ""),
        # a year keeps its digits; the scales of the other values of at least a thousand, counted from the end
        normalizers.Replace(Regex(f"(?<={START}-?(?:19|20)[0-9]{{2}}){END}"), YEAR_END),
        normalizers.Replace(Regex(f"(?<={START}-?[0-9]{{1,3}})(?=[0-9]{{9}}{END})"), billion),
        normalizers.Replace(Regex(f"(?<=[{START}{billion}-][0-9]{{1,3}})(?=[0-9]{{6}}{END})"), million),
        normalizers.Replace(Regex(f"(?<=[{START}{billion}{million}-][0-9]{{1,3}})(?=[0-9]{{3}}{END})"), thousand),
        # a scale with a count of zero is not said, nor a last group of zeros; the other groups lose their leading zeros
        normalizers.Replace(Regex(f"(?<=[{billion}{million}{thousand}])000[{million}{thousand}]"), ""),
        normalizers.Replace(Regex(f"(?<=[{billion}{million}{thousand}])000(?={END})"), ""),
        normalizers.Replace(Regex(f"(?<=[{billion}{million}{thousand}])0+(?=[0-9])"), ""),
    ]
    for mark, word in SCALES.items():
        steps += [normalizers.Replace(Regex(f"{mark}(?!{END})"), f" {word} "), normalizers.Replace(mark, f" {word}")]
    steps += [
        normalizers.Replace(Regex(f"(?<={START})-"), "minus "),
        normalizers.Replace(Regex(rf"(?<=[{END}{YEAR_END}])\.(?=[0-9])"), " point "),
        normalizers.Replace(Regex(f"[{START}{END}{YEAR_END}]"), ""),
        # `prepare_text_prompt`
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
        normalizers.Replace(Regex(r"(?<=[\p{Alphabetic}\p{N}])$"), "."),
    ]
    return steps


# Original codec tensor name (regex) -> audio tokenizer name. Applied in order, first match wins. The fused `in_proj`
# weights are split into q/k/v and the q/k rows permuted for rotate-half RoPE (the original applies RoPE to interleaved
# pairs), below.
AUDIO_TOKENIZER_MAPPING = [
    (r"^mimi\.quantizer\.output_proj\.weight$", "latent_proj.weight"),
    (r"^mimi\.upsample\.convtr\.convtr\.weight$", "decoder.upsample.conv.weight"),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.self_attn\.out_proj\.weight$",
        r"decoder.decoder_transformer.layers.\1.self_attn.o_proj.weight",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.linear1\.weight$",
        r"decoder.decoder_transformer.layers.\1.mlp.fc1.weight",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.linear2\.weight$",
        r"decoder.decoder_transformer.layers.\1.mlp.fc2.weight",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.norm1\.(weight|bias)$",
        r"decoder.decoder_transformer.layers.\1.input_layernorm.\2",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.norm2\.(weight|bias)$",
        r"decoder.decoder_transformer.layers.\1.post_attention_layernorm.\2",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.layer_scale_1\.scale$",
        r"decoder.decoder_transformer.layers.\1.self_attn_layer_scale.scale",
    ),
    (
        r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.layer_scale_2\.scale$",
        r"decoder.decoder_transformer.layers.\1.mlp_layer_scale.scale",
    ),
    # SEANet decoder: the original's `model.K` indices are the ones `PocketTTSAudioTokenizerDecoder.layers` uses, and
    # `.conv` / `.convtr` both become the `.conv` of the wrapping (transposed) convolution module.
    (r"^mimi\.decoder\.model\.(\d+)\.(?:conv|convtr)\.(weight|bias)$", r"decoder.decoder_seanet.layers.\1.conv.\2"),
    (
        r"^mimi\.decoder\.model\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
        r"decoder.decoder_seanet.layers.\1.block.\2.conv.\3",
    ),
]
AUDIO_TOKENIZER_FUSED_QKV = (
    r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.self_attn\.in_proj\.weight$",
    r"decoder.decoder_transformer.layers.\1.self_attn",
)


def permute_for_rope(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reorders the rows of a q/k projection from interleaved RoPE pairs to the rotate-half layout."""
    out_dim, in_dim = weight.shape
    return weight.view(num_heads, out_dim // num_heads // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def build_audio_tokenizer_config(original: dict) -> PocketTTSAudioTokenizerConfig:
    """The package's codec, decoder only: the encoder is not released."""
    mimi = original["mimi"]
    samples_per_frame = mimi["sample_rate"] / mimi["frame_rate"]
    fields = {
        "latent_stride": int(samples_per_frame) // math.prod(mimi["ratios"]),
        "audio_channels": mimi["channels"],
        "latent_dim": mimi["quantizer_dimension"],
        "hidden_size": mimi["dimension"],
        "num_filters": mimi["n_filters"],
        "num_residual_layers": mimi["n_residual_layers"],
        "upsampling_ratios": mimi["ratios"],
        "kernel_size": mimi["kernel_size"],
        "last_kernel_size": mimi["last_kernel_size"],
        "residual_kernel_size": mimi["residual_kernel_size"],
        "dilation_growth_rate": mimi["dilation_base"],
        "compress": mimi["compress"],
        "upsample_groups": mimi["dimension"],
        "num_hidden_layers": mimi["transformer_num_layers"],
        "intermediate_size": mimi["transformer_dim_feedforward"],
        "num_attention_heads": mimi["transformer_num_heads"],
        "num_key_value_heads": mimi["transformer_num_heads"],
        # The Rust runtime attends to `transformer_context` past frames plus the current one, with the exact GELU
        # (Pocket TTS's Python original: 250 keys, tanh GELU).
        "sliding_window": mimi["transformer_context"] + 1,
        "hidden_act": "gelu",
        "layer_scale_initial_scale": mimi["transformer_layer_scale"],
        "rope_parameters": {"rope_type": "default", "rope_theta": mimi["transformer_max_period"]},
    }
    return PocketTTSAudioTokenizerConfig(**fields, encoder_config=None)


def build_conditioner_fields(original: dict) -> dict:
    """The runtime's summed conditioners, at their runtime defaults (`num_speakers` 1, `duration_delta` 0)."""
    conditioners = {conditioner["name"]: conditioner for conditioner in original["conditioners"]}
    num_speakers = conditioners["num_speakers"]["lut"]
    duration_delta = conditioners["duration_delta"]["continuous"]
    return {
        "num_speakers": num_speakers["possible_values"].index("1"),
        "num_speakers_vocab_size": num_speakers["n_bins"] + 1,
        "num_speakers_embedding_dim": num_speakers["dim"],
        "duration_delta": 0.0,
        "duration_delta_embedding_dim": duration_delta["dim"],
        "duration_delta_scale": duration_delta["scale_factor"],
        "duration_delta_max_period": duration_delta["max_period"],
    }


def build_config(original: dict, tokenizer_vocab: dict[str, int]) -> GradiumPhononTTSConfig:
    flow_lm = original["flow_lm"]
    flow_head_config = GradiumPhononTTSFlowHeadConfig(
        hidden_size=flow_lm["flow_dim"],
        num_hidden_layers=flow_lm["flow_depth"],
        latent_dim=flow_lm["ldim"],
        condition_dim=flow_lm["d_model"],
    )
    config = GradiumPhononTTSConfig(
        flow_head_config=flow_head_config,
        # n_bins + 1 rows: the tokenizer's ids, then a spare one
        vocab_size=flow_lm["n_bins"] + 1,
        hidden_size=flow_lm["d_model"],
        num_hidden_layers=flow_lm["num_layers"],
        num_attention_heads=flow_lm["num_heads"],
        intermediate_size=flow_lm["dim_feedforward"],
        # the exact GELU (`gelu: erf`)
        hidden_act="gelu",
        latent_dim=flow_lm["ldim"],
        rope_parameters={"rope_type": "default", "rope_theta": flow_lm["max_period"]},
        audio_prompt_dim=original["speaker_mimi"]["mimi"]["dimension"],
        audio_prompt_token_id=tokenizer_vocab[AUDIO_PROMPT_TOKEN],
        audio_latent_token_id=tokenizer_vocab[AUDIO_LATENT_TOKEN],
        audio_bos_token_id=tokenizer_vocab[AUDIO_BOS_TOKEN],
        dtype="float32",
        **build_conditioner_fields(original),
    )
    config.architectures = [GradiumPhononTTSForConditionalGeneration.__name__]
    return config


def merge_configs(original: dict, config: GradiumPhononTTSConfig) -> dict:
    """The runtime's config with the transformers fields added; the runtime ignores unknown keys and so does
    transformers (it keeps them as attributes)."""
    with tempfile.TemporaryDirectory() as config_dir:
        config.save_pretrained(config_dir)
        fields = json.loads((Path(config_dir) / "config.json").read_text())
    if clashes := [key for key in fields if key in original and original[key] != fields[key]]:
        raise ValueError(f"Fields of both configs with different values: {clashes}")
    return {**original, **fields}


def convert_audio_tokenizer_state_dict(
    original: dict[str, torch.Tensor], config: PocketTTSAudioTokenizerConfig
) -> dict[str, torch.Tensor]:
    converted = {}
    for name, tensor in original.items():
        if not name.startswith("mimi."):
            continue
        pattern, replacement = AUDIO_TOKENIZER_FUSED_QKV
        if re.match(pattern, name):
            prefix = re.sub(pattern, replacement, name)
            query, key, value = tensor.chunk(3, dim=0)
            converted[f"{prefix}.q_proj.weight"] = permute_for_rope(query, config.num_attention_heads)
            converted[f"{prefix}.k_proj.weight"] = permute_for_rope(key, config.num_attention_heads)
            converted[f"{prefix}.v_proj.weight"] = value
            continue
        for pattern, replacement in AUDIO_TOKENIZER_MAPPING:
            if re.match(pattern, name):
                converted[re.sub(pattern, replacement, name)] = tensor
                break
        else:
            raise ValueError(f"Unmapped codec tensor: {name}")
    return converted


def convert_tokenizer(model_dir: Path, output_dir: Path) -> dict[str, int]:
    """The runtime's `tokenizer.json` with the placeholders' reserved tokens renamed, and the tokenizer config. Returns
    the vocabulary."""
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    for token in tokenizer["added_tokens"]:
        token["content"] = RESERVED_TOKENS.get(token["content"], token["content"])
    for piece in tokenizer["model"]["vocab"]:
        piece[0] = RESERVED_TOKENS.get(piece[0], piece[0])
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False))
    backend = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    backend.normalizer = normalizers.Sequence([*build_normalizer(), backend.normalizer])
    backend.save(str(output_dir / TRANSFORMERS_TOKENIZER_FILE))

    # The checkpoint's special pieces carry a leading NUL in their content.
    special = {piece: f"\x00{piece}" for piece in ("<unk>", "<s>", "</s>", "<pad>")}
    tokenizer_config = {
        "tokenizer_class": "PocketTTSTokenizer",
        "fast_tokenizer_files": [TRANSFORMERS_TOKENIZER_FILE],
        "unk_token": special["<unk>"],
        "bos_token": special["<s>"],
        "eos_token": special["</s>"],
        "pad_token": special["<pad>"],
        # Call defaults of the processor (TRF019) travel with the tokenizer; the Unigram model has no post-processor,
        # so `add_special_tokens` adds nothing and needs no default.
        "padding": True,
        "padding_side": "left",
        "model_input_names": ["input_ids", "attention_mask"],
        "clean_up_tokenization_spaces": False,
        "model_specific_special_tokens": {
            "audio_prompt_token": AUDIO_PROMPT_TOKEN,
            "audio_latent_token": AUDIO_LATENT_TOKEN,
            "audio_bos_token": AUDIO_BOS_TOKEN,
        },
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    return backend.get_vocab()


def load_voice(model_dir: Path) -> torch.Tensor:
    """The package's default voice, as `audio_prompt_latents` `(num_frames, audio_prompt_dim)`."""
    return load_file(model_dir / "default-voice.safetensors")["speaker_wavs"][0].transpose(0, 1).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Gradium's runtime package (kmhf/phonon-40m-en): config.json, tokenizer.json, model.safetensors",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Gets the runtime package's files and the transformers ones"
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="eustlb/gradium-phonon-tts-audio-tokenizer",
        help="Repo id (or local path) of the audio tokenizer, which the processor points to",
    )
    parser.add_argument(
        "--audio_tokenizer_output_dir", type=Path, default=None, help="Where to save the audio tokenizer"
    )
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repo id of the runtime package to add to")
    parser.add_argument("--create_pr", action="store_true", help="Open a pull request instead of committing")
    parser.add_argument("--push_audio_tokenizer", action="store_true", help="Upload the audio tokenizer too")
    parser.add_argument("--private", action="store_true", help="Create the audio tokenizer repo as private")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_config = json.loads((args.model_dir / "config.json").read_text())
    config = build_config(original_config, convert_tokenizer(args.model_dir, args.output_dir))
    audio_tokenizer_config = build_audio_tokenizer_config(original_config)

    # The weights stay in the runtime's layout (copied for the loading check below, not uploaded).
    weights = load_file(args.model_dir / "model.safetensors")
    shutil.copy(args.model_dir / "model.safetensors", args.output_dir / "model.safetensors")
    merged_config = merge_configs(original_config, config)
    (args.output_dir / "config.json").write_text(json.dumps(merged_config, indent=2) + "\n")

    # The audio tokenizer is a model of its own, held by the processor: decoder only.
    audio_tokenizer = PocketTTSAudioTokenizerModel(audio_tokenizer_config)
    audio_tokenizer_state_dict = convert_audio_tokenizer_state_dict(weights, audio_tokenizer_config)
    missing, unexpected = audio_tokenizer.load_state_dict(audio_tokenizer_state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Audio tokenizer missing keys: {missing}\nUnexpected keys: {unexpected}")
    audio_tokenizer.eval()
    if args.audio_tokenizer_output_dir is not None:
        audio_tokenizer.save_pretrained(args.audio_tokenizer_output_dir)
    # what the processor's `processor_config.json` points to
    audio_tokenizer.name_or_path = args.audio_tokenizer_repo

    processor = PocketTTSProcessor(
        PocketTTSFeatureExtractor(sampling_rate=original_config["mimi"]["sample_rate"]),
        AutoTokenizer.from_pretrained(args.output_dir),
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE.strip() + "\n",
        **PROCESSOR_KWARGS,
    )
    # only the processor's own files: the tokenizer's stay as written above
    with tempfile.TemporaryDirectory() as processor_dir:
        processor.save_pretrained(processor_dir)
        for name in ("processor_config.json", "chat_template.jinja"):
            shutil.copy(Path(processor_dir) / name, args.output_dir / name)
    # The Rust runtime's defaults: sampling with temperature 0.3 (the noise variance of the flow sampler), its number of
    # flow steps and end-of-speech threshold, and no frames before which the end of speech is ignored.
    GenerationConfig(
        do_sample=True,
        temperature=0.3,
        pad_token_id=config.pad_token_id,
        num_flow_steps=original_config["lsd_decode_steps"],
        eos_threshold=original_config["eos_threshold"],
        min_frames_before_eos=0,
        return_dict_in_generate=True,
    ).save_pretrained(args.output_dir)

    # Load from disk (the runtime's weights through the conversion mapping), fail on any loading issue, and print one
    # cheap sanity value.
    model, loading_info = GradiumPhononTTSForConditionalGeneration.from_pretrained(
        args.output_dir, output_loading_info=True
    )
    # training-only state (the loss weighting, the latent-statistics counter) is not in the package
    loading_info["missing_keys"] = {
        key
        for key in loading_info["missing_keys"]
        if not any(re.search(pattern, key) for pattern in model._keys_to_ignore_on_load_missing)
    }
    if any(loading_info.values()):
        raise ValueError(f"Loading issues: {loading_info}")
    processor = PocketTTSProcessor.from_pretrained(args.output_dir)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "This is Gradium's open-source on-device Text-to-Speech model Phonon"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        processor_kwargs={"audio_prompt_latents": [load_voice(args.model_dir)]},
    )
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, output_scores=True)
    print(
        f"Sanity: {output.audio_latents_mask.sum().item()} frames generated, first end-of-speech logit "
        f"{output.scores[0].item():.4f}."
    )

    if args.push_to_hub:
        HfApi().upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.push_to_hub,
            allow_patterns=TRANSFORMERS_FILES,
            commit_message="Add the transformers format (GradiumPhononTTSForConditionalGeneration)",
            create_pr=args.create_pr,
        )
    if args.push_audio_tokenizer:
        if args.audio_tokenizer_output_dir is None:
            raise ValueError("`--push_audio_tokenizer` needs `--audio_tokenizer_output_dir`.")
        api = HfApi()
        api.create_repo(args.audio_tokenizer_repo, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(args.audio_tokenizer_output_dir), repo_id=args.audio_tokenizer_repo)


if __name__ == "__main__":
    main()
