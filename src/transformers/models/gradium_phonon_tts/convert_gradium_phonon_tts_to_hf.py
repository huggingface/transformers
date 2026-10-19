# Copyright 2026 Gradium and The HuggingFace Inc. team. All rights reserved.
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
"""Adds the transformers format to a Gradium Phonon TTS runtime package (config.json, model.safetensors or a GGUF,
tokenizer.json, voices), next to the runtime's files, and converts its audio tokenizer (the codec, decoder only: the
package ships no encoder) to a checkpoint of its own."""

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    GradiumPhononTTSConfig,
    GradiumPhononTTSFlowHeadConfig,
    GradiumPhononTTSForConditionalGeneration,
    GradiumPhononTTSProcessor,
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerModel,
    PocketTTSFeatureExtractor,
)


# The chat template: the original runtime's `normalize_text` (in `lang`, a template variable) and `prepare_text_prompt`,
# then the layout; checked against the Rust runtime and the Python port (`tests/fixtures/gradium_phonon_tts`).
CHAT_TEMPLATE = r"""{#- Gradium Phonon TTS: the original runtime's `normalize_text` (in `lang`, "none" to skip it) then `prepare_text_prompt`,
laid out as the voice placeholder (when the user turn has an `audio` item: an empty one for precomputed latents), the text
and, as the generation prompt, the first audio position (`<|audio_bos|>`, the BOS latent). With an `assistant` turn
holding the target audio (training), the text is kept as written and may be empty, and the target is the first audio
position and one placeholder, which the processor expands to the target's frames. -#}
{%- set lang = lang | default('en') -%}
{%- set spoken_symbols = {
    'en': {'@': 'at', '+': 'plus', '=': 'equals'},
    'fr': {'@': 'arobaze', '+': 'plus', '=': 'égal'},
    'de': {'@': 'ät', '+': 'Plus', '=': 'Gleich'},
    'es': {'@': 'arroba', '+': 'mas', '=': 'igual'},
    'pt': {'@': 'arroba', '+': 'mais', '=': 'igual'},
} -%}
{%- set decimal_separators = {'en': 'point', 'fr': 'virgule', 'de': 'Komma', 'es': 'coma', 'pt': 'vírgula'} -%}
{%- set minus_words = {'en': 'minus', 'fr': 'moins', 'de': 'minus', 'es': 'menos', 'pt': 'menos'} -%}
{#- scale words from a billion down: (divisor, singular, plural) -#}
{%- set scales = {
    'en': [[1000000000, 'billion', 'billion'], [1000000, 'million', 'million'], [1000, 'thousand', 'thousand']],
    'fr': [[1000000000, 'milliard', 'milliards'], [1000000, 'million', 'millions'], [1000, 'mille', 'mille']],
    'de': [[1000000000, 'Milliarde', 'Milliarden'], [1000000, 'Million', 'Millionen'], [1000, 'Tausend', 'Tausend']],
    'es': [[1000000000, 'mil millones', 'mil millones'], [1000000, 'millón', 'millones'], [1000, 'mil', 'mil']],
    'pt': [[1000000000, 'bilhão', 'bilhões'], [1000000, 'milhão', 'milhões'], [1000, 'mil', 'mil']],
} -%}
{%- set double_quotes = '»«“”„‟″‶⹂❝❞❠〝〞〟＂' -%}
{%- set single_quotes = '‘’‚‛′‵❛❜❟＇ʼʻʹ´`' -%}
{%- set dashes = '‐‑‒―' -%}
{%- set dropped_to_space = '–*—[]{}•‣◦·→←↑↓➡➜' -%}
{%- set suffix_chars = '!?.:,;…' -%}

{#- The output buffer keeps clear of punctuation pile-ups: a `.` or `,` swallows the whitespace and ASCII punctuation
(quotes excepted) right before it. -#}
{%- macro push(buffer, char) -%}
    {%- set state = namespace(buffer=buffer, done=false) -%}
    {%- if char in '.,' -%}
        {%- for _ in range(buffer | length) -%}
            {%- if not state.done -%}
                {%- set last = state.buffer[-1:] -%}
                {%- if last and (last.isspace() or (last.isascii() and not last.isalnum() and last not in '"\'')) -%}
                    {%- set state.buffer = state.buffer[:-1] -%}
                {%- else -%}
                    {%- set state.done = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- state.buffer ~ char -}}
{%- endmacro -%}
{%- macro push_whitespace(buffer) -%}
    {{- push(buffer, ' ') if buffer and not buffer[-1].isspace() else buffer -}}
{%- endmacro -%}

{#- Large numbers with their scale words ("1,234.56" -> "1 thousand 234 point 56"); values under a thousand and years stay
digits. Renders nothing when the rule does not claim the word. -#}
{%- macro rewrite_number(word, lang) -%}
    {%- set number = namespace(claimed=true, group='', point=',', rest=word, integer='', decimal=none, suffix='') -%}
    {%- if word | length > 1 and word[0] == '0' and word[1].isdigit() and word[1].isascii() -%}
        {%- set number.claimed = false -%}
    {%- endif -%}
    {%- if lang == 'en' -%}
        {%- set number.group = ',' -%}
        {%- set number.point = '.' -%}
    {%- elif lang != 'fr' -%}
        {#- a dot only groups when every group after the first has three digits -#}
        {%- set integer = word.rsplit(',', 1)[0] if ',' in word else word -%}
        {%- set grouped = namespace(value='.' in integer) -%}
        {%- for part in integer.split('.') -%}
            {%- if not loop.first and part | length != 3 -%}
                {%- set grouped.value = false -%}
            {%- endif -%}
        {%- endfor -%}
        {%- set number.group = '.' if grouped.value else '' -%}
    {%- endif -%}
    {%- set negative = word.startswith('-') -%}
    {%- if negative -%}
        {%- set number.rest = word[1:] -%}
    {%- endif -%}
    {#- the integer part: ASCII digits and group separators -#}
    {%- set scan = namespace(end=number.rest | length) -%}
    {%- for char in number.rest -%}
        {%- if scan.end == number.rest | length and not (char.isdigit() and char.isascii()) and char != number.group -%}
            {%- set scan.end = loop.index0 -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set number.integer = number.rest[:scan.end] -%}
    {%- if number.group -%}
        {%- set number.integer = number.integer.rstrip(number.group) -%}
    {%- endif -%}
    {%- if not number.integer -%}
        {%- set number.claimed = false -%}
    {%- endif -%}
    {%- set number.rest = number.rest[number.integer | length:] -%}
    {%- if number.rest.startswith(number.point) -%}
        {%- set after = number.rest[1:] -%}
        {%- set scan = namespace(end=after | length) -%}
        {%- for char in after -%}
            {%- if scan.end == after | length and not (char.isdigit() and char.isascii()) -%}
                {%- set scan.end = loop.index0 -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if scan.end > 0 -%}
            {%- set number.decimal = after[:scan.end] -%}
            {%- set number.rest = after[scan.end:] -%}
        {%- endif -%}
    {%- endif -%}
    {#- only trailing punctuation may follow the number -#}
    {%- for char in number.rest -%}
        {%- if char not in suffix_chars -%}
            {%- set number.claimed = false -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set digits = namespace(value='') -%}
    {%- for char in number.integer -%}
        {%- if char.isdigit() and char.isascii() -%}
            {%- set digits.value = digits.value ~ char -%}
        {%- endif -%}
    {%- endfor -%}
    {%- if not digits.value or (digits.value | int) >= 1000000000000 -%}
        {%- set number.claimed = false -%}
    {%- endif -%}
    {%- if number.claimed -%}
        {%- set value = digits.value | int -%}
        {%- if value < 1000 or (1900 <= value and value < 2100) -%}
            {%- set words = value | string -%}
        {%- else -%}
            {%- set parts = namespace(items=[]) -%}
            {%- for divisor, one, many in scales[lang] -%}
                {%- set count = (value // divisor) % 1000 -%}
                {%- if count == 1 -%}
                    {%- if lang in ['fr', 'es', 'pt'] and divisor == 1000 -%}
                        {#- "mille" and "mil" are numeral adjectives: no "1" in front of them -#}
                        {%- set parts.items = parts.items + [one] -%}
                    {%- elif lang == 'de' -%}
                        {#- a bare "1" would be read "eins": German inflects the count like an article -#}
                        {%- set parts.items = parts.items + [('ein ' if divisor == 1000 else 'eine ') ~ one] -%}
                    {%- else -%}
                        {%- set parts.items = parts.items + ['1 ' ~ one] -%}
                    {%- endif -%}
                {%- elif count > 1 -%}
                    {%- set parts.items = parts.items + [(count | string) ~ ' ' ~ many] -%}
                {%- endif -%}
            {%- endfor -%}
            {%- if value % 1000 != 0 -%}
                {%- set parts.items = parts.items + [(value % 1000) | string] -%}
            {%- endif -%}
            {%- set words = parts.items | join(' ') -%}
        {%- endif -%}
        {%- if number.decimal is not none -%}
            {%- set words = words ~ ' ' ~ decimal_separators[lang] ~ ' ' ~ number.decimal -%}
        {%- endif -%}
        {%- if negative -%}
            {%- set words = minus_words[lang] ~ ' ' ~ words -%}
        {%- endif -%}
        {{- words ~ number.rest -}}
    {%- endif -%}
{%- endmacro -%}

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
{%- set text = texts[0] if texts else '' -%}

{%- if training -%}
    {{- voice ~ (text | trim) ~ '<|audio_bos|><|audio_prompt|>' -}}
{%- else -%}
{%- if lang != 'none' -%}
    {%- if lang not in spoken_symbols -%}
        {{- raise_exception("Unsupported language '" ~ lang ~ "': expected one of en, fr, de, es, pt, or none.") -}}
    {%- endif -%}
    {%- set spoken = spoken_symbols[lang] -%}
    {%- set result = namespace(buffer='') -%}
    {%- for char in text -%}
        {%- set previous = text[loop.index0 - 1] if not loop.first else none -%}
        {%- set following = text[loop.index0 + 1] if not loop.last else none -%}
        {%- if char in double_quotes -%}
            {%- set result.buffer = push(result.buffer, '"') -%}
        {%- elif char in single_quotes -%}
            {%- set result.buffer = push(result.buffer, "'") -%}
        {%- elif char in dashes -%}
            {%- set result.buffer = push(result.buffer, '-') -%}
        {%- elif char in dropped_to_space -%}
            {%- set result.buffer = push_whitespace(result.buffer) -%}
        {%- elif char == '…' -%}
            {%- set result.buffer = push(result.buffer, '.') -%}
        {%- elif char in spoken -%}
            {%- set result.buffer = push_whitespace(result.buffer) -%}
            {%- for spoken_char in spoken[char] -%}
                {%- set result.buffer = push(result.buffer, spoken_char) -%}
            {%- endfor -%}
            {%- set result.buffer = push_whitespace(result.buffer) -%}
        {%- elif char == ':' and previous is not none and not previous.isspace() and following is not none and not following.isspace() -%}
            {#- a `:` between two non-space characters, as in `10:30`, is kept -#}
            {%- set result.buffer = push(result.buffer, ':') -%}
        {%- elif char in ';:()' -%}
            {%- set result.buffer = push(result.buffer.rstrip(), ',') -%}
            {%- set result.buffer = push_whitespace(result.buffer) -%}
        {%- elif ('\U0001F600' <= char and char <= '\U0001FAFF') or ('☀' <= char and char <= '➿')
            or ('\U0001F1E6' <= char and char <= '\U0001F1FF') or char < ' ' or ('\x7f' <= char and char <= '\x9f')
            or char.isspace() -%}
            {#- emoji, control characters and whitespace -#}
            {%- set result.buffer = push_whitespace(result.buffer) -%}
        {%- else -%}
            {%- set result.buffer = push(result.buffer, char) -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set words = namespace(items=[]) -%}
    {%- for word in result.buffer.rstrip().split(' ') -%}
        {%- set rewritten = rewrite_number(word, lang) -%}
        {%- set words.items = words.items + [rewritten if rewritten else word] -%}
    {%- endfor -%}
    {%- set text = words.items | join(' ') -%}
{%- endif -%}

{#- `prepare_text_prompt`: collapse whitespace, capitalise the first character, end alphanumeric text with a period -#}
{%- set text = text.replace('\n', ' ').replace('\r', ' ').split() | join(' ') -%}
{%- if text -%}
    {%- set text = text[0].upper() ~ text[1:] -%}
    {%- if text[-1].isalnum() -%}
        {%- set text = text ~ '.' -%}
    {%- endif -%}
{%- endif -%}
{{- voice ~ text -}}
{%- if add_generation_prompt -%}
    {{- '<|audio_bos|>' -}}
{%- endif -%}
{%- endif -%}
"""

AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_PROMPT_TOKEN = "<|audio_prompt|>"
AUDIO_LATENT_TOKEN = "<|audio_latent|>"
# The placeholders take the tokenizer's first reserved control tokens (special pieces that text never produces, unused
# by the runtime), renamed: the first audio position, the voice frames and the audio latents.
RESERVED_TOKENS = {
    "\x00<|c000|>": AUDIO_BOS_TOKEN,
    "\x00<|c001|>": AUDIO_PROMPT_TOKEN,
    "\x00<|c002|>": AUDIO_LATENT_TOKEN,
}

# The transformers format sits next to Gradium's runtime package: the runtime's `config.json` gets the transformers
# fields and its `tokenizer.json` the placeholder names, its weights (`model.safetensors`, renamed at load by
# `conversion_mapping.py`) are read as they are, and these files are added.
TRANSFORMERS_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
TRANSFORMERS_FILES += ["processor_config.json", "generation_config.json"]

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


def read_gguf(path: Path) -> dict[str, torch.Tensor]:
    reader = GGUFReader(str(path))
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type).astype("float32").copy())
    return tensors


def hidden_act(original: dict, default: str = "erf") -> str:
    """The runtime's `gelu` field (`erf` unless stated, for Gradium's checkpoints) as a transformers activation."""
    return {"erf": "gelu", "tanh": "gelu_pytorch_tanh"}[original.get("gelu", default)]


def build_audio_tokenizer_config(original: dict) -> PocketTTSAudioTokenizerConfig:
    """The package's codec, decoder only: the encoder is not released."""
    mimi = original["mimi"]
    if mimi["quantizer_output_dimension"] != mimi["dimension"]:
        raise ValueError("The quantizer output projection is expected to map to the codec width.")
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
        "hidden_act": hidden_act(mimi),
        "layer_scale_initial_scale": mimi["transformer_layer_scale"],
        "rope_parameters": {"rope_type": "default", "rope_theta": mimi["transformer_max_period"]},
    }
    config = PocketTTSAudioTokenizerConfig(**fields, encoder_config=None)
    if config.frame_size != samples_per_frame:
        raise ValueError(f"Frame size {config.frame_size} != {samples_per_frame} samples per frame.")
    return config


def build_conditioner_fields(original: dict) -> dict:
    """The runtime's summed conditioners, at their runtime defaults (`num_speakers` 1, `duration_delta` 0)."""
    conditioners = {conditioner["name"]: conditioner for conditioner in original["conditioners"]}
    if set(conditioners) - {"num_speakers", "duration_delta"}:
        raise ValueError(f"Unsupported conditioners: {sorted(conditioners)}")
    if set(original["fuser"]["sum"]) != set(conditioners):
        raise ValueError("Every conditioner is expected to be summed.")
    num_speakers = conditioners["num_speakers"]["lut"]
    fields = {
        "num_speakers": num_speakers["possible_values"].index("1"),
        "num_speakers_vocab_size": num_speakers["n_bins"] + 1,
        "num_speakers_embedding_dim": num_speakers["dim"],
        "duration_delta_embedding_dim": None,
    }
    if "duration_delta" in conditioners:
        duration_delta = conditioners["duration_delta"]["continuous"]
        fields.update(
            duration_delta=0.0,
            duration_delta_embedding_dim=duration_delta["dim"],
            duration_delta_scale=duration_delta["scale_factor"],
            duration_delta_max_period=duration_delta.get("max_period", 10000.0),
        )
    return fields


def build_config(original: dict, tokenizer_vocab: dict[str, int]) -> GradiumPhononTTSConfig:
    flow_lm = original["flow_lm"]
    # absent, the runtime takes Pocket TTS's (`rms_var`)
    if flow_lm.get("time_rms_norm", "rms_var") != "rms":
        raise ValueError("Only Gradium's time-embedding norm (`time_rms_norm: rms`) is supported.")
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
        hidden_act=hidden_act(flow_lm),
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

    # The checkpoint's special pieces carry a leading NUL in their content.
    special = {piece: f"\x00{piece}" for piece in ("<unk>", "<s>", "</s>", "<pad>")}
    tokenizer_config = {
        "tokenizer_class": "TokenizersBackend",
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
    return Tokenizer.from_file(str(output_dir / "tokenizer.json")).get_vocab()


def load_voice(model_dir: Path) -> torch.Tensor:
    """The package's default voice (or its first one), as `audio_prompt_latents` `(num_frames, audio_prompt_dim)`."""
    path = model_dir / "default-voice.safetensors"
    if not path.is_file():
        path = sorted((model_dir / "voices").glob("*.safetensors"))[0]
    return load_file(path)["speaker_wavs"][0].transpose(0, 1).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Gradium's runtime package: config.json, tokenizer.json, model.safetensors (or a GGUF), voices",
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

    # The weights stay in the runtime's layout. A GGUF-only package gets them dequantized as `model.safetensors`, which
    # the runtime reads too.
    files = list(TRANSFORMERS_FILES)
    weights_path = args.model_dir / "model.safetensors"
    if weights_path.is_file():
        weights = load_file(weights_path)
        shutil.copy(weights_path, args.output_dir / "model.safetensors")
    else:
        weights = read_gguf(args.model_dir / "model.q8.gguf")
        save_file(weights, args.output_dir / "model.safetensors")
        files.append("model.safetensors")
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

    processor = GradiumPhononTTSProcessor(
        PocketTTSFeatureExtractor(sampling_rate=original_config["mimi"]["sample_rate"]),
        AutoTokenizer.from_pretrained(args.output_dir),
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE,
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
    processor = GradiumPhononTTSProcessor.from_pretrained(args.output_dir)
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
            allow_patterns=files,
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
