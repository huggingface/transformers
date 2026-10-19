# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Convert a MuseSpark checkpoint to a Hugging Face transformers checkpoint.

The conversion streams one source shard at a time. `--resume` picks an interrupted run back up
from the last shard it finished.
"""

import argparse
import json
import math
import re
import shutil
import struct
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import (
    AutoTokenizer,
    MuseGlimmerImageProcessor,
    MuseGlimmerVideoProcessor,
    MuseSparkAudioConfig,
    MuseSparkConfig,
    MuseSparkFeatureExtractor,
    MuseSparkProcessor,
    MuseSparkTextConfig,
    MuseSparkVisionConfig,
)


SOURCE_INDEX_NAME = "model.safetensors.index.json"

# `tokenizer.parse_response()` / `tokenizer.get_response_parser()` turn generated tokens back into a
# message dict, which RL training and structured serving need. Mirrors the shape the template emits.
RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start|>assistant",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"to=self<\|message\|>",
            "close": "<|eom|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">',
            "close": "</atem:invoke>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>'
                    r"(?P<value>.*?)</atem:parameter>"
                ),
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "open_pattern": r"to=(?:user|commentary)<\|message\|>",
            "close": ["<|eot|>", "<|eom|>"],
            "repeats": True,
            "join": "",
            "content": "text",
        },
    },
}


# Differences from the reference template:
#
#  - The bare `<image>` / `<video>` / `<audio>` placeholders the template *emits* are the special
#    tokens `<|patch|>` / `<|video|>` / `<|audio|>` instead. The `esc` macro still neutralises the
#    bare forms in caller-supplied text, since `MuseSparkProcessor` aliases them onto those tokens.
#  - The assistant branch is wrapped in `{% generation %}` / `{% endgeneration %}`, so
#    `apply_chat_template(..., return_assistant_tokens_mask=True)` returns a usable mask for
#    assistant-only loss. Rendered output is unchanged.
CHAT_TEMPLATE = r"""{%- set tools = tools | default(none) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set reasoning_effort = reasoning_effort | default('medium') -%}
{%- set cutoff_date = cutoff_date | default('2026-01-04') -%}
{%- set current_date = current_date | default(strftime_now('%Y-%m-%d') if strftime_now is defined else none) -%}
{%- set reasoning_effort_map = {'minimal': 1, 'low': 16, 'medium': 64, 'high': 256, 'xhigh': 512} -%}

{#- Neutralise control sequences in caller-supplied text so it cannot be
    tokenized as structure. "<|" is broken because every special token has the
    form "<|...|>"; the bare media placeholders are broken because the
    multimodal processor rewrites them positionally and a literal one in text
    would desynchronise placeholders from the attached media. -#}
{%- macro esc(value) -%}
    {{- value | string
        | replace('<|', '< |')
        | replace('<image>', '< image>')
        | replace('<video>', '< video>')
        | replace('<audio>', '< audio>') -}}
{%- endmacro -%}

{%- macro esc_header(value) -%}
    {{- esc(value) | replace('\r', ' ') | replace('\n', ' ') -}}
{%- endmacro -%}

{%- macro esc_attr(value) -%}
    {{- esc_header(value)
        | replace('&', '&amp;')
        | replace('"', '&quot;')
        | replace("'", '&#39;')
        | replace('<', '&lt;')
        | replace('>', '&gt;') -}}
{%- endmacro -%}

{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- esc(content) -}}
    {%- elif content is not none -%}
        {%- for part in content -%}
            {%- set part_type = part.get('type', '') -%}
            {%- if part_type in ['image', 'input_image', 'image_url'] -%}
                {{- '<|patch|>' -}}
            {%- elif part_type in ['video', 'input_video', 'video_url'] -%}
                {{- '<|video|>' -}}
            {%- elif part_type in ['audio', 'input_audio', 'audio_url'] -%}
                {{- '<|audio|>' -}}
            {%- elif part_type in ['text', 'input_text', 'output_text'] -%}
                {{- esc(part.get('text', '')) -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_atem(tc) -%}
    {%- set args = tc.function.arguments -%}
    {%- if args is not mapping -%}
        {{- raise_exception('This chat template requires tool_call.function.arguments to be a dict (mapping). The jinja sandbox has no JSON parser, so deserialize the arguments string before calling apply_chat_template.') -}}
    {%- endif -%}
    {{- '<atem:function_calls>\n<atem:invoke name="' -}}{{- esc_attr(tc.function.name) -}}{{- '">\n' -}}
    {%- for k, v in args.items() -%}
        {{- '<atem:parameter name="' -}}{{- esc_attr(k) -}}{{- '">' -}}
        {%- if v is boolean -%}
            {%- if v -%}
                true
            {%- else -%}
                false
            {%- endif -%}
        {%- elif v is none -%}
            null
        {%- elif v is mapping or (v is iterable and v is not string) -%}
            {{- esc(v | tojson) -}}
        {%- else -%}
            {{- esc(v) -}}
        {%- endif -%}
        {{- '</atem:parameter>\n' -}}
    {%- endfor -%}
    {{- '</atem:invoke>\n</atem:function_calls>' -}}
{%- endmacro -%}

{%- macro render_tool_defs(tools) -%}
    {{- 'In this environment you have access to a set of tools you can use to answer the user\'s question.\n\n' -}}
    {{- 'You can invoke a function by writing a "<atem:function_calls>" block like the following as part of your reply to the user:\n' -}}
    {{- '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n' -}}
    {{- 'String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n' -}}
    {{- 'Here are the functions available in JSONSchema format:\n' -}}
    {{- '// Tool metadata\n{"name": "", "description": ""}\n' -}}
    {{- '// Function schemas' -}}
    {%- for tool in tools -%}
        {%- set fn = tool.function if tool.function is defined else tool -%}
        {{- '\n{"name": ' -}}{{- esc(fn.name | tojson) -}}
        {{- ', "description": ' -}}{{- esc((fn.description if fn.description is defined and fn.description else '') | tojson) -}}
        {{- ', "parameters": ' -}}{{- esc((fn.parameters if fn.parameters is defined and fn.parameters else {}) | tojson) -}}{{- '}' -}}
    {%- endfor -%}
    {{- '\n\n' -}}
    {{- 'Here are examples of how to call functions in the tool set. To make parallel tool calls, emit each call as its own message (one tool call per message) in consecutive messages, as shown below:\n' -}}
    {{- '(If the tool namespace is not specified, invoke the function directly as `example_function_name_1` rather than `example_tool_name_1.example_function_name_1`)\n\n' -}}
    {{- 'to=example_tool_name_1.example_function_name_1\n\n' -}}
    {{- '<atem:function_calls>\n<atem:invoke name="example_tool_name_1.example_function_name_1">\n' -}}
    {{- '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n' -}}
    {{- '</atem:invoke>\n</atem:function_calls>\n\n' -}}
    {{- 'to=example_tool_name_2.example_function_name_2\n\n' -}}
    {{- '<atem:function_calls>\n<atem:invoke name="example_tool_name_2.example_function_name_2">\n' -}}
    {{- '<atem:parameter name="example_parameter_2">This is the value for the second parameter\nthat can span\n"multiple" lines\n</atem:parameter>\n' -}}
    {{- '<atem:parameter name="example_parameter_3">value_3</atem:parameter>\n' -}}
    {{- '</atem:invoke>\n</atem:function_calls>' -}}
{%- endmacro -%}

{%- macro render_valid_recipients(tools) -%}
    {%- set rns = namespace(recipients=[]) -%}
    {{- 'Use the appropriate recipient for each message:\n\n' -}}
    {%- set rns.recipients = rns.recipients + ['"self"'] -%}
    {{- '- "self": private reasoning and tool planning.\n' -}}
    {%- set rns.recipients = rns.recipients + ['"commentary"'] -%}
    {{- '- "commentary": user-visible intermediate updates while the assistant will continue working, including messages sent to the user before or between tool calls.\n' -}}
    {%- if tools -%}
        {%- set first_tool = tools | first -%}
        {%- set first_fn = (first_tool.function if first_tool.function is defined else first_tool).name -%}
        {%- set rns.recipients = rns.recipients + ['"[func_name] (e.g., ' ~ esc_header(first_fn) ~ ')"'] -%}
        {{- '- "tool" or "[tool name]": the name of the tool to call.\n' -}}
    {%- endif -%}
    {%- set rns.recipients = rns.recipients + ['"user"'] -%}
    {{- '- "user": messages that end the assistant turn, such as a completed response or clarification question that waits for the user\'s reply; do not use for pre-tool updates or partial responses.\n' -}}
    {{- '\n# Valid recipients: ' ~ rns.recipients | join(', ') ~ '.' -}}
{%- endmacro -%}

{{- bos_token -}}
{%- set ns = namespace(instructions=none, rest=messages) -%}
{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
    {%- set ns.instructions = render_content(messages[0]['content'])
          | replace('Reasoning effort', 'Reasoning strength')
          | replace('Reasoning Effort', 'Reasoning Strength')
          | replace('reasoning effort', 'reasoning strength')
          | replace('REASONING EFFORT', 'REASONING STRENGTH') -%}
    {%- set ns.rest = messages[1:] -%}
{%- endif -%}
{%- set sys = namespace(parts=['You are a helpful AI assistant.', 'Knowledge cutoff: ' ~ cutoff_date ~ '.']) -%}
{%- if current_date -%}
    {%- set sys.parts = sys.parts + ['Current date: ' ~ current_date ~ '.'] -%}
{%- endif -%}
{%- if ns.instructions -%}
    {%- set sys.parts = sys.parts + [ns.instructions] -%}
{%- endif -%}
{%- if tools -%}
    {%- set sys.parts = sys.parts + [render_tool_defs(tools) | trim] -%}
{%- endif -%}
{%- if reasoning_effort in reasoning_effort_map and 'reasoning strength:' not in ((ns.instructions if ns.instructions else '') | lower) -%}
    {%- set sys.parts = sys.parts + ['Reasoning strength: ' ~ reasoning_effort_map[reasoning_effort] ~ '.'] -%}
{%- endif -%}
{%- set sys.parts = sys.parts + [render_valid_recipients(tools) | trim] -%}
{{- '<|start|>system<|message|>' ~ sys.parts | join('\n\n') ~ '<|eot|>' -}}
{%- for message in ns.rest -%}
    {%- set role = message['role'] -%}
    {%- set end_token = '<|eom|>' if (not loop.last and ns.rest[loop.index0 + 1]['role'] == role) else '<|eot|>' -%}
    {%- if role in ['system', 'developer'] -%}
        {{- '<|start|>' + role + '<|message|>' -}}
        {{- render_content(message['content']) -}}
        {{- '<|eot|>' -}}
    {%- elif role == 'user' -%}
        {{- '<|start|>user<|message|>' -}}
        {%- if message.get('render_timestamp') and message.get('timestamp') -%}
            {%- set ts = message['timestamp'] -%}
            {%- if ts is string -%}
                {{- '[' ~ esc_header(ts) ~ '] ' -}}
            {%- elif ts.strftime is defined -%}
                {{- '[' ~ (ts.strftime('%a %Y-%m-%d %H:%M %Z') | trim) ~ '] ' -}}
            {%- endif -%}
        {%- endif -%}
        {{- render_content(message['content']) -}}
        {{- '<|eot|>' -}}
    {%- elif role == 'tool' -%}
        {%- set tname = message.get('name') -%}
        {%- if not tname -%}
            {%- set tcid = message.get('tool_call_id') -%}
            {%- set tns = namespace(name=tcid if tcid else '') -%}
            {%- for m in messages -%}
                {%- if m.get('tool_calls') -%}
                    {%- for tc in m['tool_calls'] -%}
                        {%- if tcid is not none and tc.id is defined and tc.id == tcid -%}
                            {%- set tns.name = tc.function.name -%}
                        {%- endif -%}
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
            {%- set tname = tns.name -%}
        {%- endif -%}
        {%- if not tname -%}
            {{- raise_exception('tool message has no name and no matching tool_call_id') -}}
        {%- endif -%}
        {{- '<|start|>tool ' -}}{{- esc_header(tname) -}}{{- '<|message|><tool_output name="' -}}{{- esc_attr(tname) -}}{{- '">\n' -}}
        {{- render_content(message['content']) -}}
        {{- '\n</tool_output><|eot|>' -}}
    {%- elif role == 'assistant' -%}
        {%- generation %}
        {%- set reasoning = message.get('reasoning_content') or message.get('reasoning') -%}
        {%- if reasoning -%}
            {{- '<|start|>assistant to=self<|message|>' -}}{{- esc(reasoning) -}}{{- '<|eom|>' -}}
        {%- endif -%}
        {%- set body = render_content(message.get('content')) -%}
        {%- if message.get('tool_calls') -%}
            {%- if body -%}
                {{- '<|start|>assistant to=commentary<|message|>' + body + '<|eom|>' -}}
            {%- endif -%}
            {%- for tc in message['tool_calls'] -%}
                {{- '<|start|>assistant to=' -}}{{- esc_header(tc.function.name) -}}{{- '<|message|>' -}}
                {{- render_atem(tc) -}}
                {{- (end_token if loop.last else '<|eom|>') -}}
            {%- endfor -%}
        {%- elif body or not reasoning -%}
            {%- set recipient = message.get('recipient') or 'user' -%}
            {%- if recipient == 'user' -%}
                {%- set end_turn = true -%}
            {%- elif recipient in ['self', 'commentary'] -%}
                {%- set end_turn = false -%}
            {%- elif message.get('end_turn') is not none -%}
                {%- set end_turn = message.get('end_turn') -%}
            {%- else -%}
                {%- set end_turn = false -%}
            {%- endif -%}
            {{- '<|start|>assistant to=' -}}{{- esc_header(recipient) -}}{{- '<|message|>' -}}
            {{- body -}}
            {{- ('<|eot|>' if end_turn else '<|eom|>') -}}
        {%- endif -%}
        {%- endgeneration %}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|start|>assistant' -}}
{%- endif -%}
"""

# The reference checkpoint keeps the audio tower under `soft_audio_perception`. Each pyramid stage is
# `stages.{layer}.{block}`, where block 0 is the layer's patch embedding and the rest are conv blocks.
AUDIO_SOURCE_PREFIX = "modalities.soft_audio_perception.interaction_encoder."
AUDIO_STAGE_PATTERN = re.compile(r"^stages\.(\d+)\.(\d+)\.(.+)$")

# Reference-config values the modeling code assumes rather than reads. A checkpoint that disagrees
# would silently run a different architecture, so the conversion refuses it instead.
EXPECTED_SOURCE_VALUES = {
    "non_linearity": "swiglu",
    "use_rope": True,
    "use_qk_norm": True,
    "use_residual_gate": True,
    "use_attn_o_gate": True,
    "use_attn_o_norm": True,
    "use_sqrt_attn_scaling": False,
    "normalize_tok_embeddings": True,
    "norm_zero_centered_gamma": True,
    "use_ffn_learnable_scales": True,
    "norm_affine": True,
    "post_norm_affine": False,
    # Sliding-window pattern [w, w, w, 0]: every 4th layer, counted backward from the last, is a
    # full-attention NoPE layer. `MuseSparkTextConfig` derives `layer_types`/`layer_rope_theta`
    # from exactly this shape.
    "every_n_layers_nope": 4,
    "moe.is_enabled": True,
    "moe.use_shared_expert": False,
    # Every decoder layer is a MoE layer, so the conversion only knows how to name MoE weights.
    "moe.interleave_moe_layer_step": 1,
    "moe.n_dense_layers_before_moe": 0,
    "moe.use_pre_expert_norm": True,
    "moe.use_post_expert_norm": True,
    # `MuseSparkTopkRouter` hardcodes the sigmoid over the router logits and the epsilon it adds to
    # the top-k score sum.
    "moe.router_score_normalization": "sigmoid",
    "moe.router_score_division_eps": 1e-15,
    # Vision tower: `MuseSparkVisionConfig` hardcodes full attention on every 4th layer, a
    # CLS-token-less patch embedder and the simple-MLP adapter.
    "vision_config.navit_config.sparse_attention_factor": 4,
    "vision_config.navit_config.use_cls_token": False,
    "vision_config.navit_config.use_rope": True,
    "vision_config.navit_config.adapter_type": "simple_mlp",
    "vision_config.navit_config.downsample_strategy": "ps",
    # Audio tower: `MuseSparkAudioModel` is the raw-waveform convolutional pyramid, with no
    # spectrogram front end and no patch encoder.
    "use_audio_perception": True,
    "audio_perception_modality_args.input_mode": "wav",
    "audio_perception_modality_args.encoder_family": "wav_ae",
    "audio_perception_modality_args.use_patch_encoder": False,
    # Token ids the tokenizer rename below has to agree with.
    "audio_perception_modality_args.next_audio_token": 201471,
    "audio_perception_modality_args.empty_audio_token": 201472,
}


def _get(source: dict, dotted_key: str):
    """Look up ``a.b.c`` in nested reference-config dicts."""
    value = source
    for key in dotted_key.split("."):
        if key not in value:
            raise KeyError(f"Reference config is missing {dotted_key!r}")
        value = value[key]
    return value


def _replace(path: Path) -> Path:
    """Remove ``path`` so it can be written as a fresh file.

    Checkpoint directories are commonly assembled out of symlinks into the hub cache, whose blobs are
    shared with every other checkpoint referencing them. Writing over such a link would follow it and
    corrupt the blob, so the link is dropped first and the write lands on a new local file.
    """
    if path.is_symlink() or path.exists():
        path.unlink()
    return path


def resolve_snapshot(input_path: Path) -> Path:
    """Accept either a plain checkpoint directory or a `models--*` hub cache directory."""
    if (input_path / "config.json").is_file():
        return input_path

    ref = input_path / "refs" / "main"
    if ref.is_file():
        snapshot = input_path / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").is_file():
            return snapshot

    snapshots = sorted((input_path / "snapshots").glob("*")) if (input_path / "snapshots").is_dir() else []
    if len(snapshots) == 1 and (snapshots[0] / "config.json").is_file():
        return snapshots[0]

    raise FileNotFoundError(f"No checkpoint with a config.json found under {input_path}")


def build_audio_config(snapshot: Path) -> MuseSparkAudioConfig:
    """Derive the audio tower's shape from the checkpoint."""
    shapes = read_source_shapes(snapshot, AUDIO_SOURCE_PREFIX)
    if not shapes:
        raise FileNotFoundError(f"Reference checkpoint has no audio weights under {AUDIO_SOURCE_PREFIX!r}")

    blocks = defaultdict(set)
    for key in shapes:
        match = AUDIO_STAGE_PATTERN.match(key.removeprefix(AUDIO_SOURCE_PREFIX))
        if match is not None:
            blocks[int(match.group(1))].add(int(match.group(2)))
    if set(blocks) != set(range(len(blocks))):
        raise ValueError(f"Reference audio tower has non-contiguous stages {sorted(blocks)}")

    hidden_sizes, patch_sizes, depths = [], [], []
    for layer_idx in range(len(blocks)):
        if blocks[layer_idx] != set(range(len(blocks[layer_idx]))):
            raise ValueError(f"Audio stage {layer_idx} has non-contiguous blocks {sorted(blocks[layer_idx])}")
        depths.append(len(blocks[layer_idx]) - 1)
        # The patch embedding folds `patch_size` frames of the previous stage's width into one token;
        # stage 0 reads the mono waveform itself, so its input width is 1.
        out_features, in_features = shapes[f"{AUDIO_SOURCE_PREFIX}stages.{layer_idx}.0.linear.weight"]
        in_channels = 1 if layer_idx == 0 else hidden_sizes[-1]
        if in_features % in_channels:
            raise ValueError(
                f"Audio stage {layer_idx} patch embedding takes {in_features} features, which is not a "
                f"whole number of {in_channels}-wide frames"
            )
        hidden_sizes.append(out_features)
        patch_sizes.append(in_features // in_channels)

    intermediate_size = shapes[f"{AUDIO_SOURCE_PREFIX}stages.0.1.linear1.weight"][0]
    return MuseSparkAudioConfig(
        patch_sizes=patch_sizes,
        hidden_sizes=hidden_sizes,
        depths=depths,
        num_hidden_layers=len(blocks),
        mlp_ratio=intermediate_size / hidden_sizes[0],
        conv_kernel_size=shapes[f"{AUDIO_SOURCE_PREFIX}stages.0.1.conv.conv.weight"][-1],
        out_hidden_size=shapes[f"{AUDIO_SOURCE_PREFIX}o_proj.weight"][0],
    )


def build_config(source: dict, snapshot: Path) -> MuseSparkConfig:
    """Translate the reference config into a `MuseSparkConfig`."""
    for key, expected in EXPECTED_SOURCE_VALUES.items():
        got = _get(source, key)
        if got != expected:
            raise ValueError(f"Unsupported reference config: {key} is {got!r}, expected {expected!r}")

    global_attn_cfg = source["global_attn_cfg"]
    sliding_window = global_attn_cfg[0]
    if global_attn_cfg != [sliding_window] * 3 + [0]:
        raise ValueError(
            f"Unsupported reference config: global_attn_cfg is {global_attn_cfg!r}, expected [w, w, w, 0]"
        )

    hidden_size = source["hidden_size"]
    head_dim = source["head_dim"]
    moe = source["moe"]

    text_config = MuseSparkTextConfig(
        vocab_size=source["vocab_size"],
        hidden_size=hidden_size,
        intermediate_size=int(source["ffn_dim_multiplier"] * hidden_size),
        num_hidden_layers=source["num_hidden_layers"],
        num_attention_heads=source["num_attention_heads"],
        num_key_value_heads=source["num_key_value_heads"],
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=source["max_position_embeddings"],
        rms_norm_eps=source["rms_norm_eps"],
        post_norm_eps=source["rms_post_norm_eps"],
        tie_word_embeddings=False,
        bos_token_id=200_000,
        # The single eos id the tokenizer uses. `generation_config.json` keeps the full stop list
        # [200001, 200008]; the two only have to agree in that the tokenizer's eos is a member of it.
        eos_token_id=200_001,
        pad_token_id=200_018,
        sliding_window=sliding_window,
        final_logit_softcapping=source["output_soft_cap_temp"],
        rope_parameters={"rope_type": "default", "rope_theta": source["rope_theta"]},
        # The reference config folds the post-QK-norm Q multiplier together with the head-dim scaling
        # as 43.7840518911 == qk_scale_factor * sqrt(head_dim); `MuseSparkTextConfig` stores the bare
        # multiplier.
        qk_scale_factor=43.7840518911 / math.sqrt(head_dim),
        # metaP output scaling: 1 / sqrt(hidden_size / base_width).
        output_multiplier=1 / math.sqrt(hidden_size / _get(source, "metap.base_width")),
        num_local_experts=moe["num_experts"],
        num_experts_per_tok=moe["topk"],
        num_shared_experts=0,
        moe_intermediate_size=moe["override_routed_hidden_dim"],
        moe_hidden_size=hidden_size // moe["compression_factor"],
        routed_scaling_factor=moe["route_scale"],
    )

    vision_source = source["vision_config"]
    navit = vision_source["navit_config"]
    # `patch_size`/`patch_temporal` are given twice
    for key in ("patch_size", "patch_temporal"):
        if vision_source[key] != navit[key]:
            raise ValueError(
                f"Unsupported reference config: vision_config.{key} is {vision_source[key]!r} but "
                f"vision_config.navit_config.{key} is {navit[key]!r}"
            )
    # The learned position-embedding table is stored for a square `image_size` grid of patches and
    # interpolated to the actual patch grid at run time.
    pos_emb = navit["image_size"] // navit["patch_size"]

    vision_config = MuseSparkVisionConfig(
        hidden_size=navit["hidden_size"],
        num_hidden_layers=navit["num_hidden_layers"],
        num_attention_heads=navit["num_attention_heads"],
        intermediate_size=navit["intermediate_size"],
        patch_size=vision_source["patch_size"],
        patch_temporal=vision_source["patch_temporal"],
        merge_size=vision_source["downsample_factor"],
        pos_emb_height=pos_emb,
        pos_emb_width=pos_emb,
        max_position_embeddings=pos_emb**2,
        hidden_act=navit["hidden_act"],
        rope_parameters={"rope_type": "default", "rope_theta": navit["rope_theta"]},
        layer_norm_eps=navit["norm_eps"],
    )

    config = MuseSparkConfig(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=build_audio_config(snapshot),
        # The reference config says 200090 (`<|image|>`) but the prompt scatters `<|patch|>` (200092)
        # per image patch, which is what the modeling code masks on.
        image_token_id=200_092,
        video_token_id=source["video_token_id"],
        audio_token_id=source["audio_token_id"],
        out_hidden_size=navit["vision_output_dim"],
        projector_hidden_size=vision_source["projector_input_dim"],
        projector_hidden_act=navit["hidden_act"],
        dtype="bfloat16",
    )
    config.architectures = ["MuseSparkForConditionalGeneration"]
    quant_config_path = snapshot / "hf_quant_config.json"
    if quant_config_path.is_file():
        with open(quant_config_path, encoding="utf-8") as f:
            quant_config = json.load(f)
        config.quantization_config = {
            "quant_method": quant_config["quant_method"],
            **{key: value for key, value in quant_config["quantization"].items() if value is not None},
        }
        target_keys = {
            target_key for entries in plan_weight_conversion(snapshot, config).values() for _, target_key, _ in entries
        }
        quantized_experts = {
            key.removesuffix("_weight_scale").rsplit(".", 1)[0]
            for key in target_keys
            if ".experts." in key and key.endswith("_weight_scale")
        }
        config.quantization_config["modules_to_not_convert"] = sorted(
            {
                key.rsplit(".", 1)[0]
                for key in target_keys
                if key.endswith(
                    (
                        ".q_proj.weight",
                        ".k_proj.weight",
                        ".v_proj.weight",
                        ".o_proj.weight",
                        ".gate_proj.weight",
                        ".up_proj.weight",
                        ".down_proj.weight",
                        ".pre_expert_proj.weight",
                        ".post_expert_proj.weight",
                        ".proj.weight",
                        ".fc1.weight",
                        ".fc2.weight",
                        ".patch_embedding.weight",
                        ".vision_projection.weight",
                        ".experts.gate_up_proj",
                        ".experts.down_proj",
                    )
                )
                or key == "lm_head.weight"
            }
            - quantized_experts,
            key=lambda name: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)],
        )
    return config


TEXT_TOP_LEVEL_RENAMES = {
    "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
    "model.norm.weight": "model.language_model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

TEXT_LAYER_RENAMES = {
    "self_attn.pre_qkv_proj_norm.weight": "input_layernorm.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "self_attn.o_gate_proj.weight": "self_attn.gate_proj.weight",
    "pre_ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
    "add_attn.gate": "post_attention_residual_gate.gate",
    "add_ffn.gate": "post_feedforward_residual_gate.gate",
    "feed_forward.expert_bias": "mlp.gate.e_score_correction_bias",
    "feed_forward.experts.w13_weight": "mlp.experts.gate_up_proj",
    "feed_forward.experts.w2_weight": "mlp.experts.down_proj",
    "feed_forward.pre_expert_norm.weight": "mlp.pre_expert_norm.weight",
    "feed_forward.post_expert_norm.weight": "mlp.experts.post_expert_norm.weight",
    "feed_forward.pooler.weight": "mlp.pre_expert_proj.weight",
    "feed_forward.unpooler.weight": "mlp.post_expert_proj.weight",
    # nvfp4 only
    "feed_forward.experts.w13_input_scale": "mlp.experts.gate_up_proj_input_scale",
    "feed_forward.experts.w13_weight_scale": "mlp.experts.gate_up_proj_weight_scale",
    "feed_forward.experts.w13_weight_scale_2": "mlp.experts.gate_up_proj_weight_scale_2",
    "feed_forward.experts.w2_input_scale": "mlp.experts.down_proj_input_scale",
    "feed_forward.experts.w2_weight_scale": "mlp.experts.down_proj_weight_scale",
    "feed_forward.experts.w2_weight_scale_2": "mlp.experts.down_proj_weight_scale_2",
}

VISION_TOP_LEVEL_RENAMES = {
    "vision_model.conv1_linear.weight": "model.vision_tower.patch_embedder.patch_embedding.weight",
    "vision_model.positional_embedding_vlm": ("model.vision_tower.patch_embedder.position_embedding_table.weight"),
    "vision_model.ln_pre.weight": "model.vision_tower.ln_pre.weight",
    "vision_model.ln_pre.bias": "model.vision_tower.ln_pre.bias",
    "vision_model.ln_post.weight": "model.vision_tower.ln_post.weight",
    "vision_model.ln_post.bias": "model.vision_tower.ln_post.bias",
    "vision_model.adapter.c_fc.weight": "model.vision_adapter.fc1.weight",
    "vision_model.adapter.c_proj.weight": "model.vision_adapter.fc2.weight",
}

AUDIO_TOP_LEVEL_RENAMES = {
    "norm.weight": "norm.weight",
    "o_proj.weight": "proj.weight",
    "o_proj.bias": "proj.bias",
    "dual_norm.weight": "post_proj_norm.weight",
}

AUDIO_PATCH_EMBEDDING_RENAMES = {
    "linear.weight": "patch_embedder.proj.weight",
    "linear.bias": "patch_embedder.proj.bias",
    "norm.weight": "patch_embedder.norm.weight",
}

AUDIO_CONV_BLOCK_RENAMES = {
    "conv.conv.weight": "conv.weight",
    "conv.conv.bias": "conv.bias",
    "norm.weight": "norm.weight",
    "linear1.weight": "mlp.fc1.weight",
    "linear1.bias": "mlp.fc1.bias",
    "linear2.weight": "mlp.fc2.weight",
    "linear2.bias": "mlp.fc2.bias",
}

VISION_LAYER_RENAMES = {
    "ln_1.weight": "norm1.weight",
    "ln_1.bias": "norm1.bias",
    "ln_2.weight": "norm2.weight",
    "ln_2.bias": "norm2.bias",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
    "attn.out_proj.weight": "attn.proj.weight",
    "attn.out_proj.bias": "attn.proj.bias",
    "attn.v_proj.weight": "attn.v_proj.weight",
    "attn.v_proj.bias": "attn.v_proj.bias",
}

TEXT_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
VISION_LAYER_PATTERN = re.compile(r"^vision_model\.encoder\.layers\.(\d+)\.(.+)$")


def permute_for_rope(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Rewrite a Q/K projection from the interleaved-pair RoPE layout to the split-half layout."""
    dim0 = tensor.shape[0]
    half_head = dim0 // num_heads // 2
    if tensor.ndim == 2:
        return tensor.view(num_heads, half_head, 2, tensor.shape[1]).transpose(1, 2).reshape(dim0, tensor.shape[1])
    if tensor.ndim == 1:
        return tensor.view(num_heads, half_head, 2).transpose(1, 2).reshape(dim0)
    raise ValueError(f"permute_for_rope: unexpected tensor shape {tuple(tensor.shape)}")


def transpose_router(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(0, 1)


def build_weight_conversions(config: MuseSparkConfig) -> tuple[dict, dict]:
    """Build the per-layer ``{source suffix: (target suffix, op)}`` tables for text and vision.

    `op` is either `None` (pure rename) or a callable applied to the tensor value.
    """
    text_layer = {source: (target, None) for source, target in TEXT_LAYER_RENAMES.items()}
    text_layer["self_attn.q_proj.weight"] = (
        "self_attn.q_proj.weight",
        partial(permute_for_rope, num_heads=config.text_config.num_attention_heads),
    )
    # k_proj is GQA-sharded to num_key_value_heads, unlike q_proj's num_attention_heads.
    text_layer["self_attn.k_proj.weight"] = (
        "self_attn.k_proj.weight",
        partial(permute_for_rope, num_heads=config.text_config.num_key_value_heads),
    )
    text_layer["feed_forward.router.weight"] = ("mlp.gate.weight", transpose_router)

    vision_layer = {source: (target, None) for source, target in VISION_LAYER_RENAMES.items()}
    permute_vision = partial(permute_for_rope, num_heads=config.vision_config.num_attention_heads)
    for projection in ("q_proj", "k_proj"):
        for suffix in ("weight", "bias"):
            name = f"attn.{projection}.{suffix}"
            vision_layer[name] = (name, permute_vision)

    return text_layer, vision_layer


def resolve_audio_key(suffix: str) -> str:
    """Map an audio weight's name, relative to `AUDIO_SOURCE_PREFIX`, onto its transformers name."""
    if suffix in AUDIO_TOP_LEVEL_RENAMES:
        return f"model.audio_tower.{AUDIO_TOP_LEVEL_RENAMES[suffix]}"

    match = AUDIO_STAGE_PATTERN.match(suffix)
    if match is not None:
        layer_idx, block_idx, rest = match.group(1), int(match.group(2)), match.group(3)
        if block_idx == 0 and rest in AUDIO_PATCH_EMBEDDING_RENAMES:
            return f"model.audio_tower.layers.{layer_idx}.{AUDIO_PATCH_EMBEDDING_RENAMES[rest]}"
        if block_idx > 0 and rest in AUDIO_CONV_BLOCK_RENAMES:
            # Block 0 is the patch embedding, so the conv blocks are renumbered from zero.
            target = AUDIO_CONV_BLOCK_RENAMES[rest]
            return f"model.audio_tower.layers.{layer_idx}.conv_blocks.{block_idx - 1}.{target}"

    raise KeyError(f"No conversion for reference weight {AUDIO_SOURCE_PREFIX + suffix!r}")


def resolve_source_key(key: str, text_layer: dict, vision_layer: dict):
    """Map a reference-checkpoint key onto its transformers name and op, or `None` to drop it."""
    if key == "modalities.vision_perception.encoder.vision_projection.weight":
        return "model.vision_projection.weight", None
    if key.startswith("modalities.vision_perception."):
        return None
    if key.startswith(AUDIO_SOURCE_PREFIX):
        return resolve_audio_key(key.removeprefix(AUDIO_SOURCE_PREFIX)), None
    if key in TEXT_TOP_LEVEL_RENAMES:
        return TEXT_TOP_LEVEL_RENAMES[key], None
    if key in VISION_TOP_LEVEL_RENAMES:
        return VISION_TOP_LEVEL_RENAMES[key], None

    match = TEXT_LAYER_PATTERN.match(key)
    if match is not None:
        layer_idx, suffix = match.groups()
        if suffix not in text_layer:
            raise KeyError(f"No conversion for reference weight {key!r}")
        target, op = text_layer[suffix]
        return f"model.language_model.layers.{layer_idx}.{target}", op

    match = VISION_LAYER_PATTERN.match(key)
    if match is not None:
        layer_idx, suffix = match.groups()
        if suffix not in vision_layer:
            raise KeyError(f"No conversion for reference weight {key!r}")
        target, op = vision_layer[suffix]
        return f"model.vision_tower.layers.{layer_idx}.{target}", op

    raise KeyError(f"No conversion for reference weight {key!r}")


def plan_weight_conversion(snapshot: Path, config: MuseSparkConfig) -> dict[str, list[tuple[str, str, object]]]:
    """Resolve every source key up front, grouped by the shard it lives in."""
    with open(snapshot / SOURCE_INDEX_NAME, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    keys_per_shard = defaultdict(list)
    for key, shard in weight_map.items():
        keys_per_shard[shard].append(key)

    text_layer, vision_layer = build_weight_conversions(config)
    plan, dropped = {}, 0
    for shard in sorted(keys_per_shard):
        entries = []
        for key in sorted(keys_per_shard[shard]):
            resolved = resolve_source_key(key, text_layer, vision_layer)
            if resolved is None:
                dropped += 1
                continue
            entries.append((key, *resolved))
        # A shard holding nothing but dropped weights contributes no output shard.
        if entries:
            plan[shard] = entries

    kept = sum(len(entries) for entries in plan.values())
    print(f"Planned {kept} weights across {len(plan)} shards; dropping {dropped} unused reference weights")
    return plan


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return a shard's tensor header and the total file size it implies."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    data_end = max((entry["data_offsets"][1] for entry in header.values()), default=0)
    return header, 8 + header_len + data_end


def read_source_shapes(snapshot: Path, prefix: str) -> dict[str, tuple[int, ...]]:
    """Read the shapes of every source weight under `prefix` out of the shard headers."""
    with open(snapshot / SOURCE_INDEX_NAME, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shapes = {}
    for shard in sorted({shard for key, shard in weight_map.items() if key.startswith(prefix)}):
        header, _ = read_safetensors_header(snapshot / shard)
        shapes.update({key: tuple(entry["shape"]) for key, entry in header.items() if key.startswith(prefix)})
    return shapes


def shard_is_complete(path: Path, target_keys: set[str]) -> bool:
    """Whether `path` is a fully written shard holding exactly `target_keys`.

    A run killed mid-write leaves a shard whose header promises more bytes than the file has, so the
    size implied by the header is checked rather than just the file's existence.
    """
    if not path.is_file():
        return False
    try:
        header, expected_size = read_safetensors_header(path)
    except (OSError, ValueError, struct.error):
        return False
    return set(header) == target_keys and path.stat().st_size == expected_size


def convert_weights(snapshot: Path, output_path: Path, config: MuseSparkConfig, resume: bool = False) -> None:
    plan = plan_weight_conversion(snapshot, config)
    num_shards = len(plan)
    weight_map = {}

    for shard_idx, (source_shard, entries) in enumerate(plan.items(), start=1):
        shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        target_keys = {target_key for _, target_key, _ in entries}
        for target_key in target_keys:
            weight_map[target_key] = shard_name

        # The plan is a pure function of the source index, so shard N holds the same weights on every
        # run and an already complete one can be left alone.
        if resume and shard_is_complete(output_path / shard_name, target_keys):
            print(f"  [{shard_idx}/{num_shards}] {shard_name} already complete, skipping", flush=True)
            continue

        tensors = {}
        with safe_open(snapshot / source_shard, framework="pt") as f:
            for source_key, target_key, op in entries:
                tensor = f.get_tensor(source_key)
                if op is not None:
                    tensor = op(tensor)
                # `transpose`/`view` leave a non-contiguous view that safetensors cannot serialize.
                tensors[target_key] = tensor.contiguous()

        save_file(tensors, _replace(output_path / shard_name), metadata={"format": "pt"})
        del tensors
        print(f"  [{shard_idx}/{num_shards}] {source_shard} -> {shard_name} ({len(entries)} weights)", flush=True)

    # Read the sizes back off the shards so a resumed run also accounts for the shards it skipped.
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        header, _ = read_safetensors_header(output_path / shard_name)
        for entry in header.values():
            start, end = entry["data_offsets"]
            total_size += end - start

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(_replace(output_path / SOURCE_INDEX_NAME), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"Wrote {len(weight_map)} weights ({total_size / 1e9:.1f} GB) across {num_shards} shards")


def write_tokenizer(snapshot: Path, output_path: Path) -> None:
    """Copy the tokenizer over, naming the audio special tokens and adding the training metadata."""
    pad_token = "<|finetune_right_pad|>"
    audio_special_token_renames = {
        201463: "<|soft_audio_start|>",
        201464: "<|soft_audio_end|>",
        201471: "<|next_audio|>",
        201472: "<|empty_audio|>",
    }

    for name in ("tokenizer.json", "tokenizer_config.json"):
        source_file = snapshot / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Reference checkpoint is missing {name}")
        content = source_file.read_text(encoding="utf-8")
        for token_id, token in audio_special_token_renames.items():
            placeholder = json.dumps(f"<|reserved_special_token_{token_id}|>")
            target = json.dumps(token)
            if content.count(placeholder) == 1:
                content = content.replace(placeholder, target)
            elif content.count(target) != 1:
                raise RuntimeError(
                    f"Expected {name} to name token {token_id} either {placeholder} or {target} exactly "
                    f"once, found {content.count(placeholder)} and {content.count(target)}"
                )
        if name == "tokenizer_config.json":
            # `pad_token` is in the vocab and in `generation_config.json`, but not declared here, so
            # `tokenizer.pad_token_id` would be None and `align_special_tokens` -- which treats the
            # tokenizer as the source of truth -- would throw the declared id away at train start.
            content = json.dumps(
                {**json.loads(content), "pad_token": pad_token, "response_template": RESPONSE_TEMPLATE},
                indent=2,
            )
            content += "\n"
        _replace(output_path / name).write_text(content, encoding="utf-8")

    # Redundant with `tokenizer_config.json`, but kept in sync with it rather than copied verbatim.
    with open(snapshot / "special_tokens_map.json", encoding="utf-8") as f:
        special_tokens_map = json.load(f)
    special_tokens_map["pad_token"] = pad_token
    with open(_replace(output_path / "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)
        f.write("\n")


def write_processor_config(
    output_path: Path, config: MuseSparkConfig, chat_template: str, source_config: dict
) -> None:
    """Write ``processor_config.json`` for `MuseSparkProcessor`."""
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    navit = source_config["vision_config"]["navit_config"]
    processor = MuseSparkProcessor(
        image_processor=MuseGlimmerImageProcessor(),
        # Timestamps from the video metadata are interleaved into the prompt by
        # `MuseSparkProcessor.replace_video_token`, so the metadata has to come back.
        video_processor=MuseGlimmerVideoProcessor(
            return_metadata=True,
            num_frames=int(navit["num_frames_per_video"]),
            max_video_frame_tokens=int(navit["max_num_tokens_per_frame"]),
            frame_timestamp_format="Time: {timestamp:.2f}s",
        ),
        tokenizer=tokenizer,
        feature_extractor=MuseSparkFeatureExtractor(samples_per_token=config.audio_config.samples_per_token),
        chat_template=chat_template,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor.save_pretrained(tmp_dir)
        shutil.copyfile(Path(tmp_dir) / "processor_config.json", _replace(output_path / "processor_config.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Reference checkpoint directory, or the `models--*` hub cache directory holding it.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Directory to write the HF checkpoint into. Must be empty unless --overwrite is passed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into --output_path even if it already holds files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the shards already written under --output_path and only convert the missing ones.",
    )
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.input_path)
    if args.output_path.resolve() == snapshot.resolve() or args.input_path.resolve() in (
        args.output_path.resolve(),
        *args.output_path.resolve().parents,
    ):
        raise ValueError("--output_path must be outside the reference checkpoint, which is read-only here")
    if args.output_path.is_dir() and any(args.output_path.iterdir()) and not (args.overwrite or args.resume):
        raise ValueError(f"{args.output_path} is not empty; pass --overwrite or --resume to convert into it")

    print(f"Converting {snapshot} -> {args.output_path}")
    args.output_path.mkdir(parents=True, exist_ok=True)

    with open(snapshot / "config.json", encoding="utf-8") as f:
        source_config = json.load(f)

    print("Writing config.json...")
    config = build_config(source_config, snapshot)
    _replace(args.output_path / "config.json")
    config.save_pretrained(args.output_path)

    print("Writing generation_config.json...")
    source_file = snapshot / "generation_config.json"
    if not source_file.is_file():
        raise FileNotFoundError("Reference checkpoint is missing generation_config.json")
    with open(source_file, encoding="utf-8") as f:
        generation_config = json.load(f)
    # Some reference checkpoints ship `top_p: 0.95`; the released checkpoints sample with `top_p: 1`.
    generation_config["top_p"] = 1
    _replace(args.output_path / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n", encoding="utf-8"
    )

    verbatim_files = ("l4_200k_base",)
    print(f"Copying {', '.join(verbatim_files)}...")
    for name in verbatim_files:
        source_file = snapshot / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Reference checkpoint is missing {name}")
        shutil.copyfile(source_file, _replace(args.output_path / name))

    print("Writing tokenizer.json, tokenizer_config.json, special_tokens_map.json...")
    write_tokenizer(snapshot, args.output_path)

    print("Writing chat_template.jinja...")
    _replace(args.output_path / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")

    print("Writing processor_config.json...")
    write_processor_config(args.output_path, config, CHAT_TEMPLATE, source_config)

    print("Converting model weights...")
    convert_weights(snapshot, args.output_path, config, resume=args.resume)
    print("Done.")


if __name__ == "__main__":
    main()
