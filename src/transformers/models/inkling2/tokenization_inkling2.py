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
"""Tokenization classes for Inkling2."""

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend


PRETOKENIZE_REGEX = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Missing from the hub checkpoint.
INKLING2_CHAT_TEMPLATE = """{%- macro block(role_token, kind, payload, name='') -%}
    {{- role_token ~ name ~ '<|content_' ~ kind ~ '|>' ~ payload ~ '<|end_message|>' -}}
{%- endmacro -%}
{%- macro render_message(message, role_token, name) -%}
    {%- set turn = namespace(blocks=0) -%}
    {%- set reasoning = message.reasoning_content -%}
    {%- if message.role == 'assistant' and reasoning -%}
        {{- block(role_token, 'thinking', reasoning) -}}
        {%- set turn.blocks = turn.blocks + 1 -%}
    {%- endif -%}
    {%- if message.content is string -%}
        {%- if not (message.role == 'assistant' and message.content == '' and (reasoning or message.tool_calls)) -%}
            {{- block(role_token, 'text', message.content, name) -}}
            {%- set turn.blocks = turn.blocks + 1 -%}
        {%- endif -%}
    {%- elif message.content is iterable -%}
        {%- for part in message.content -%}
            {%- if part.type == 'text' -%}
                {{- block(role_token, 'text', part.text, name) -}}
            {%- elif part.type in ['image', 'image_url'] -%}
                {{- block(role_token, 'image', '<|unused_200054|>', name) -}}
            {%- elif part.type == 'thinking' -%}
                {{- block(role_token, 'thinking', part.thinking, name) -}}
            {%- elif part.type == 'tool_error' -%}
                {{- block(role_token, 'tool_error', part.text, name) -}}
            {%- else -%}
                {{- raise_exception('Unsupported content type for ' ~ message.role ~ ': ' ~ part.type) -}}
            {%- endif -%}
            {%- set turn.blocks = turn.blocks + 1 -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if message.role == 'assistant' and message.tool_calls -%}
        {%- for tool_call in message.tool_calls -%}
            {%- set function = tool_call.function if tool_call.function is defined else tool_call -%}
            {%- if function.arguments is string -%}
                {%- set arguments = function.arguments -%}
            {%- else -%}
                {%- set arguments = (function.arguments or {}) | tojson(separators=(',', ':'), sort_keys=true) -%}
            {%- endif -%}
            {{- block(role_token, 'invoke_tool_json', '{"name":' ~ (function.name | tojson) ~ ',"args":' ~ arguments ~ '}') -}}
            {%- set turn.blocks = turn.blocks + 1 -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if message.role == 'assistant' and turn.blocks > 0 -%}
        {{- '<|content_model_end_sampling|>' -}}
    {%- endif -%}
{%- endmacro -%}
{%- set role_tokens = {
    'system': '<|message_system|>',
    'developer': '<|message_system|>',
    'user': '<|message_user|>',
    'assistant': '<|message_model|>',
    'tool': '<|message_tool|>',
} -%}
{%- set effort_levels = {'none': 0.0, 'minimal': 0.1, 'low': 0.2, 'medium': 0.7, 'high': 0.9, 'xhigh': 0.99, 'max': 0.99} -%}
{%- if reasoning_effort is not defined or reasoning_effort is none -%}
    {%- if enable_thinking is defined and enable_thinking is false -%}
        {%- set effort = 0.0 -%}
    {%- else -%}
        {%- set effort = 0.9 -%}
    {%- endif -%}
{%- elif reasoning_effort is boolean -%}
    {%- set effort = -1.0 -%}
{%- elif reasoning_effort is string and reasoning_effort in effort_levels -%}
    {%- set effort = effort_levels[reasoning_effort] -%}
{%- else -%}
    {%- set effort = reasoning_effort | float(-1.0) -%}
{%- endif -%}
{%- if effort < 0 or effort > 0.99 -%}
    {{- raise_exception('reasoning_effort must be none, minimal, low, medium, high, xhigh, max, or a number in [0, 0.99]') -}}
{%- endif -%}
{%- set effort = effort | round(2) -%}
{%- set effort_text = (effort | int | string) if effort == (effort | int) else (effort | string) -%}
{%- set leading = namespace(count=0, done=false) -%}
{%- for message in messages -%}
    {%- if not leading.done and message.role in ['system', 'developer'] -%}
        {%- set leading.count = leading.count + 1 -%}
    {%- else -%}
        {%- set leading.done = true -%}
    {%- endif -%}
{%- endfor -%}
{%- if tools -%}
    {%- set declarations = namespace(items=[]) -%}
    {%- for tool in tools -%}
        {%- set function = tool.function if tool.function is defined else tool -%}
        {%- set declarations.items = declarations.items + [{
            'description': function.description or '',
            'name': function.name,
            'parameters': function.parameters or {},
            'type': tool.type or 'function',
        }] -%}
    {%- endfor -%}
    {{- block('<|message_system|>', 'xml', declarations.items | tojson(separators=(',', ':'), sort_keys=true), 'tool_declare') -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if loop.index0 == leading.count -%}
        {{- block('<|message_system|>', 'text', 'Thinking effort level: ' ~ effort_text) -}}
    {%- endif -%}
    {%- if message.role not in role_tokens -%}
        {{- raise_exception('Unsupported role: ' ~ message.role) -}}
    {%- endif -%}
    {%- set role_token = role_tokens[message.role] -%}
    {%- set name = message.name if message.role == 'tool' and message.name else '' -%}
    {%- if message.role == 'assistant' -%}
        {%- generation -%}
            {{- render_message(message, role_token, name) -}}
        {%- endgeneration -%}
    {%- else -%}
        {{- render_message(message, role_token, name) -}}
    {%- endif -%}
{%- endfor -%}
{%- if leading.count == messages | length -%}
    {{- block('<|message_system|>', 'text', 'Thinking effort level: ' ~ effort_text) -}}
{%- endif -%}
"""


# Missing from the hub checkpoint.
INKLING2_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|message_(?:system|user|tool)\|>.*?<\|end_message\|>|<\|content_model_end_sampling\|>",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"(?:<\|message_model\|>[^<]*)?<\|content_thinking\|>",
            "close": "<|end_message|>",
            "repeats": True,
            "join": "",
            "content": "text",
            "content_args": {"strip": False},
        },
        "content": {
            "open_pattern": r"(?:<\|message_model\|>[^<]*)?<\|content_(?:text|tool_error|invoke_tool_text)\|>",
            "close": "<|end_message|>",
            "repeats": True,
            "join": "",
            "content": "text",
            "content_args": {"strip": False},
        },
        "tool_calls": {
            "open_pattern": r"(?:<\|message_model\|>[^<]*)?<\|content_invoke_tool_json\|>",
            "close": "<|end_message|>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": {"name": "{content.name}", "arguments": "{content.args}"}},
        },
    },
}


# Custom tokenizer is needed because chat templates are missing from the hub checkpoint.
class Inkling2Tokenizer(TokenizersBackend):
    """
    Construct an Inkling2 (h200k) byte-level BPE tokenizer, backed by HuggingFace's *tokenizers* library.

    Args:
        vocab (`dict[str, int]`, *optional*):
            Mapping from tokens to ids.
        merges (`list[tuple[str, str]]`, *optional*):
            BPE merge rules.
        unk_token (`str`, *optional*):
            The unknown token. Unused, as byte-level BPE can encode any input.
        bos_token (`str`, *optional*):
            The beginning of sequence token.
        eos_token (`str`, *optional*):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The padding token.

    The `chat_template` and `response_template` keyword arguments default to the Inkling2 templates.
    """

    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = "<|endoftext|>",
        **kwargs,
    ):
        self._vocab = vocab if vocab is not None else {"<|endoftext|>": 0}
        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                fuse_unk=False,
                byte_fallback=False,
                ignore_merges=True,
            )
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()

        if kwargs.get("chat_template") is None:
            kwargs["chat_template"] = INKLING2_CHAT_TEMPLATE
        if kwargs.get("response_template") is None:
            kwargs["response_template"] = INKLING2_RESPONSE_TEMPLATE

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )


__all__ = ["Inkling2Tokenizer"]
