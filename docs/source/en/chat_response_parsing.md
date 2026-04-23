<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Response Parsing

It is increasingly common for chat models to generate structured outputs, rather than just a single reply string. For example,
a [reasoning model](https://huggingface.co/reasoning-course) might emit a chain of thought containing its reasoning trace,
while a [tool calling](./chat_extras) model might emit function names and arguments to be called.

In all of these cases, though, the model simply emits a chain of tokens. We need some system to turn those tokens into
a structured response dict. That system is **response parsing**. It is controlled by a **response format**: a
small declarative spec that describes how the model's output is laid out.

Calling the parser is simple — you pass the generated text to [`~PreTrainedTokenizerBase.parse_response`]:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Summarize the end of the Cold War, very briefly."}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=1024)[0, input_ids.shape[1]:]
out_text = tokenizer.decode(outputs)
print(tokenizer.parse_response(out_text))
# → {"role": "assistant", "thinking": "...", "content": "..."}
```

If the tokenizer has no response format set, `parse_response` raises. Not every tokenizer ships one yet — support
is being added model by model.

## Writing a response format

The spec describes the **input stream**, left-to-right: a flat list of fields, where each field declares what opens
its region in the stream, what closes it, and what kind of content lives inside. The output dict falls out as a
byproduct.

Take SmolLM. Its output looks like:

```txt
<think>
Some chain of thought...
</think>

<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hi!"}}</tool_call>
```

The spec that parses this:

```python
{
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "content_args": {"transform": "{type: 'function', function: @}"},
        },
        "content": {
            "close_pattern": r"(?:<\|im_end\|>|$)",
            "content": "text",
        },
    },
}
```

Three fields, each describing one region of the stream. The `content` field has no `open` — that makes it the
**implicit / leftover** field that picks up any text not claimed by another region.

You attach the spec to a tokenizer the same way you attach a chat template:

```python
tokenizer.response_format = spec
tokenizer.save_pretrained(...)  # persisted to tokenizer_config.json
```

## Field keys

Each field supports:

| Key             | Type             | Purpose |
|-----------------|------------------|---------|
| `open`          | str              | Literal string that opens this region. |
| `open_pattern`  | str (regex)      | Regex alternative to `open`; named groups become capture variables available to `assemble`. |
| `close`         | str              | Literal string that closes this region. `"eos"` means end-of-stream. |
| `close_pattern` | str (regex)      | Regex alternative to `close`. |
| `content`       | str              | Name of the content parser (see below). Defaults to `"text"`. |
| `content_args`  | dict             | Parser-specific arguments. |
| `repeats`       | bool             | If true, the field is a list and each match appends. Default `false`. |
| `optional`      | bool             | If false and the region never matches, parsing raises. Default `true`. |
| `assemble`      | dict/list/string | Output template (see **Assemble**). Defaults to returning the parsed content directly. |
| `coerce`        | str              | `"int"`, `"float"`, or `"bool"` — applied to the final value. |

Use **either** `open` or `open_pattern`, not both. Same for `close`/`close_pattern`.

A field with **neither** `open` nor `open_pattern` is the **implicit** field: it's active whenever no explicit
region is open, so it captures leftover text. At most one field can be implicit.

## Content parsers

The parser decides how the region body is interpreted. The registry is closed — schemas can select and configure
parsers, but can't ship their own code:

| Parser       | Produces      | Useful `content_args`                                                              |
|--------------|---------------|-------------------------------------------------------------------------------------|
| `text`       | string        | `strip` (default `true`)                                                            |
| `raw`        | string        | (identity)                                                                          |
| `json`       | any           | `transform` (jmespath), `allow_non_json`                                            |
| `json-lax`   | any           | `unquoted_keys`, `string_delims: [[open, close], ...]`, plus all `json` args        |
| `xml-inline` | dict          | `tag_pattern` (regex w/ named groups `key`/`value`), `value_parser`                 |
| `kv-lines`   | dict          | `line_sep`, `kv_sep`, `value_parser`                                                |

`json-lax` is the escape hatch for models that emit JSON with quirks — unquoted keys, custom string delimiters
like Gemma's `<|"|>…<|"|>`. You configure it with parameters rather than writing a new parser.

## Assemble

By default, a region's parsed content becomes the field's value directly. Sometimes you want to reshape it — for
example, a tool call where the function name comes from the delimiter and the arguments come from the body.

`assemble` is a template: a dict/list/string where `{content}` is replaced by the parsed body and `{name}` (or any
other named group from `open_pattern`/`close_pattern`) is replaced by the captured text. Here's how GPT-OSS
handles tool calls whose function name is embedded in the channel header:

```python
"tool_calls": {
    "open_pattern": r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>",
    "close": "<|call|>",
    "repeats": True,
    "content": "json",
    "assemble": {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{content}"},
    },
},
```

For a call like `to=functions.get_weather ... {"location":"SF"}`, this produces
`{"type": "function", "function": {"name": "get_weather", "arguments": {"location": "SF"}}}`.

## Streaming

`tokenizer.response_event_stream()` returns a [`~utils.chat_parsing.ResponseEventStream`] — a stateful parser you
feed text incrementally as the model generates:

```python
streamer = tokenizer.response_event_stream()
for chunk in text_streamer:
    streamer.feed(chunk)
message = streamer.finalize()
```

The invariant that matters: `tokenizer.parse_response(full_text)` equals
`streamer.feed(chunk1); streamer.feed(chunk2); ...; streamer.finalize()` for any chunking of `full_text`.

## Example: re-expressing real formats

All of the formats currently tested in this repo — Cohere, ERNIE, GPT-OSS, SmolLM, Qwen3, Gemma 4 — fit in 15–25
lines of spec each. As a concrete case, here's Gemma 4, which emits tool call arguments with unquoted keys and
custom string delimiters (`<|"|>…<|"|>`):

```python
{
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {"open": "<|channel>thought\n", "close": "<channel|>", "content": "text"},
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json-lax",
            "content_args": {
                "unquoted_keys": True,
                "string_delims": [['<|"|>', '<|"|>']],
            },
            "assemble": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}
```

No model-specific Python — Gemma's format is fully expressed by configuring `json-lax`.

## Legacy `response_schema`

An earlier, nested-JSON-schema-shaped parser lives under `tokenizer.response_schema`. It is still honored by
`parse_response` for tokenizers that have it set, but new models should ship `response_format` instead — the new
format streams, handles format quirks without custom parsers, and is significantly shorter in practice. The legacy
path is expected to be removed in a future release.
