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
while a [tool calling](./chat_extras) model might emit function names and arguments.

In all of these cases, though, the model simply emits a chain of tokens. We need some system to turn those tokens into
a structured response dict. That system is **response parsing**. It is controlled by a **response template**: a
small declarative spec that describes how the model's output is laid out. Just as the [`chat_template`](./chat_templating) turns
structured messages into tokens ready to input into the model, response templates turn generated tokens back into
structured dicts. These two systems form the "glue" layer that allows a universal API to be used with models
that have very different internal chat formats.

Just like chat templates, response templates are attached to the tokenizer. The main entry point is the
[`~PreTrainedTokenizerBase.parse_response`] method:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Summarize the end of the Cold War, very briefly."}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=1024)[0, input_ids.shape[1]:]
out_text = tokenizer.decode(outputs)
print(tokenizer.parse_response(out_text, prefix=input_ids[0]))
# → {"role": "assistant", "thinking": "...", "content": "..."}
```

Note that we need to pass the `prefix` (the prompt tokens) as well. This is because many chat templates start
messages or open thinking blocks before letting the model begin its response. Without the prefix, message parsing
becomes ambiguous. If the tokenizer has no response template set, `parse_response` will raise an error. We're working on adding
templates to more models as quickly as we can!

## Streaming response parsing

In the above example, we parse the model response all at once after generation has finished. Often, though, we may
want to parse partial messages as they are generated, especially in user-facing apps where we don't just want to
display a static page for a minute or two until the model is finished.

When you want streaming parsing, call `tokenizer.get_response_parser()`, which returns a [`~utils.chat_parsing.ResponseParser`].
As with `parse_response`, pass the chat prompt as `prefix=` so the parser knows about any parts of the message that 
were prefilled by the chat template. The returned object is a
stateful parser that you can feed text into as the model generates it:

```python
parser = tokenizer.get_response_parser(prefix=input_ids[0])
for event in parser.initial_events:
    handle(event)
for chunk in model_output:
    for event in parser.feed(chunk):
        handle(event)
message, final_events = parser.finalize()
for event in final_events:
    handle(event)
```

The parser will emit **events** as text is fed in, which indicate which region is currently being parsed. When
the region is complete, it will be emitted in a separate event with the fully parsed content. At the end of generation,
the `finalize()` method flushes any remaining text and emits any final events, as well as the complete message dict.

## Streaming events

Each streamed parsing event is a dict with a `type` key. There are three kinds:

| Type           | Description                                                                         | Contents                                                                                                                                                                                                           |
|----------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `region_open`  | Indicates that the model has started a new region, such as `content` or `thinking`. | `field` (str): the field name.                                                                                                                                                                                     |
| `region_chunk` | A chunk of text for the current region.                                             | `field` (str): the field name. `text` (str): the new chunk. `dirty` (bool): `True` if the chunk is raw text from a structured region (`json`, `xml-inline`, `kv-lines`) that has yet to be parsed; `False` if the chunk is part of a text-like region whose body is its final value. |
| `region_close` | Indicates that a region has finished, and that key is now finalized.                | `field` (str): the field name. `value` (any): the fully parsed value for the region                                                                                      |

`region_chunk` events are emitted for every region as bytes arrive, so a streaming UI can render progress
even for structured regions. For text-like regions (`text`, `int`, `float`, `bool`) chunks are flagged
`dirty=False`: each chunk is already part of the final value (modulo trailing whitespace stripped at
close). For structured regions (`json`, `xml-inline`, `kv-lines`) chunks are flagged `dirty=True`: the
text is the raw, un-parsed body — it's safe to display incrementally, but the *parsed* value (a dict,
list, etc.) only arrives in the matching `region_close` event. Either way, the finalized value of a
region is always carried by `region_close`, so consumers that don't care about intermediate rendering
can simply ignore `region_chunk` events.

If the chat `prefix` wrote anything into the message (e.g. the template opened a thinking block, or an
assistant prefill started a response before handing off to the model), the parser exposes those events as
`parser.initial_events` — a list you can replay into your renderer before feeding any model output. Regions
that were opened *and* closed inside the prefix produce a full `region_open` / `region_chunk` / `region_close`
sequence and their parsed value lands in the output dict, exactly as if the model itself had written them.

A typical event stream for the SmolLM example above looks like:

```python
{"type": "region_open",  "field": "thinking"}
{"type": "region_chunk", "field": "thinking", "text": "Some chain ",   "dirty": False}
{"type": "region_chunk", "field": "thinking", "text": "of thought...", "dirty": False}
{"type": "region_close", "field": "thinking", "value": "Some chain of thought..."}
{"type": "region_open",  "field": "tool_calls"}
{"type": "region_chunk", "field": "tool_calls", "text": '{"name": "greet_user", ', "dirty": True}
{"type": "region_chunk", "field": "tool_calls", "text": '"arguments": {"greeting": "Hi!"}}', "dirty": True}
{"type": "region_close", "field": "tool_calls",
 "value": {"type": "function", "function": {"name": "greet_user", "arguments": {"greeting": "Hi!"}}}}
```


## Advanced: Writing a response template

The best way to understand how to write a response template is to pick a concrete example. Here's what a raw
reply from `SmolLM` might look like:

```txt
<think>
Some chain of thought...
</think>

<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hi!"}}</tool_call>
```

When we parse this output in the standard message dict format, it should look like this:

```json
{
    "role": "assistant",
    "thinking": "Some chain of thought...",
    "tool_calls": [
        {"type": "function", "function": {"name": "greet_user", "arguments": {"greeting": "Hi!"}}}
    ]
}
```

And here's the template that parses it. Don't be intimidated - a lot of it is fairly self-explanatory!

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
            "close": "<|im_end|>",
            "content": "text",
        },
    },
}
```

Essentially, the template defines **fields** and **delimiters**. Each field corresponds to a key in the
output dict. Fields also include information for parsing the text inside their delimiters. There's one subtlety: The
`content` field has no `open`, because in SmolLM (and several other models), it's not marked by a special token. Instead,
`content` is stored in the space after the other regions, but before the end of the sequence. In our template, we
represent this as an **implicit / leftover** field that picks up any text not claimed by another region.

In addition to `fields`, the template supports two optional top-level keys:

- `defaults` — A dict of values pre-populated in the output (e.g. `{"role": "assistant"}`). Keys here are always
  retained in the parsed output, even if no field wrote to them; other keys are dropped when their field captured
  nothing.
- `start_anchor` (str) / `start_anchor_pattern` (str regex) — Marks where the current assistant message begins
  inside a chat prompt. When you pass `prefix=` to `parse_response` or `get_response_parser`, the parser
  right-truncates the prefix past the **last** occurrence of this anchor before processing it, so earlier
  turns in a multi-turn conversation don't pollute the current message's state. For ChatML-style models this
  is typically `"<|im_start|>assistant\n"`. If a template has no anchor, the whole prefix is processed
  verbatim (which still works for single-turn prompts).

As with chat templates, response templates are stored as tokenizer attributes and saved with the tokenizer. Unlike
chat templates, we save them inside `tokenizer_config.json` and not as a separate file, because their format fits
naturally in JSON, unlike a chat template Jinja script.

```python
tokenizer.response_template = template
tokenizer.save_pretrained(...)  # Written as a key in tokenizer_config.json
```

## Advanced: Field API Reference

Each field supports several keys. First, there are the keys that define how the field should be captured:

| Key             | Type               | Purpose                                                                                     |
|-----------------|--------------------|---------------------------------------------------------------------------------------------|
| `open`          | str or list[str]   | Literal string that opens this region. A list of strings means "match any of these".        |
| `open_pattern`  | str (regex)        | Regex alternative to `open`; named groups become capture variables available to `assemble`. |
| `close`         | str or list[str]   | Literal string (or list of strings) that closes this region. `"eos"` means end-of-stream.   |
| `close_pattern` | str (regex)        | Regex alternative to `close`.                                                               |
| `repeats`       | bool               | If true, the field is a list and each match appends. Default `false`.                       |
| `optional`      | bool               | If false and the region never matches, we raise an error. Default `true`.                   |

A field should have **either** `open` or `open_pattern`, but not both, and the same is true for `close` and `close_pattern`.

A field with **neither** `open` nor `open_pattern` is the **implicit** field: it's active whenever no explicit
region is open, so it captures leftover text. At most one field can be implicit. This is most often used when `content`
does not have special token tags, it's just written as plaintext after the other fields.

In addition to opening and closing delimiters, you can also specify `repeats`, which indicates that the field is a list
and the delimiters can match multiple times. This is most common for parallel tool calling, when a model emits
multiple tool calls simultaneously.

Finally, you can specify `optional: false` for fields that must be present. If parsing finishes and an optional field
was never opened, we raise an error instead of silently omitting the field.

The end of generation will close and finalize any open regions, even if their closing delimiter was not seen.

### Parsing the content of a field

Next, there are the keys that define how the content of the field is parsed after it's captured:

| Key             | Type             | Purpose                                                                                     |
|-----------------|------------------|---------------------------------------------------------------------------------------------|
| `content`       | str              | Name of the content parser (see below). Defaults to `"text"`.                               |
| `content_args`  | dict             | Parser-specific arguments.                                                                  |
| `assemble`      | dict/list/string | Output template (see **Assemble**). Defaults to returning the parsed content directly.      |

Let's go through these keys in order. The first (and most important) key is `content`. This indicates the content type
of the field, which determines the parser that will be used to convert the raw text captured in the field to the final output.
`content_args` are used to configure the parser, and allow us to support various format quirks without needing custom code.

The available parsers are:

| Parser       | Produces      | Useful `content_args`                                                                          |
|--------------|---------------|------------------------------------------------------------------------------------------------|
| `text`       | string        | `strip` (default `true`) — set `false` for verbatim capture                                    |
| `int`        | int           | `strip` (default `true`)                                                                       |
| `float`      | float         | `strip` (default `true`)                                                                       |
| `bool`       | bool          | `strip` (default `true`); accepts `"true"`/`"1"` (case-insensitive) as true                    |
| `json`       | any           | `transform` (jmespath), `allow_non_json`, `unquoted_keys`, `string_delims: [[open, close],...]`|
| `xml-inline` | dict          | `tag_pattern` (regex w/ named groups `key`/`value`), `value_parser`, `merge_duplicates`        |
| `kv-lines`   | dict          | `line_sep`, `kv_sep`, `value_parser`, `strip` (default `true`)                                 |

The `json` parser also accepts dialect arguments (`unquoted_keys`, `string_delims`) for models that emit JSON with cute
quirks that completely break the standard parser. The model
authors who are responsible for this being necessary know who they are and should feel an appropriate amount of shame.

### Assemble

For most models, the `assemble` key is unnecessary, which is fortunate because it's definitely the most complex and messy
part of this entire operation. It's used when the information we want is scattered inside the target field, possibly even
in the delimiters, and has to be captured separately before being reshaped into the final output. 

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