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

The problem with structured outputs, though, is that LLMs outputs are not inherently structured. LLM APIs usually
accept and return message dicts, with keys like `role` and `content` and `thinking`, but internally, LLMs actually 
just continue a single sequence of tokens. We use a glue layer to connect the user-facing API to the actual token
stream of the model. To turn inputs into a token stream, we use [`chat_templates`](./chat_templating), which are covered in other
documents. This document is about the other half of that glue layer: **Response templates**, the system for turning the
generated tokens output by the model back into a structured response dict. 

In many ways, response templates perform the inverse operation to chat templates. With chat templates, you feed in
a list of messages, and you get tokens ready to input to the model. With response templates, you feed in the raw
model output tokens, and you get a structured message. Like chat templates, response 
templates allow users to ignore the messy details of what specific formats and control tokens a model expects,
and use a universal API of message dicts that works with any model.

The best way to understand response templates is to see them in action. The main entry point is the
[`~PreTrainedTokenizerBase.parse_response`] method:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Summarize the end of the Cold War, very briefly."}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
outputs = model.generate(input_ids, max_new_tokens=1024)[0, input_ids.shape[1]:]
out_text = tokenizer.decode(outputs)
print(tokenizer.parse_response(out_text, prefix=input_ids[0]))
# Outputs a structured dict: {"role": "assistant", "thinking": "...", "content": "..."}
```

When a tokenizer has a `response_template`, the `parse_response` method will cleanly turn an output message into a
structured dict, ready to append to the chat. Note that we need to pass the `prefix` (the prompt tokens) to this method as well. This is because many chat templates start
messages or open thinking blocks before letting the model begin its response, and so our parser needs to see the 
prompt to understand the message. 

If the tokenizer has no response template set, `parse_response` will raise an error. We're working on adding
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
    render(event)  # Display the partial message to the user however you want to
for chunk in model_output:
    for event in parser.feed(chunk):
        render(event)
message, final_events = parser.finalize()
for event in final_events:
    render(event)
```

The parser will emit **events** as text from the generation process is fed in. This indicates which region is currently being generated. When
the region is complete, it will be emitted in a separate event with the fully parsed content. At the end of generation,
the `finalize()` method flushes any remaining text and emits any final events, as well as the complete message dict.

## Streaming events

Each streamed parsing event is a dict with a `type` key. There are three kinds:

| Type           | Description                                                                         | Contents                                                                                                                        |
|----------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `region_open`  | Indicates that the model has started a new region, such as `content` or `thinking`. | `field` (str): the field name.                                                                                                  |
| `region_chunk` | A chunk of text for the current region.                                             | `field` (str): the field name. `text` (str): the new chunk. `dirty` (bool): `True` if the chunk is raw text that needs parsing. |
| `region_close` | Indicates that a region has finished, and that key is now finalized.                | `field` (str): the field name. `value` (any): the fully parsed value for the region                                             |

`region_chunk` events are emitted for every region as bytes arrive, so a streaming UI can render progress
even for structured regions. For text-like regions (`text`, `int`, `float`, `bool`) chunks are flagged
`dirty=False`: each chunk is already part of the final value (modulo trailing whitespace stripped at
close). For structured regions, like JSON-format tool calls, chunks are flagged `dirty=True`. This means
text is the raw, un-parsed body; it's safe to display incrementally, but the *parsed* value (a dict,
list, etc.) only arrives in the matching `region_close` event. Either way, the finalized value of a
region is always carried by `region_close`, so consumers that don't care about intermediate rendering
can simply ignore `region_chunk` events.

If the chat `prefix` wrote anything into the message (e.g. the template opened a thinking block, or an
assistant prefill started a response before handing off to the model), the parser exposes those events as
`parser.initial_events`, a list you can replay into your renderer before feeding any model output. Regions
that were opened *and* closed inside the prefix produce a full `region_open` / `region_chunk` / `region_close`
sequence and their parsed value lands in the output dict, exactly as if the model itself had written them.

A typical event stream might look like this:

```python
{"type": "region_open",  "field": "thinking"}
{"type": "region_chunk", "field": "thinking", "text": "I should ", "dirty": False}
{"type": "region_chunk", "field": "thinking", "text": "greet the user", "dirty": False}
{"type": "region_close", "field": "thinking", "value": "I should greet the user"}
{"type": "region_open",  "field": "tool_calls"}
{"type": "region_chunk", "field": "tool_calls", "text": '{"name": "greet_user", ', "dirty": True}
{"type": "region_chunk", "field": "tool_calls", "text": '"arguments": {"greeting": "Hi!"}}', "dirty": True}
{"type": "region_close", "field": "tool_calls", "value": {"type": "function", "function": {"name": "greet_user", "arguments": {"greeting": "Hi!"}}}}
```

Note how `thinking` is emitted with `dirty=False`, because fields like `thinking` and `content` are usually just raw 
text. This means you can treat the chunks as valid "partial output". However, `tool_calls` is flagged as `dirty` because
the raw text needs significant cleanup - tool calls often need to be parsed as JSON or another format and then
restructured to generate the final tool call dict. As a result, the final output for these regions often looks very,
very different from the raw text. This final parsing will only happen when `region_close` is reached. It's
up to you what you want to do with the `dirty` chunks until then - you can display them as-is to show the user the 
"raw" output, or you can simply wait until you have something clean to display.

This concludes most of what you need to know to use response templates. The rest of this document is focused on
the internals of the parsing system and how to write response templates. This is mostly relevant for developers
and model authors. Most people can safely stop here!

## Advanced: Writing a response template

The best way to understand how to write a response template is to pick a concrete example. Here's what a raw
reply from `SmolLM` might look like:

```txt
<think>
I should greet the user
</think>

<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hi!"}}</tool_call>
```

When we parse this output in the standard message dict format, it should look like this:

```json
{
    "role": "assistant",
    "thinking": "I should greet the user",
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
            "transform": {"type": "function", "function": "{content}"},
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

Each field supports several keys. We can divide these into two types. First, there are the keys that define how the field should be captured:

| Key             | Type               | Purpose                                                                                       |
|-----------------|--------------------|-----------------------------------------------------------------------------------------------|
| `open`          | str or list[str]   | Literal string that opens this region. A list of strings means "match any of these".          |
| `open_pattern`  | str (regex)        | Regex alternative to `open`. Named groups become capture variables available to `transform`.  |
| `close`         | str or list[str]   | Literal string (or list of strings) that closes this region. `"eos"` means end-of-stream.     |
| `close_pattern` | str (regex)        | Regex alternative to `close`. Named groups become capture variables available to `transform`. |
| `repeats`       | bool               | If true, the field is a list and each match appends. Default `false`.                         |
| `optional`      | bool               | If false and the region never matches, we raise an error. Default `true`.                     |

A field should have **either** `open` or `open_pattern`, but not both, and the same is true for `close` and `close_pattern`.

A field with **neither** `open` nor `open_pattern` is the **implicit** field: it's active whenever no explicit
region is open, so it captures leftover text. At most one field can be implicit. This is most often used when `content`
does not have special token tags, it's just written as plaintext after the other fields.

In addition to opening and closing delimiters, you can also specify `repeats`, which indicates that the field is a list
and the delimiters can match multiple times. This is most common for parallel tool calling, when a model emits
multiple tool calls simultaneously.

Finally, you can specify `optional: false` for fields that must be present. If parsing finishes and a non optional field
was never opened, we raise an error instead of silently omitting the field.

The end of generation will close and finalize any open regions, even if their closing delimiter was not seen.

### Parsing the content of a field

Once we define how to capture a field, we also need to specify how to parse the raw text inside that capture. There are four
keys that control this:

| Key              | Type         | Purpose                                                                                  |
|------------------|--------------|------------------------------------------------------------------------------------------|
| `content`        | str          | The content type inside this region. Defaults to `"text"`. Each type has its own parser. |
| `content_args`   | dict         | Arguments to be passed to the content parser for this region.                            |
| `transform`      | dict/list    | Optional post-parse template that reshapes the parsed body (see **Transform**).          |
| `transform_each` | bool         | If true, the parsed content must be a list and `transform` is applied per-element.       |

The first (and most important) key is `content`. This indicates the content type
of the field, which determines the parser that will be used to convert the raw text captured in the field to the final output.
`content_args` are used to configure the parser, and allow us to support various format quirks without needing custom code.
We'll take a look at each type of parser and its arguments in turn.

#### Basic types
`text`, `int`, `float` and `bool` are the basic types. These content types all just strip whitespace and then do a simple
type conversion if required. They do not have any `content_args`, except for `text` which supports the arg `strip`,
which strips whitespace from the start and end of the captured text, and defaults to `true`.

#### json

The `json` parser parses the captured text as JSON. It's the workhorse for tool-call arguments and
anything else with nested structure. It accepts a handful of optional `content_args` to handle the
various ways models mangle JSON in the wild:

- `unquoted_keys` (bool, default `false`): rewrite bare-identifier keys (e.g. `{name: "foo"}`) into
  quoted form (`{"name": "foo"}`) before parsing. Useful for models that emit Javascript-style
  object literals rather than strict JSON.
- `string_delims` (list of `[open, close]` pairs, optional): for models that wrap string
  values in custom delimiters instead of `"..."`. Each pair gives an opening and closing marker;
  matching regions are extracted and spliced back in as standard JSON strings before parsing.
- `allow_non_json` (bool, default `false`): if parsing fails, return the stripped raw text instead
  of raising. Useful as a fallback for fields where the model *usually* emits JSON but occasionally
  drops to plain text.

The model authors responsible for the existence of `unquoted_keys` and `string_delims` know who
they are and should feel an appropriate amount of shame.

#### xml-inline

The `xml-inline` parser is for regions made up of a flat sequence of XML-ish tags, where each tag
becomes one entry in a dict. It's most often used inside a `tool_calls` field for models that emit
each argument as its own tag rather than as a JSON blob:

- `tag_pattern` (str, **required**): regex matching a single tag. Must contain named groups
  `key` (the resulting dict key) and `value` (the raw text that becomes the dict value).
- `value_parser` (dict, optional): nested content parser applied to each captured `value`. A dict
  with `name` (the parser, e.g. `"json"`, `"int"`) and optional `args` (its `content_args`). If
  omitted, values stay as raw strings.
- `merge_duplicates` (bool, default `false`): when the same key appears multiple times, collect the
  values into a list instead of letting later matches overwrite earlier ones.

For example, Qwen3 emits each tool-call argument as its own `<parameter>` tag, and we parse it
like this:

```python
"tool_calls": {
    "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
    "close": "</tool_call>",
    "repeats": True,
    "content": "xml-inline",
    "content_args": {
        "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
        "value_parser": {"name": "json", "args": {"allow_non_json": True}},
    },
    "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
}
```

Note the nested `value_parser`: each parameter value is itself run through the `json` parser (with
`allow_non_json` so plain strings still pass through).

#### kv-lines

The `kv-lines` parser handles line-delimited `key: value` pairs (think YAML-ish metadata or `.env`
files). Each line becomes one entry in the resulting dict. All arguments are optional:

- `line_sep` (str, default `"\n"`): separator between pairs.
- `kv_sep` (str, default `":"`): separator between a key and its value inside a single line. Only
  the first occurrence is used as the split point, so values may themselves contain the separator.
- `strip` (bool, default `true`): strip surrounding whitespace from each key and value.
- `value_parser` (dict, optional): nested content parser applied to each value, in the same
  `{"name": ..., "args": ...}` format as for `xml-inline`. If omitted, values stay as raw strings.

Lines that are empty or do not contain `kv_sep` are silently skipped, so stray blank lines in the
captured region are tolerated.


### Transform

For most fields, the `transform` key is unnecessary. It's used when the parsed body needs to be reshaped into the final
output, or when information from the delimiters has to be merged into the result. It most commonly appears in
`tool_calls` fields, as these often have complex structure.

`transform` is a **template**: a dict (or list) that describes the output shape, where any string of the
form `"{name}"` is replaced with the corresponding value. Values can be accessed from `content` (the parsed
body of this region) and any named groups captured by `open_pattern` / `close_pattern`. A very common use-case is to wrap a tool
call dict in an outer dict with a `function` key, as these are part of our standard tool call format:

```python
"tool_calls": {
    "open": "<tool_call>",
    "close": "</tool_call>",
    "repeats": True,
    "content": "json",
    "transform": {"type": "function", "function": "{content}"},
},
```

A whole-string placeholder like `"{content}"` returns the looked-up value with its type preserved — so above, the
parsed JSON dict slots in directly as the value of `function`. A placeholder must be the entire string: mixing
text and placeholders (`"abc {name} def"`) is not permitted. They're not f-strings!

You can abuse `transform` quite a lot, which becomes necessary when the model output has a wildly different format
to our standard API. GPT-OSS is a good example - it embeds the function name in the channel header rather than in
the JSON body, so we have to capture it with a named group in `open_pattern` and merge it with `content` inside the
transform. All named groups in `open_pattern` and `close_pattern` become available as variables alongside `content`:

```python
"tool_calls": {
    "open_pattern": r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>",
    "close": "<|call|>",
    "repeats": True,
    "content": "json",
    "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
},
```

Sometimes a field's parsed content is *itself* a list of records and you want to reshape each one. The Cohere
template is a good example: It emits all tool calls inside a single JSON array, so we set `transform_each: True` to 
apply the transform per element. Each array element's keys are unpacked into the template scope, so 
`"{tool_name}"` looks up `tool_name` in the current element:

```python
"tool_calls": {
    "open": "<|START_ACTION|>",
    "close": "<|END_ACTION|>",
    "content": "json",
    "transform_each": True,
    "transform": {"type": "function", "function": {"name": "{tool_name}", "arguments": "{parameters}"}},
},
```

This will convert an output like this:

```json
[
    {"tool_name": "greet_user", "parameters": {"greeting": "Hi!"}},
    {"tool_name": "search", "parameters": {"query": "weather tomorrow"}}
]
```

Into an output like this, which fits our standard API:

```json
[
    {"type": "function", "function": {"name": "greet_user", "arguments": {"greeting": "Hi!"}}},
    {"type": "function", "function": {"name": "search", "arguments": {"query": "weather tomorrow"}}}
]
```

The `transform_each` flag is only needed when `content` is already a list; for the more common case where each
match contributes one element (and `repeats: True` accumulates them), then the transform will apply to each element
by default.

## Very advanced: Regex portability

`open_pattern`, `close_pattern`, and `start_anchor_pattern` are regex strings. For most users, and even for most
model authors, this shouldn't be a problem, but if you are a developer writing an implementation of response parsing
in another language, you should be aware of our implementation details. This section is dedicated to everyone
who had to implement an entire Jinja parser to get non-Python chat templating to work - we hope that if you
follow the simple guidelines below, then response templates should be much less painful:

- We use Python's `re` module for regexes. Since all Python3 strings are unicode, this means **all of our regex matches
  are unicode-aware.** This particularly affects common characters like `\w`. Make sure you set the relevant
  unicode flags in your engine.
- We compile all regexes with `re.DOTALL` enabled and `re.MULTILINE` disabled, so `.` matches `\n` but `^` and `$`
  only match the start/end of the whole input, not line breaks.
- We use `(?P<name>...)` syntax for named groups. Other regex implementations have very different named capture group
  syntax, so you may need to search for this pattern in regexes and rewrite it to match your local implementation.
- Your regex engine may (rarely) not support lookarounds like `(?!...)`. Although these aren't commonly used in response
  templates, they can appear and we do support them! You might need to either throw an error in those cases, or manually
  extract the lookarounds and enforce them in your code when the regex engine finds a possible match.
- Other advanced features like backreferences, atomic groups, possessive quantifiers, recursion and so on are generally
  not used in response templates. We'll try to dissuade model authors from using them, so you can hopefully safely 
  ignore them.