<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Writing a chat template

A chat template is a [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) template stored in the tokenizer's [chat_template](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.chat_template) attribute. Jinja is a templating language that allows you to write Python-like code and syntax.


```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + |>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

If you stare at this for a while, you should realize that this is actually very like Python, albeit with some strange
`{%-` syntax. The template iterates over a list of messages, and for each message, it prints the role and content of 
the message, followed by an end-of-sequence token. If `add_generation_prompt=True`, it adds 
the starting header for an assistant message to the end of the conversation.

Load the written template as a string and assign it to the tokenizer's `chat_template` attribute. Once set, the template is used whenever you call [`~PreTrainedTokenizerBase.apply_chat_template`]. It is also saved
with the tokenizer whenever [`~PreTrainedTokenizer.save_pretrained`] or [`~PreTrainedTokenizer.push_to_hub`] is called. The template is saved in the `chat_template.jinja` file in the tokenizer directory. You can
edit this file directly to change the template, which is often easier than manipulating a template string.

## Template writing tips

The easiest way to start writing Jinja templates is to refer to existing templates. Use `print(tokenizer.chat_template)` on any chat model to see the template it's using. Try starting with simple models that don't call any tools or support RAG because tool-use models can have very complex templates. Finally, take a look at the [Jinja documentation](https://jinja.palletsprojects.com/en/stable/templates/#synopsis) for more details about formatting and syntax.

There are some specific tips and pitfalls you may encounter while writing chat templates specifically, though, and this section will cover some of them in more detail. 

### Writing multimodal chat templates

For multimodal templates, the `chat_template` attribute is set on the **processor**, not the tokenizer. The `content` key of a message is often a list of content dicts,
rather than just a single string. You may wish to check the type of each content item in the list, and handle it accordingly.

Generally, the template should not directly access image or video data. This is normally handled by the processor after template rendering has finished. Instead,
your template should emit a single special token like `<|image|>` or `<|video|>` when it encounters image or video content.  The processor will
expand the single special token out into a sequence of image or video tokens later. The exact tokens to emit depends on the model you're working with. We strongly recommend loading an existing multimodal processor to see how it handles data.

The example template below handles mixed image and text content.

```jinja
{%- for message in messages %}
    {%- if loop.index0 == 0 %}
        {{- bos_token }}
    {%- endif %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- '<|image|>' }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
```

This multimodal template is very similar to the more simple template above, but it checks for `content` lists,
and iterates over them to render `<|image|>` tokens where necessary. This allows images to be inserted "into the flow"
of user text.

Not all models work this way - some may move all images to the end of the user message,
for example. The chat template should always match the format the model was trained with.

### Trimming whitespace

Jinja prints any whitespace before or after a block of text. This can be an issue for chat templates because adding extra whitespace that was not present during model training can harm performance. To remove the whitespace, add `-` to the Jinja line syntax. This allows you to write your template with Pythonic indentation and linebreaks, without accidentally printing an indentation in the rendered output.

The example template below doesn't use `-`, resulting in extra whitespace being printed in the output.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

We strongly recommend using `-` to ensure only the intended content is printed.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### Special variables and callables


The only constants in a template are the `messages` variable and the `add_generation_prompt` boolean. However, you have
access to **any other keyword arguments that are passed** to the [`~PreTrainedTokenizerBase.apply_chat_template`] method.

This provides flexibility and enables support for use-cases we may not have thought of while designing the spec. The most common additional variable is `tools`, which contains a list of tools in JSON schema format. Although you can use any variable name you like, we highly recommend sticking to convention and using `tools` for this purpose. This makes templates more compatible with the standard API.

You also have access to any tokens contained in `tokenizer.special_tokens_map`, which often includes special tokens like `bos_token` and `eos_token`. Access these directly by name, like `{{- bos_token }}`.

There are two callable functions available to you. To call them, use `{{- function_name(argument) }}`.

- `raise_exception(msg)` raises a `TemplateException`. This is useful for debugging or warning users about incorrect template usage.
- `strftime_now(format_str)` retrieves the current date and time in a specific format, which is often required in system messages. It is equivalent to [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now) in Python.

### Compatibility with non-Python Jinja

Jinja is implemented in multiple languages and they generally have the same syntax. Writing a template in Python allows you to use Python methods such as [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) on strings or [items](https://docs.python.org/3/library/stdtypes.html#dict.items) on dicts. But this won't work if the template is used in a non-Python implementation, for example, when deploying with Javascript or Rust.

Make the changes below to ensure compatibility across all Jinja implementations.

- Replace Python methods with Jinja filters. For example, replace `string.lower()` with `string|lower` or `dict.items()` with `dict|dictitems`. Most of the changes follow the same pattern except `string.strip()`, which is replaced with `string|trim`. Refer to the list of [built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) for a complete list of filters.
- Replace `True`, `False`, and `None` (these are Python specific) with `true`, `false`, and `none` respectively.
- Directly rendering a dict or list may return different results in other implementations. For example, string entries may change from single-quote to double-quote. To avoid this, add the [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) filter to maintain consistency.

### Big templates

Newer models or models with features like [tool-calling](./chat_extras#tools) and [RAG](./chat_extras#retrieval-augmented-generation-rag) require larger templates that can be longer than 100 lines. It may be easier to write larger templates in a separate file. The line numbers in the separate file corresponds exactly to the line numbers in template parsing or execution errors, making it easier to debug any potential issues.

Write the template in a separate file and extract it to the chat template.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

You could also load an edited template back into the tokenizer.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## Templates for tools

There isn't a specific format for writing templates for tools but it is best to follow the standard API. This ensures the template is widely accessible across models without requiring users to write custom code to use tools with your model.

> [!WARNING]
> Formatting such as whitespace and special tokens are model-specific. Make sure everything exactly matches the format a model was trained with.

The following section lists elements of the standard API for writing templates for tools.

### Tool definitions

[Tools](./chat_extras) are passed as Python functions or a JSON schema. When functions are passed, a JSON schema is automatically generated and passed to the template. When a template accesses the `tools` variable, it is always a list of JSON schemas.

Even though a template always receive tools as a JSON schema, you may need to radically change this format when rendering them to match the format a model was trained with. For example, [Command-R](./model_doc/cohere) was trained with tools defined with Python function headers. The template internally converts JSON schema types and renders the input tools as Python headers.

The example below shows how a tool is defined in JSON schema format.

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

An example of handling tool definitions in a chat template is shown below. The specific tokens and layouts should be changed to match the ones the model was trained with.

```
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}
{%- endif %}
```

### Tool calls

In addition to rendering the tool definitions, you also need to render **tool calls** and **tool responses** in the template.

Tool calls are generally passed in the `tool_calls` key of an `"assistant”` message. This is always a list even though most tool-calling models only support single tool calls, which means the list usually only contains a single element.

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```

A common pattern for handling tool calls is shown below. You can use this as a starting point, but make sure you template actually matches the format the model was trained with!

```
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

### Tool responses

Tool responses are message dicts with the `tool` role. They are much simpler than tool calls, and usually only contain the `role`, `name` and `content` keys.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

Some templates may not even need the `name` key, in which case, you can write your template to only read the `content` key.

```
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## Contribute

Once a template is ready, set it to the `chat_template` attribute in the tokenizer and test it with [`~PreTrainedTokenizerBase.apply_chat_template`]. If it works as expected, then upload it to the Hub with [`~PreTrainedTokenizer.push_to_hub`].

Even if you're not the model owner, it is still helpful to add a template for a model with an empty or incorrect chat template. Open a [pull request](https://hf.co/docs/hub/repositories-pull-requests-discussions) on the model repository to add the template!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
