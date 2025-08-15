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

A chat template is a [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) template stored in the tokenizer's [chat_template](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.chat_template) attribute. Jinja is a templating language that allows you to write Python-like code and syntax.

An example template is shown below:

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
the message, followed by an end-of-sequence token. If the `add_generation_prompt` variable is set to `True`, it adds 
the starting header for an assistant message to the end of the conversation.

To use the template you've written, simply assign the template string to the tokenizer's `chat_template` attribute. Once set, the template is used whenever you call [`~PreTrainedTokenizerBase.apply_chat_template`]. It will also be saved
with the tokenizer whenever you call [`~PreTrainedTokenizer.save_pretrained`] or [`~PreTrainedTokenizer.push_to_hub`]. The template will be saved in the `chat_template.jinja` file in the tokenizer directory. You can
edit this file directly to change the template, which is often easier than manipulating a template string.

## Template writing tips

The easiest way to start writing Jinja templates is to refer to existing templates. Use `print(tokenizer.chat_template)` on any chat model to see the template it's using. Try starting with simple models that don't call any tools or support RAG, as tool-use models in particular can have very complex templates! Finally, take a look at the [Jinja documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) for more details about formatting and syntax.

There are some specific tips and pitfalls you may encounter while writing chat templates specifically, though, and this section will cover some of them in more detail. 

### Writing multimodal chat templates

For multimodal templates, remember that you are normally setting the `chat_template` attribute on the **processor**, not the tokenizer. Also, the `content` key of a message will often be a list of content dicts,
rather than just a single string. You may wish to check the type of each content item in the list, and handle it accordingly.

Secondly, be aware that your template should generally not directly access image or video data. This is normally handled by the processor after template rendering has finished. Instead,
your template should generally emit a single special token like `<|image|>` or `<|video|>` when it encounters image or video content, which the processor will later
expand out into a sequence of image or video tokens. The exact tokens you should emit will depend on the model you're working with. This can be quite confusing, so again,
we strongly recommend loading an existing multimodal processor and seeing how it handles things!

Below is an example of a template that handles mixed image and text content:

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

Note how this template is mostly doing the same thing as the simple template above, but it checks for `content` lists,
and iterates over them to render `<|image|>` tokens where necessary. This allows images to be inserted "into the flow"
of user text. Note that not all models work this way - some may move all images to the end of the user message,
for example. The chat template should always match the format the model was trained with.

### Trimming whitespace

Jinja prints any whitespace before or after a block of text. This can be an issue for chat templates because adding extra
whitespace that was not present during model training can harm performance! To avoid this, you can add `-` to
Jinja line syntax. Doing this will remove any whitespace before the symbol, which means you can write your template
with Pythonic indentation and linebreaks, without worrying about accidentally printing that indentation in the rendered
output. We strongly recommend using `-`!

Here is an example template where the lack of `-` will result in extra whitespace being printed in the output:

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

By adding `-`, we ensure that we only print the content we intend to:

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### Special variables and callables

You may have noticed that the templates above use the `messages` variable a lot, in addition to things like `add_generation_prompt`, `bos_token` and `eos_token`. A reasonable question to ask, then, is
"Where do these variables come from, and what other variables are available to me inside the template?"

The short answer is that the only constants are the `messages` variable and the `add_generation_prompt` boolean. However, you will also have
access to **any other keyword arguments that are passed to the [`~PreTrainedTokenizerBase.apply_chat_template`] method**.
This is a lot of flexibility, but we do it this way because it allows templates to support use-cases that we may not have thought of
while designing the spec. The most common additional variable is `tools`, which contains a list of tools in JSON schema format. Although you can use any variable name you like,
we highly recommend sticking to convention and using `tools` for this purpose, as it will make your template more compatible with the standard API.

You will also always have access to any tokens contained in `tokenizer.special_tokens_map`, which often includes special tokens like `bos_token` and `eos_token`. You can access these directly by name, like `{{- bos_token }}`.

And finally, there are two callable functions available to you. To call them, use `{{- function_name(argument) }}`.

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

Transformers chat template methods allow a user to pass tools as Python functions or a JSON schema. When functions are passed, a JSON schema is automatically generated and passed to the template. The `tools` variable in a template always takes a list of JSON schemas.

The specific tokens and tool descriptions should match the ones your model was trained with. Your model doesn't need to understand the JSON schema input because your template can translate the JSON schema into your models format. For example, [Command-R](./model_doc/cohere) was trained with tools defined with Python function headers, but the Command-R tool template accepts JSON schemas. The template internally converts types and renders the input tools as Python headers.

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

An example for handling tool definitions in a chat template is shown below. The specific tokens and tool descriptions should be changed to match the ones a model was trained with.

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

Tool calls, if present, is a list with the `"assistant”` role. This is always a list even though most tool-calling models only support single tool calls, which means the list usually only contains a single element.

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

A common pattern for handling tool calls is shown below.

```
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

### Tool responses

Tool responses are a message dict with the `role`, `name` (name of the function) and `content` (result of the tool call) keys.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

Not all the keys need to be used in the tool response. For example, if a model doesn’t expect the function name to be included in the tool response, then you can just include the `role` and `content`.

```
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## Contribute

Add a chat template by setting the `chat_template` attribute in the tokenizer and testing it with [`~PreTrainedTokenizerBase.apply_chat_template`]. If it works as expected, then you can upload it to the Hub with with [`~PreTrainedTokenizer.push_to_hub`].

Even if you're not the model owner, it is still helpful to add a template for a model with an empty chat template or a model that is using a default class template. Open a [pull request](https://hf.co/docs/hub/repositories-pull-requests-discussions) on the model repository to add the template.

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("model_name")
```
