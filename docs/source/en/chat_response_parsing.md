<!--Copyright 2025 The HuggingFace Team. All rights reserved.

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

It is increasingly common for chat models to generate structured outputs, rather than just a single reply string. 
The most common uses for structured outputs are [tool calling](./chat_extras) and [reasoning models](https://huggingface.co/reasoning-course).
Tool calling models can output tool calls, containing the name of the tool to call and any arguments to be passed to it,
while reasoning models often output reasoning steps as a "chain of thought". Some recent models even use both of these,
and may output reasoning and/or one or more tool calls before their final answer.

Models with structured outputs pose a challenge for chat templating, because the output needs to be parsed before it
can be appended to the chat. For a concrete example, let's say we ask [GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)
what the weather is like, and it thinks and decides to call a tool. Here's what the raw model output might look like:

```
<|start|><|assistant|><|channel|>analysis<|message|>The user asks: "What is the weather like in SF?" We need to get the location of the user? The user explicitly asks about SF (San Francisco).
So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. But we need to call function to get weather data.
So we should call get_current_weather with location "San Francisco, CA". Let's do that.

We will call function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{
  "location": "San Francisco, CA"
}
```

And here's what that output would look like as a chat message dict:

```json
{
  "role": "assistant",
  "thinking": "The user asks: \"What is the weather like in SF?\" We need to get the location of the user? The user explicitly asks about SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. But we need to call function to get weather data. So we should call get_current_weather with location \"San Francisco, CA\". Let's do that.",
  "tool_calls": [
    {
      "name": "get_current_weather",
      "arguments": {
        "location": "San Francisco, CA"
      }
    }
  ]
}
```

Chat **templates** give us a way to turn messages into formatted input for a model, but we need something else to
parse model output back into a standard message dict. This is what chat **parsing** is for.

## The `parse_response` method

Parsing a chat response on a model that supports it is straightforward. Simply take the raw, decoded output from
`generate()`, and pass it to the tokenizer's `parse_response` method:

# TODO Make a full example with SmolLM3
```python
inputs = tokenizer.apply_chat_template(chat, return_dict=True, return_tensors="pt", add_generation_prompt=True)
generated_ids = model.generate(**inputs)
output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
out_text = tokenizer.decode(output_ids, skip_special_tokens=False)

parsed = tokenizer.parse_response(out_text)
```

`parse_response` should return a complete message dict that is ready to be appended to the chat history. 
When the tokenizer does not support response parsing, `parse_response` will throw an error. Remember to include special
tokens when calling `decode()`!

## Understanding response schemas

Under the hood, `parse_response` uses a **JSON schema** to parse the model output. a JSON schema represents
the structure of the output message dict. The schema is augmented with additional fields that indicate how the 
output message string should be parsed into the expected format. Let's take a look at what those schemas look like:

TODO Use a reasoning model as a simple example - a tool model will have a much bigger schema!

## Writing schemas

(not done yet)

## Schema reference

(not done yet)