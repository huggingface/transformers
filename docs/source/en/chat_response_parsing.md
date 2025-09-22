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

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": "Hey! Can you summarize the end of the Cold War as briefly as possible? Like, comically briefly. It should really leave out almost most of the relevant information."
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=1024)[0, input_ids.shape[1]:]
out_text = tokenizer.decode(outputs)
parsed = tokenizer.parse_response(out_text)
print(parsed)
```

And that's all you need to start using response parsing! `parse_response` should return a complete message dict that is ready to be appended to the chat history. 
When the tokenizer does not support response parsing, `parse_response` will throw an error. We hope to add support
to more tokenizers over time.

## Developers: Understanding a simple response schema

Under the hood, `parse_response` uses a **JSON schema** to parse the model output. A JSON schema represents
the structure of the output message dict. The schema is augmented with additional fields that indicate how the 
output message string should be parsed into the expected format. Let's take a look at the schema for a SmolLM response,
excluding tool calls for now:

```json
{
    "x-regex": "(?:<think>\\n?(?P<thinking>.+?)\\n?</think>)?\\s*(?P<content>.+?)?\\s*(?:<\\|im_end\\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "thinking": {"type": "string"}
    }
}
```

We can see that the schema describes a JSON "object" (a `dict`, in other words) with three keys: `role`, `content`, and `thinking`.
Because all assistant responses have the role "assistant", the `role` key is a `const`(ant). The other two keys are strings, extracted
from the named groups in the regex in the `x-regex` field.

Like chat templates, response schemas are set as a property of the tokenizer. To enable response parsing, all you need
to do is set `tokenizer.response_schema` to a valid schema dict, and `tokenizer.parse_response()` will work! Again, like
chat templates, this schema will be saved with the processor, so once you set it, you can use `save_pretrained()` or `push_to_hub()` to
save and share the schema. 

## Developers: Complex schemas

Now, let's look at a more complex schema, which includes tool calls, to gain more of an understanding of the parser
internals. For this, we'll use the `GPT-OSS` schema. GPT-OSS emits both tool calls and thinking blocks, and it uses
an unusual format where model responses are tagged with one of three "channels": `commentary` for things like
tool calls, `analysis` for chain of thought blocks, and `final` for messages intended to be sent to the user. 
A full message where the model calls a tool named `get_current_weather` might look like this, with some extra linebreaks added for clarity:

```text
<|channel|>analysis<|message|>
The user asks: "What is the weather like in SF?" So we need to get the current weather in San Francisco, CA. 
We need to call get_current_weather function. So we should call get_current_weather with location "San Francisco, CA".
<|end|>
<|start|>assistant<|channel|>commentary 
to=functions.get_current_weather <|constrain|>json<|message|>
{
  "location": "San Francisco, CA"
}
<|call|>
```

Parsing proceeds recursively; the output of a regex (or other parser) at one level becomes the input to the nodes below it.
In other words, don't feel like you have to parse the entire output in one enormous regex! Instead, start with the schema,
and then add regexes to extract the relevant chunks as you go. Here's a schema that will parse it, with some
explanatory comments:

```python
{
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        # "content" and "thinking" are both similar to the previous example, and just extract a single string
        # However, rather than using a single regex with named groups to extract both, we use a regex in each subkey.
        # When an object node has no parser/regex, the entire input string is passed to all of its children, so 
        # parsing can either be done with named groups at the object level, or with separate regexes at the property level.
        "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
        "tool_calls": {
            # "x-regex-iterator" uses re.findall to find multiple possible manages, and returns them as an
            # array/list. You don't need to worry about array handling, though - each item in the array will be
            # parsed by the `items` schema, so just write the schema for a single item.
            "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # A const property is a fixed value, and the input has no effect on it.
                    "type": {"const": "function"},
                    # Here, we wrap the entire tool call dict in a `{"function": ...}` block. The input string is passed through to it unchanged.
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex": "<\|message\|>(.*)",
                                # The "x-parser" field indicates that the extracted string should be parsed as JSON.
                                # The output is then passed to the schema nodes below and recursive parsing continues.
                                "x-parser": "json",
                                "additionalProperties": {"type": "any"},
                            },
                        },
                    },
                },
            },
        },
    },
}
```

## Developers: Schema reference

(not done yet)