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

# Tool Use

It is very common for chat models to be trained with support for "function-calling" or "tool-use". These "tools" are functions,
supplied by the user, which the model can choose to call as part of its response. For example, models could have access to a calculator
tool so that they can perform arithmetic without having to do it internally in the neural net itself, which usually becomes unreliable
for larger inputs.

This guide will demonstrate how to define tools, how to pass them to a chat model, and how to handle the model's output when it calls a tool.

## Passing tools

When a model supports tool-use, you can pass functions to the `tools` argument of [`~PreTrainedTokenizerBase.apply_chat_template`].
The tools can be passed as either [JSON schema](https://json-schema.org/learn) or as Python functions. If you pass Python functions,
the arguments, argument types and function docstring will be parsed in order to generate the JSON schema automatically.

Although passing Python functions is very convenient, the parser can only handle [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
docstrings. Therefore, make sure you follow that format! Here are a couple of examples of well-formatted functions
that are ready to use as tools:


```py
def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]
```

You may, if you wish, add a `Returns:` block to the docstring and a return type to the function header, but most models
will not use this information. The parser will also ignore any of the actual code inside your function! What really
matters is the function name, the argument names, the argument types, and the docstring describing the function's purpose
and the purpose of its arguments. These create the "signature" that the model will use to decide whether to call the tool.

## A tool-calling example

Let's start by loading a model and tokenizer that supports tool-use like [NousResearch/Hermes-2-Pro-Llama-3-8B](https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B), but you can also consider a larger model like [Command-R](./model_doc/cohere) and [Mixtral-8x22B](./model_doc/mixtral) if your hardware can support it.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
```

Next, let's create a chat, just like you would with a normal chat model.

```py
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

Next, pass `messages` and a list of tools to [`~PreTrainedTokenizerBase.apply_chat_template`], then tokenize and generate a response. This is exactly
the same as with a normal chat model, except for the `tools` argument.

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=128)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
```

```txt
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

The chat model has emitted a tool call! Specifically, it called the `get_current_temperature` tool with the correct parameters from the docstring. It inferred France as the location based on Paris, and that it should use Celsius for the units of temperature.

Note that the model **cannot actually call the tool itself**. Instead, it will request a tool call, and it's your job to handle the call and append both the call and the result to the chat history.

Let's start with the call: We use the `tool_calls` key of an `assistant` message to hold the call. This is the recommended API, and should be supported by the chat template of most tool-using models.

> [!WARNING]
> Although this is similar to the OpenAI API, the OpenAI API uses a JSON string as its `tool_call` format. This may cause errors or strange model behavior if used in Transformers, which expects a dict.

<hfoptions id="tool-call">
<hfoption id="Llama">

```py
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

Next, we have to actually append the tool response to the chat history. For this, we generally use the `tool` role:

```py
messages.append({"role": "tool", "content": "22"})  # Note that the returned content is always a string!
```

Finally, allow the model to read the tool response and reply to the user:

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs.to(model.device), max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

```txt
The temperature in Paris, France right now is 22°C.<|im_end|>
```

> [!WARNING]
> Although the key in the assistant message is called `tool_calls`, in most cases models only emit a single tool call at a time. Some older models emit multiple tool calls at the same time, but this is a
> signficantly more complex process, as you need to handle multiple tool responses at once and disambiguate them, often using tool call IDs. If you try this sample code with a model and get an error from the
> chat template telling you to add a tool call ID, this is probably what's happened! Please refer to the model card to see exactly what format that model expects for tool calls.


## Advanced: Manually writing JSON Schemas

In the examples above, we passed Python functions to the `tools` argument of [`~PreTrainedTokenizerBase.apply_chat_template`]. This is a convenient way to define tools, but it is not the only way. You can also pass a [JSON schema](https://json-schema.org/learn/getting-started-step-by-step) directly.
You can also manually call the low-level functions that convert Python functions to JSON schemas, and then check or edit the generated schemas. This is usually not necessary, but we include it here so that advanced users can understand the underlying mechanics. It's particularly important
for chat template authors, since they will need to access the JSON schema to render the tool definitions.

The function that [`~PreTrainedTokenizerBase.apply_chat_template`] uses to convert Python functions to JSON schema is [get_json_schema](https://github.com/huggingface/transformers/blob/14561209291255e51c55260306c7d00c159381a5/src/transformers/utils/chat_template_utils.py#L205). Let's try calling it directly:

```py
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

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

We won't go into the details of JSON schema itself here, since it's already [very well documented](https://json-schema.org/) elsewhere. We will, however, mention that you can pass JSON schema dicts to the `tools` argument of [`~PreTrainedTokenizerBase.apply_chat_template`] instead of Python functions:

```py
# A simple function that takes no arguments
current_time = {
  "type": "function", 
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# A more complete function that takes two numerical arguments
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers', 
    'parameters': {
      'type': 'object', 
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        }, 
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      }, 
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```