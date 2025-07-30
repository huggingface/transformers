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

# Templates

The [chat pipeline](./conversations) guide introduced [`TextGenerationPipeline`] and the concept of a chat prompt or chat template for conversing with a model. Underlying this high-level pipeline is the [`apply_chat_template`] method. A chat template is a part of the tokenizer and it specifies how to convert conversations into a single tokenizable string in the expected model format.

In the example below, Mistral-7B-Instruct and Zephyr-7B are finetuned from the same base model but they’re trained with different chat formats. Without chat templates, you have to manually write formatting code for each model and even minor errors can hurt performance. Chat templates offer a universal way to format chat inputs to any model.

<hfoptions id="template">
<hfoption id="Mistral">

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
```
```md
<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
```

</hfoption>
<hfoption id="Zephyr">

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
```
```md
<|user|>\nHello, how are you?</s>\n<|assistant|>\nI'm doing great. How can I help you today?</s>\n<|user|>\nI'd like to show off how chat templating works!</s>\n
```

</hfoption>
</hfoptions>

This guide explores [`apply_chat_template`] and chat templates in more detail.

## apply_chat_template

Chats should be structured as a list of dictionaries with `role` and `content` keys. The `role` key specifies the speaker (usually between you and the system), and the `content` key contains your message. For the system, the `content` is a high-level description of how the model should behave and respond when you’re chatting with it.

Pass your messages to [`apply_chat_template`] to tokenize and format them. You can set [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) to `True` to indicate the start of a message.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
```md
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

Now pass the tokenized chat to [`~GenerationMixin.generate`] to generate a response.

```py
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```
```md
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

### add_generation_prompt
The [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) parameter adds tokens that indicate the start of a response. This ensures the chat model generates a system response instead of continuing a users message.

Not all models require generation prompts, and some models, like [Llama](./model_doc/llama), don’t have any special tokens before the system response. In this case, [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) has no effect.

```py
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
tokenized_chat
```
```md
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
```

### continue_final_message

The [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) parameter controls whether the final message in the chat should be continued or not instead of starting a new one. It removes end of sequence tokens so that the model continues generation from the final message.

This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy for instruction following when you know how to start its replies.

```py
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```

> [!WARNING]
> You shouldn’t use [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) and [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) together. The former adds tokens that start a new message, while the latter removes end of sequence tokens. Using them together returns an error.

[`TextGenerationPipeline`] sets [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) to `True` by default to start a new message. However, if the final message in the chat has the “assistant” role, it assumes the message is a prefill and switches to `continue_final_message=True`. This is because most models don’t support multiple consecutive assistant messages. To override this behavior, explicitly pass the [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) to the pipeline.

## Multiple templates

A model may have several different templates for different use cases. For example, a model may have a template for regular chat, tool use, and RAG.

When there are multiple templates, the chat template is a dictionary. Each key corresponds to the name of a template. [`apply_chat_template`] handles multiple templates based on their name. It looks for a template named `default` in most cases and if it can’t find one, it raises an error.

For a tool calling template, if a user passes a `tools` parameter and a `tool_use` template exists, the tool calling template is used instead of `default`.

To access templates with other names, pass the template name to the `chat_template` parameter in [`apply_chat_template`]. For example, if you’re using a RAG template then set `chat_template="rag"`.

It can be confusing to manage multiple templates though, so we recommend using a single template for all use cases. Use Jinja statements like `if tools is defined` and `{% macro %}` definitions to wrap multiple code paths in a single template.

## Template selection

It is important to set a chat template format that matches the template format a model was pretrained on, otherwise performance may suffer. Even if you’re training the model further, performance is best if the chat tokens are kept constant.

But if you’re training a model from scratch or finetuning a model for chat, you have more options to select a template. For example, [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) is a popular format that is flexbile enough to handle many use cases. It even includes support for [generation prompts](#add_generation_prompt), but it doesn’t add beginning-of-string (`BOS`) or end-of-string (`EOS`) tokens. If your model expects `BOS` and `EOS` tokens, set `add_special_tokens=True` and make sure to add them to your template.

```py
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

Set the template with the following logic to support [generation prompts](#add_generation_prompt). The template wraps each message with `<|im_start|>` and `<|im_end|>` tokens and writes the role as a string. This allows you to easily customize the roles you want to train with.

```py
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

The `user`, `system` and `assistant` roles are standard roles in chat templates. We recommend using these roles when it makes sense, especially if you’re using your model with the [`TextGenerationPipeline`].

```py
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

## Model training

Training a model with a chat template is a good way to ensure a chat template matches the tokens a model is trained on. Apply the chat template as a preprocessing step to your dataset. Set `add_generation_prompt=False` because the additional tokens to prompt an assistant response aren’t helpful during training.

An example of preprocessing a dataset with a chat template is shown below.

```py
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```
```md
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

After this step, you can continue following the [training recipe](./tasks/language_modeling) for causal language models using the `formatted_chat` column.

Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` as well to avoid duplicating them.

```py
apply_chat_template(messages, tokenize=False, add_special_tokens=False)
```

This isn’t an issue if `apply_chat_template(tokenize=True)`.
