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

# Chat templates

The [chat pipeline](./conversations) guide covers the basics of storing chat histories and generating text from chat models using [`TextGenerationPipeline`]. 
This guide is intended for more advanced users, and covers the underlying classes and methods, as well as the key concepts you need to understand what's actually going on when you chat with a model.

The critical insight needed to understand chat models is this: All causal LMs, whether chat-trained or not, continue a sequence of tokens. When causal LMs are trained, the training usually begins with "pre-training" on a huge corpus of text, which creates a "base" model.
These base models are then often "fine-tuned" for chat, which means training them on data that is formatted as a sequence of messages. The chat is still just a sequence of tokens, though! The list of `role` and `content` dictionaries that you pass
to a chat model get converted to a token sequence, often with control tokens like `<|user|>` or `<|assistant|>` or `<|end_of_message|>`, which allow the model to see the chat structure. 
There are many possible chat formats, and different models may use different formats or control tokens, even if they were fine-tuned from the same base model!

Don't panic, though - you don't need to memorize every possible chat format in order to use chat models. Chat models come with **chat templates**, which indicate how they expect chats to be formatted.
You can access these with the [`apply_chat_template`] method. Let's see two examples. Both of these models are fine-tuned from the same `Mistral-7B` base model:

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

Note how `Mistral-7B-Instruct` uses `[INST]` and `[/INST]` tokens to indicate the start and end of user messages, while `Zephyr-7B` uses `<|user|>` and `<|assistant|>` tokens to indicate the roles of the speakers. This is why chat templates are important - with the wrong tokens, these models would have drastically worse performance!

## Using `apply_chat_template` in a chat

The input to `apply_chat_template` should be structured as a list of dictionaries with `role` and `content` keys. The `role` key specifies the speaker, and the `content` key contains the message. The common roles are `user` for messages from the user, `assistant` for messages from the model, and `system`, which represent directives on how the model should act, and is usually placed at the beginning of the chat.

[`apply_chat_template`] takes this list and returns a formatted, and optionally tokenized, sequence:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", torch_dtype=torch.bfloat16)

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

Now we can simply pass the tokenized chat to [`~GenerationMixin.generate`] to generate a response.

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

> [!WARNING]
> Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you tokenize later to avoid duplicating these tokens.
> This isn’t an issue if you use `apply_chat_template(tokenize=True)`, which means it's usually the safer option!

### add_generation_prompt

You may have noticed the [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) argument in the above examples. 
This argument adds tokens to the end of the chat that indicate the start of an `assistant` response. Remember: Beneath all the chat abstractions, chat models are still just language models that continue a sequence of tokens!
If you include tokens that tell it that it's now in an `assistant` response, it will correctly write a response, but if you don't include these tokens, the model may get confused and do something strange, like **continuing** the user's message instead of replying to it! 

Let's see an example to understand what `add_generation_prompt` is actually doing. First, let's format a chat without `add_generation_prompt`:

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

Now, let's format the same chat with `add_generation_prompt=True`:

```py
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenized_chat
```
```md
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant

```

Notice the extra `<|im_start|>assistant` at the end - this indicates the start of an `assistant` message, and so the model knows that what's coming next is an assistant response.

Not all models require generation prompts, and some models, like [Llama](./model_doc/llama), don’t have any special tokens before the `assistant` response. In these cases, [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) has no effect.

### continue_final_message

The [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) parameter controls whether the final message in the chat should be continued or not instead of starting a new one. It removes end of sequence tokens so that the model continues generation from the final message.

This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy of instruction following when you know how to start its replies.

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

[`TextGenerationPipeline`] sets [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) to `True` by default to start a new message. However, if the final message in the chat has the “assistant” role, it assumes the message is a prefill and switches to `continue_final_message=True`. This is because most models don’t support multiple consecutive assistant messages. To override this behavior, explicitly pass the [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) argument to the pipeline.


## Advanced: Model training

Training a model with a chat template is a good way to ensure the template matches the tokens the model was trained on. Apply the chat template as a preprocessing step to your dataset. Set `add_generation_prompt=False` because the additional tokens to prompt an assistant response aren’t helpful during training.

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
