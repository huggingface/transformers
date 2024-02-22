<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Templates for Chat Models

## Introduction

An increasingly common use case for LLMs is **chat**. In a chat context, rather than continuing a single string
of text (as is the case with a standard language model), the model instead continues a conversation that consists
of one or more **messages**, each of which includes a **role**, like "user" or "assistant", as well as message text.

Much like tokenization, different models expect very different input formats for chat. This is the reason we added
**chat templates** as a feature. Chat templates are part of the tokenizer. They specify how to convert conversations, 
represented as lists of messages, into a single tokenizable string in the format that the model expects. 

Let's make this concrete with a quick example using the `BlenderBot` model. BlenderBot has an extremely simple default 
template, which mostly just adds whitespace between rounds of dialogue:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...    {"role": "user", "content": "Hello, how are you?"},
...    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...    {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

Notice how the entire chat is condensed into a single string. If we use `tokenize=True`, which is the default setting,
that string will also be tokenized for us. To see a more complex template in action, though, let's use the 
`mistralai/Mistral-7B-Instruct-v0.1` model.

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

Note that this time, the tokenizer has added the control tokens [INST] and [/INST] to indicate the start and end of 
user messages (but not assistant messages!). Mistral-instruct was trained with these tokens, but BlenderBot was not.

## How do I use chat templates?

As you can see in the example above, chat templates are easy to use. Simply build a list of messages, with `role`
and `content` keys, and then pass it to the [`~PreTrainedTokenizer.apply_chat_template`] method. Once you do that,
you'll get output that's ready to go! When using chat templates as input for model generation, it's also a good idea
to use `add_generation_prompt=True` to add a [generation prompt](#what-are-generation-prompts). 

Here's an example of preparing input for `model.generate()`, using the `Zephyr` assistant model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
This will yield a string in the input format that Zephyr expects. 
```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

Now that our input is formatted correctly for Zephyr, we can use the model to generate a response to the user's question:

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```

This will yield:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

Arr, 'twas easy after all!

## Is there an automated pipeline for chat?

Yes, there is! Our text generation pipelines support chat inputs, which makes it easy to use chat models. In the past,
we used to use a dedicated "ConversationalPipeline" class, but this has now been deprecated and its functionality
has been merged into the [`TextGenerationPipeline`]. Let's try the `Zephyr` example again, but this time using 
a pipeline:

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```

```text
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

The pipeline will take care of all the details of tokenization and calling `apply_chat_template` for you -
once the model has a chat template, all you need to do is initialize the pipeline and pass it the list of messages!

## What are "generation prompts"?

You may have noticed that the `apply_chat_template` method has an `add_generation_prompt` argument. This argument tells
the template to add tokens that indicate the start of a bot response. For example, consider the following chat:

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

Here's what this will look like without a generation prompt, using the ChatML template we saw in the Zephyr example:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

And here's what it looks like **with** a generation prompt:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

Note that this time, we've added the tokens that indicate the start of a bot response. This ensures that when the model
generates text it will write a bot response instead of doing something unexpected, like continuing the user's 
message. Remember, chat models are still just language models - they're trained to continue text, and chat is just a 
special kind of text to them! You need to guide them with appropriate control tokens, so they know what they're 
supposed to be doing.

Not all models require generation prompts. Some models, like BlenderBot and LLaMA, don't have any
special tokens before bot responses. In these cases, the `add_generation_prompt` argument will have no effect. The exact
effect that `add_generation_prompt` has will depend on the template being used.

## Can I use chat templates in training?

Yes! We recommend that you apply the chat template as a preprocessing step for your dataset. After this, you
can simply continue like any other language model training task. When training, you should usually set 
`add_generation_prompt=False`, because the added tokens to prompt an assistant response will not be helpful during 
training. Let's see an example:

```python
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
And we get:
```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

From here, just continue training like you would with a standard language modelling task, using the `formatted_chat` column.

## Advanced: How do chat templates work?

The chat template for a model is stored on the `tokenizer.chat_template` attribute. If no chat template is set, the
default template for that model class is used instead. Let's take a look at the template for `BlenderBot`:

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.default_chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

That's kind of intimidating. Let's add some newlines and indentation to make it more readable. Note that the first
newline after each block as well as any preceding whitespace before a block are ignored by default, using the 
Jinja `trim_blocks` and `lstrip_blocks` flags. However, be cautious - although leading whitespace on each
line is stripped, spaces between blocks on the same line are not. We strongly recommend checking that your template
isn't printing extra spaces where it shouldn't be!

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ ' ' }}
    {% endif %}
    {{ message['content'] }}
    {% if not loop.last %}
        {{ '  ' }}
    {% endif %}
{% endfor %}
{{ eos_token }}
```

If you've never seen one of these before, this is a [Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/).
Jinja is a templating language that allows you to write simple code that generates text. In many ways, the code and
syntax resembles Python. In pure Python, this template would look something like this:

```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

Effectively, the template does three things:
1. For each message, if the message is a user message, add a blank space before it, otherwise print nothing.
2. Add the message content
3. If the message is not the last message, add two spaces after it. After the final message, print the EOS token.

This is a pretty simple template - it doesn't add any control tokens, and it doesn't support "system" messages, which 
are a common way to give the model directives about how it should behave in the subsequent conversation.
But Jinja gives you a lot of flexibility to do those things! Let's see a Jinja template that can format inputs
similarly to the way LLaMA formats them (note that the real LLaMA template includes handling for default system
messages and slightly different system message handling in general - don't use this one in your actual code!)

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

Hopefully if you stare at this for a little bit you can see what this template is doing - it adds specific tokens based
on the "role" of each message, which represents who sent it. User, assistant and system messages are clearly
distinguishable to the model because of the tokens they're wrapped in.

## Advanced: Adding and editing chat templates

### How do I create a chat template?

Simple, just write a jinja template and set `tokenizer.chat_template`. You may find it easier to start with an 
existing template from another model and simply edit it for your needs! For example, we could take the LLaMA template
above and add "[ASST]" and "[/ASST]" to assistant messages:

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {% endif %}
{% endfor %}
```

Now, simply set the `tokenizer.chat_template` attribute. Next time you use [`~PreTrainedTokenizer.apply_chat_template`], it will
use your new template! This attribute will be saved in the `tokenizer_config.json` file, so you can use
[`~utils.PushToHubMixin.push_to_hub`] to upload your new template to the Hub and make sure everyone's using the right
template for your model!

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

The method [`~PreTrainedTokenizer.apply_chat_template`] which uses your chat template is called by the [`TextGenerationPipeline`] class, so 
once you set the correct chat template, your model will automatically become compatible with [`TextGenerationPipeline`].

<Tip>
If you're fine-tuning a model for chat, in addition to setting a chat template, you should probably add any new chat
control tokens as special tokens in the tokenizer. Special tokens are never split, 
ensuring that your control tokens are always handled as single tokens rather than being tokenized in pieces. You 
should also set the tokenizer's `eos_token` attribute to the token that marks the end of assistant generations in your
template. This will ensure that text generation tools can correctly figure out when to stop generating text.
</Tip>


### What are "default" templates?

Before the introduction of chat templates, chat handling was hardcoded at the model class level. For backwards 
compatibility, we have retained this class-specific handling as default templates, also set at the class level. If a
model does not have a chat template set, but there is a default template for its model class, the `TextGenerationPipeline`
class and methods like `apply_chat_template` will use the class template instead. You can find out what the default
template for your tokenizer is by checking the `tokenizer.default_chat_template` attribute.

This is something we do purely for backward compatibility reasons, to avoid breaking any existing workflows. Even when
the class template is appropriate for your model, we strongly recommend overriding the default template by
setting the `chat_template` attribute explicitly to make it clear to users that your model has been correctly configured
for chat.

Now that actual chat templates have been adopted more widely, default templates have been deprecated and will be
removed in a future release. We strongly recommend setting the `chat_template` attribute for any tokenizers that
still depend on them!

### What template should I use?

When setting the template for a model that's already been trained for chat, you should ensure that the template
exactly matches the message formatting that the model saw during training, or else you will probably experience
performance degradation. This is true even if you're training the model further - you will probably get the best 
performance if you keep the chat tokens constant. This is very analogous to tokenization - you generally get the
best performance for inference or fine-tuning when you precisely match the tokenization used during training.

If you're training a model from scratch, or fine-tuning a base language model for chat, on the other hand,
you have a lot of freedom to choose an appropriate template! LLMs are smart enough to learn to handle lots of different
input formats. One popular choice is the `ChatML` format, and this is a good, flexible choice for many use-cases. 
It looks like this:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

If you like this one, here it is in one-liner form, ready to copy into your code. The one-liner also includes
handy support for [generation prompts](#what-are-generation-prompts), but note that it doesn't add BOS or EOS tokens!
If your model expects those, they won't be added automatically by `apply_chat_template` - in other words, the
text will be tokenized with `add_special_tokens=False`. This is to avoid potential conflicts between the template and
the `add_special_tokens` logic. If your model expects special tokens, make sure to add them to the template!

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

This template wraps each message in `<|im_start|>` and `<|im_end|>` tokens, and simply writes the role as a string, which
allows for flexibility in the roles you train with. The output looks like this:

```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

The "user", "system" and "assistant" roles are the standard for chat, and we recommend using them when it makes sense,
particularly if you want your model to operate well with [`TextGenerationPipeline`]. However, you are not limited
to these roles - templating is extremely flexible, and any string can be a role.

### I want to add some chat templates! How should I get started?

If you have any chat models, you should set their `tokenizer.chat_template` attribute and test it using
[`~PreTrainedTokenizer.apply_chat_template`], then push the updated tokenizer to the Hub. This applies even if you're
not the model owner - if you're using a model with an empty chat template, or one that's still using the default class
template, please open a [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to the model repository so that this attribute can be set properly!

Once the attribute is set, that's it, you're done! `tokenizer.apply_chat_template` will now work correctly for that
model, which means it is also automatically supported in places like `TextGenerationPipeline`!

By ensuring that models have this attribute, we can make sure that the whole community gets to use the full power of
open-source models. Formatting mismatches have been haunting the field and silently harming performance for too long - 
it's time to put an end to them!

## Advanced: Template writing tips

If you're unfamiliar with Jinja, we generally find that the easiest way to write a chat template is to first
write a short Python script that formats messages the way you want, and then convert that script into a template.

Remember that the template handler will receive the conversation history as a variable called `messages`. Each
message is a dictionary with two keys, `role` and `content`. You will be able to access `messages` in your template
just like you can in Python, which means you can loop over it with `{% for message in messages %}` or access
individual messages with, for example, `{{ messages[0] }}`.

You can also use the following tips to convert your code to Jinja:

### For loops

For loops in Jinja look like this:

```
{% for message in messages %}
{{ message['content'] }}
{% endfor %}
```

Note that whatever's inside the {{ expression block }} will be printed to the output. You can use operators like
`+` to combine strings inside expression blocks.

### If statements

If statements in Jinja look like this:

```
{% if message['role'] == 'user' %}
{{ message['content'] }}
{% endif %}
```

Note how where Python uses whitespace to mark the beginnings and ends of `for` and `if` blocks, Jinja requires you
to explicitly end them with `{% endfor %}` and `{% endif %}`.

### Special variables

Inside your template, you will have access to the list of `messages`, but you can also access several other special
variables. These include special tokens like `bos_token` and `eos_token`, as well as the `add_generation_prompt`
variable that we discussed above. You can also use the `loop` variable to access information about the current loop
iteration, for example  using `{% if loop.last %}` to check if the current message is the last message in the 
conversation. Here's an example that puts these ideas together to add a generation prompt at the end of the
conversation if add_generation_prompt is `True`:

```
{% if loop.last and add_generation_prompt %}
{{ bos_token + 'Assistant:\n' }}
{% endif %}
```

### Notes on whitespace

As much as possible, we've tried to get Jinja to ignore whitespace outside of {{ expressions }}. However, be aware
that Jinja is a general-purpose templating engine, and it may treat whitespace between blocks on the same line
as significant and print it to the output. We **strongly** recommend checking that your template isn't printing extra
spaces where it shouldn't be before you upload it!