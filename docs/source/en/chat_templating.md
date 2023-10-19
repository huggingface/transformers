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
of one or more **messages**, each of which includes a **role** as well as message text.

Most commonly, these roles are "user" for messages sent by the user, and "assistant" for messages sent by the model.
Some models also support a "system" role. System messages are usually sent at the beginning of the conversation
and include directives about how the model should behave in the subsequent chat.

All language models, including models fine-tuned for chat, operate on linear sequences of tokens and do not intrinsically
have special handling for roles. This means that role information is usually injected by adding control tokens
between messages, to indicate both the message boundary and the relevant roles.

Unfortunately, there isn't (yet!) a standard for which tokens to use, and so different models have been trained
with wildly different formatting and control tokens for chat. This can be a real problem for users - if you use the
wrong format, then the model will be confused by your input, and your performance will be a lot worse than it should be.
This is the problem that **chat templates** aim to resolve. 

Chat conversations are typically represented as a list of dictionaries, where each dictionary contains `role`
and `content` keys, and represents a single chat message. Chat templates are strings containing a Jinja template that
specifies how to format a conversation for a given model into a single tokenizable sequence. By storing this information
with the tokenizer, we can ensure that models get input data in the format they expect.

Let's make this concrete with a quick example using the `BlenderBot` model. BlenderBot has an extremely simple default 
template, which mostly just adds whitespace between rounds of dialogue:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

Notice how the entire chat is condensed into a single string. If we use `tokenize=True`, which is the default setting,
that string will also be tokenized for us. To see a more complex template in action, though, let's use the 
`meta-llama/Llama-2-7b-chat-hf` model. Note that this model has gated access, so you will have to
[request access on the repo](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) if you want to run this code yourself:

```python
>> from transformers import AutoTokenizer
>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>> tokenizer.use_default_system_prompt = False
>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
```

Note that this time, the tokenizer has added the control tokens [INST] and [/INST] to indicate the start and end of 
user messages (but not assistant messages!)

## How do chat templates work?

The chat template for a model is stored on the `tokenizer.chat_template` attribute. If no chat template is set, the
default template for that model class is used instead. Let's take a look at the template for `BlenderBot`:

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.default_chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

That's kind of intimidating. Let's add some newlines and indentation to make it more readable. Note that
we remove the first newline after each block as well as any preceding whitespace before a block by default, using the 
Jinja `trim_blocks` and `lstrip_blocks` flags. This means that you can write your templates with indentations and 
newlines and still have them function correctly!

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

## How do I create a chat template?

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

The method [`~PreTrainedTokenizer.apply_chat_template`] which uses your chat template is called by the [`ConversationalPipeline`] class, so 
once you set the correct chat template, your model will automatically become compatible with [`ConversationalPipeline`].

## What are "default" templates?

Before the introduction of chat templates, chat handling was hardcoded at the model class level. For backwards 
compatibility, we have retained this class-specific handling as default templates, also set at the class level. If a
model does not have a chat template set, but there is a default template for its model class, the `ConversationalPipeline`
class and methods like `apply_chat_template` will use the class template instead. You can find out what the default
template for your tokenizer is by checking the `tokenizer.default_chat_template` attribute.

This is something we do purely for backward compatibility reasons, to avoid breaking any existing workflows. Even when
the class template is appropriate for your model, we strongly recommend overriding the default template by
setting the `chat_template` attribute explicitly to make it clear to users that your model has been correctly configured
for chat, and to future-proof in case the default templates are ever altered or deprecated.

## What template should I use?

When setting the template for a model that's already been trained for chat, you should ensure that the template
exactly matches the message formatting that the model saw during training, or else you will probably experience
performance degradation. This is true even if you're training the model further - you will probably get the best 
performance if you keep the chat tokens constant. This is very analogous to tokenization - you generally get the
best performance for inference or fine-tuning when you precisely match the tokenization used during training.

If you're training a model from scratch, or fine-tuning a base language model for chat, on the other hand,
you have a lot of freedom to choose an appropriate template! LLMs are smart enough to learn to handle lots of different
input formats. Our default template for models that don't have a class-specific template follows the 
[ChatML format](https://github.com/openai/openai-python/blob/main/chatml.md), and this is a good, flexible choice for many use-cases. It looks like this:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

If you like this one, here it is in one-liner form, ready to copy into your code:

```
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

This template wraps each message in `<|im_start|>` and `<|im_end|>` tokens, and simply writes the role as a string, which
allows for flexibility in the roles you train with. The output looks like this:

```
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

The "user", "system" and "assistant" roles are the standard for chat, and we recommend using them when it makes sense,
particularly if you want your model to operate well with [`ConversationalPipeline`]. However, you are not limited
to these roles - templating is extremely flexible, and any string can be a role.

## I want to use chat templates! How should I get started?

If you have any chat models, you should set their `tokenizer.chat_template` attribute and test it using
[`~PreTrainedTokenizer.apply_chat_template`]. This applies even if you're not the model owner - if you're using a model
with an empty chat template, or one that's still using the default class template, please open a [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to
the model repository so that this attribute can be set properly!

Once the attribute is set, that's it, you're done! `tokenizer.apply_chat_template` will now work correctly for that
model, which means it is also automatically supported in places like `ConversationalPipeline`!

By ensuring that models have this attribute, we can make sure that the whole community gets to use the full power of
open-source models. Formatting mismatches have been haunting the field and silently harming performance for too long - 
it's time to put an end to them!