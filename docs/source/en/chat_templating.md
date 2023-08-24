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

# Chat Templates

## Introduction

An increasingly common use case for LLMs is **chat**. In a chat context, rather than continuing a single string
of text (as is the case with a standard language model), the model instead continues a conversation that consists
of one or more **messages**, each of which includes a **role** as well as message text.

All LLMs, including models fine-tuned for chat, operate on linear sequences of tokens, and do not intrinsically
have special handling for 'roles'. This means that role information is usually injected by adding control tokens
between messages, to indicate both the message boundary and the relevant roles.

Unfortunately, there isn't (yet!) a standard for which tokens to use, and so different models have been trained
with wildly different formatting and control tokens for chat. This is what **chat templates** aim to resolve. 

Chat conversations are typically represented as a list of dictionaries, where each dictionary contains `role`
and `content` keys, and represents a single chat message. Chat templates are strings containing a Jinja template that
specifies how to format a conversation for a given model into a single tokenizable sequence.

Let's make this concrete with a quick example using the "facebook/blenderbot-400M-distill" model:

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
that string will also be tokenized for us. Blenderbot's *chat template* is very simple, however. To see a more complex
template in action, let's use the "meta-llama/Llama-2-7b-chat-hf" model. Note that this model has gated access, so you
will have to request access on the repo if you want to run this code yourself:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
```

Note that this time, the tokenizer has added the control tokens [INST] and [/INST] to indicate the start and end of 
user messages (but not assistant messages!). 

## How do chat templates work?

The chat template for a model is stored on the `tokenizer.chat_template` attribute. If no chat template is set, the
default template for that model class is used instead. Let's take a look at one of those templates now:

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.chat_template
"{% for message in messages %}{% if message.role == 'user' %}{{ ' ' }}{% endif %}{{ message.content }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

That's kind of intimidating. Let's add some newlines and indentation to make it more readable:

```jinja2
{% for message in messages %}
{% if message.role == 'user' %}
    {{ ' ' }}
{% endif %}
{{ message.content }}
{% if not loop.last %}
    {{ '  ' }}
{% endif %}
{% endfor %}
{{ eos_token }}
```

If you've never seen one of these before, this is a [Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/).
Jinja is a templating language that allows you to write simple code that generates text. In many ways, the code and
syntax resembles Python.

