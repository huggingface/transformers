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

# 聊天模型的模板

## 介绍

LLM 的一个常见应用场景是聊天。在聊天上下文中，不再是连续的文本字符串构成的语句（不同于标准的语言模型），
聊天模型由一条或多条消息组成的对话组成，每条消息都有一个“用户”或“助手”等 **角色**，还包括消息文本。

与`Tokenizer`类似，不同的模型对聊天的输入格式要求也不同。这就是我们添加**聊天模板**作为一个功能的原因。
聊天模板是`Tokenizer`的一部分。用来把问答的对话内容转换为模型的输入`prompt`。


让我们通过一个快速的示例来具体说明，使用`BlenderBot`模型。
BlenderBot有一个非常简单的默认模板，主要是在对话轮之间添加空格：

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

注意，整个聊天对话内容被压缩成了一整个字符串。如果我们使用默认设置的`tokenize=True`，那么该字符串也将被tokenized处理。
不过，为了看到更复杂的模板实际运行，让我们使用`mistralai/Mistral-7B-Instruct-v0.1`模型。

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

可以看到，这一次tokenizer已经添加了[INST]和[/INST]来表示用户消息的开始和结束。
Mistral-instruct是有使用这些token进行训练的，但BlenderBot没有。

## 我如何使用聊天模板？

正如您在上面的示例中所看到的，聊天模板非常容易使用。只需构建一系列带有`role`和`content`键的消息，
然后将其传递给[`~PreTrainedTokenizer.apply_chat_template`]方法。
另外，在将聊天模板用作模型预测的输入时，还建议使用`add_generation_prompt=True`来添加[generation prompt](#什么是generation-prompts)。

这是一个准备`model.generate()`的示例，使用`Zephyr`模型：

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
这将生成Zephyr期望的输入格式的字符串。它看起来像这样：
```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

现在我们已经按照`Zephyr`的要求传入prompt了，我们可以使用模型来生成对用户问题的回复：

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```

输出结果是：

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```
啊，原来这么容易！

## 有自动化的聊天`pipeline`吗？

有的，[`ConversationalPipeline`]。这个`pipeline`的设计是为了方便使用聊天模型。让我们再试一次 Zephyr 的例子，但这次使用`pipeline`：

```python
from transformers import pipeline

pipe = pipeline("conversational", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages))
```

```text
Conversation id: 76d886a0-74bd-454e-9804-0467041a63dc
system: You are a friendly chatbot who always responds in the style of a pirate
user: How many helicopters can a human eat in one sitting?
assistant: Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

[`ConversationalPipeline`]将负责处理所有的`tokenized`并调用`apply_chat_template`，一旦模型有了聊天模板，您只需要初始化pipeline并传递消息列表！

## 什么是"generation prompts"?

您可能已经注意到`apply_chat_template`方法有一个`add_generation_prompt`参数。
这个参数告诉模板添加模型开始答复的标记。例如，考虑以下对话：

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

这是`add_generation_prompt=False`的结果，使用ChatML模板：
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

下面这是`add_generation_prompt=True`的结果：

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

这一次我们添加了模型开始答复的标记。这可以确保模型生成文本时只会给出答复，而不会做出意外的行为，比如继续用户的消息。
记住，聊天模型只是语言模型，它们被训练来继续文本，而聊天对它们来说只是一种特殊的文本！
你需要用适当的控制标记来引导它们，让它们知道自己应该做什么。

并非所有模型都需要生成提示。一些模型，如BlenderBot和LLaMA，在模型回复之前没有任何特殊标记。
在这些情况下，`add_generation_prompt`参数将不起作用。`add_generation_prompt`参数取决于你所使用的模板。

## 我可以在训练中使用聊天模板吗？

可以！我们建议您将聊天模板应用为数据集的预处理步骤。之后，您可以像进行任何其他语言模型训练任务一样继续。
在训练时，通常应该设置`add_generation_prompt=False`，因为添加的助手标记在训练过程中并不会有帮助。
让我们看一个例子：

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
结果是：
```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

这样，后面你可以使用`formatted_chat`列，跟标准语言建模任务中一样训练即可。
## 高级：聊天模板是如何工作的？

模型的聊天模板存储在`tokenizer.chat_template`属性上。如果没有设置，则将使用该模型的默认模板。
让我们来看看`BlenderBot`的模板：
```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.default_chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

这看着有点复杂。让我们添加一些换行和缩进，使其更易读。
请注意，默认情况下忽略每个块后的第一个换行以及块之前的任何前导空格，
使用Jinja的`trim_blocks`和`lstrip_blocks`标签。
这里，请注意空格的使用。我们强烈建议您仔细检查模板是否打印了多余的空格！
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

如果你之前不了解[Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/)。
Jinja是一种模板语言，允许你编写简单的代码来生成文本。
在许多方面，代码和语法类似于Python。在纯Python中，这个模板看起来会像这样：
```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

这里使用Jinja模板处理如下三步：
1. 对于每条消息，如果消息是用户消息，则在其前面添加一个空格，否则不打印任何内容
2. 添加消息内容
3. 如果消息不是最后一条，请在其后添加两个空格。在最后一条消息之后，打印`EOS`。

这是一个简单的模板，它不添加任何控制tokens，也不支持`system`消息（常用于指导模型在后续对话中如何表现）。
但 Jinja 给了你很大的灵活性来做这些事情！让我们看一个 Jinja 模板，
它可以实现类似于LLaMA的prompt输入（请注意，真正的LLaMA模板包括`system`消息，请不要在实际代码中使用这个简单模板！）
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

这里稍微看一下，就能明白这个模板的作用：它根据每条消息的“角色”添加对应的消息。
`user`、`assistant`、`system`的消息需要分别处理，因为它们代表不同的角色输入。

## 高级：编辑聊天模板

### 如何创建聊天模板？

很简单，你只需编写一个jinja模板并设置`tokenizer.chat_template`。你也可以从一个现有模板开始，只需要简单编辑便可以！
例如，我们可以采用上面的LLaMA模板，并在助手消息中添加"[ASST]"和"[/ASST]"：
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

现在，只需设置`tokenizer.chat_template`属性。下次使用[`~PreTrainedTokenizer.apply_chat_template`]时，它将使用您的新模板！
此属性将保存在`tokenizer_config.json`文件中，因此您可以使用[`~utils.PushToHubMixin.push_to_hub`]将新模板上传到 Hub，
这样每个人都可以使用你模型的模板！

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

由于[`~PreTrainedTokenizer.apply_chat_template`]方法是由[`ConversationalPipeline`]类调用，
因此一旦你设置了聊天模板，您的模型将自动与[`ConversationalPipeline`]兼容。
### “默认”模板是什么？

在引入聊天模板（chat_template）之前，聊天prompt是在模型中通过硬编码处理的。为了向前兼容，我们保留了这种硬编码处理聊天prompt的方法。
如果一个模型没有设置聊天模板，但其模型有默认模板，`ConversationalPipeline`类和`apply_chat_template`等方法将使用该模型的聊天模板。
您可以通过检查`tokenizer.default_chat_template`属性来查找`tokenizer`的默认模板。

这是我们纯粹为了向前兼容性而做的事情，以避免破坏任何现有的工作流程。即使默认的聊天模板适用于您的模型，
我们强烈建议通过显式设置`chat_template`属性来覆盖默认模板，以便向用户清楚地表明您的模型已经正确的配置了聊天模板，
并且为了未来防范默认模板被修改或弃用的情况。
### 我应该使用哪个模板？

在为已经训练过的聊天模型设置模板时，您应确保模板与模型在训练期间看到的消息格式完全匹配，否则可能会导致性能下降。
即使您继续对模型进行训练，也应保持聊天模板不变，这样可能会获得最佳性能。
这与`tokenization`非常类似，在推断时，你选用跟训练时一样的`tokenization`，通常会获得最佳性能。

如果您从头开始训练模型，或者在微调基础语言模型进行聊天时，您有很大的自由选择适当的模板！
LLMs足够聪明，可以学会处理许多不同的输入格式。我们为没有特定类别模板的模型提供一个默认模板，该模板遵循
`ChatML` format格式要求，对于许多用例来说，
这是一个很好的、灵活的选择。

默认模板看起来像这样：

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```


如果您喜欢这个模板，下面是一行代码的模板形式，它可以直接复制到您的代码中。这一行代码还包括了[generation prompts](#什么是"generation prompts"?)，
但请注意它不会添加`BOS`或`EOS`token。
如果您的模型需要这些token，它们不会被`apply_chat_template`自动添加，换句话说，文本的默认处理参数是`add_special_tokens=False`。
这是为了避免模板和`add_special_tokens`逻辑产生冲突，如果您的模型需要特殊tokens，请确保将它们添加到模板中！

```
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

该模板将每条消息包装在`<|im_start|>`和`<|im_end|>`tokens里面，并将角色简单地写为字符串，这样可以灵活地训练角色。输出如下：
```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

`user`，`system`和`assistant`是对话助手模型的标准角色，如果您的模型要与[`ConversationalPipeline`]兼容，我们建议你使用这些角色。
但您可以不局限于这些角色，模板非常灵活，任何字符串都可以成为角色。

### 如何添加聊天模板？

如果您有任何聊天模型，您应该设置它们的`tokenizer.chat_template`属性，并使用[`~PreTrainedTokenizer.apply_chat_template`]测试，
然后将更新后的`tokenizer`推送到 Hub。
即使您不是模型所有者，如果您正在使用一个空的聊天模板或者仍在使用默认的聊天模板，
请发起一个[pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)，以便正确设置该属性！

一旦属性设置完成，就完成了！`tokenizer.apply_chat_template`现在将在该模型中正常工作，
这意味着它也会自动支持在诸如`ConversationalPipeline`的地方！

通过确保模型具有这一属性，我们可以确保整个社区都能充分利用开源模型的全部功能。
格式不匹配已经困扰这个领域并悄悄地损害了性能太久了，是时候结束它们了！


## 高级：模板写作技巧

如果你对Jinja不熟悉，我们通常发现编写聊天模板的最简单方法是先编写一个简短的Python脚本，按照你想要的方式格式化消息，然后将该脚本转换为模板。

请记住，模板处理程序将接收对话历史作为名为`messages`的变量。每条`message`都是一个带有两个键`role`和`content`的字典。
您可以在模板中像在Python中一样访问`messages`，这意味着您可以使用`{% for message in messages %}`进行循环，
或者例如使用`{{ messages[0] }}`访问单个消息。

您也可以使用以下提示将您的代码转换为Jinja：
### For循环

在Jinja中，for循环看起来像这样：

```
{% for message in messages %}
{{ message['content'] }}
{% endfor %}
```

请注意，`{{ expression block }}`中的内容将被打印到输出。您可以在表达式块中使用像`+`这样的运算符来组合字符串。
### If语句

Jinja中的if语句如下所示：

```
{% if message['role'] == 'user' %}
{{ message['content'] }}
{% endif %}
```
注意Jinja使用`{% endfor %}`和`{% endif %}`来表示`for`和`if`的结束。

### 特殊变量

在您的模板中，您将可以访问`messages`列表，但您还可以访问其他几个特殊变量。
这些包括特殊`token`，如`bos_token`和`eos_token`，以及我们上面讨论过的`add_generation_prompt`变量。
您还可以使用`loop`变量来访问有关当前循环迭代的信息，例如使用`{% if loop.last %}`来检查当前消息是否是对话中的最后一条消息。

以下是一个示例，如果`add_generation_prompt=True`需要在对话结束时添加`generate_prompt`：


```
{% if loop.last and add_generation_prompt %}
{{ bos_token + 'Assistant:\n' }}
{% endif %}
```

### 空格的注意事项

我们已经尽可能尝试让Jinja忽略除`{{ expressions }}`之外的空格。
然而，请注意Jinja是一个通用的模板引擎，它可能会将同一行文本块之间的空格视为重要，并将其打印到输出中。
我们**强烈**建议在上传模板之前检查一下，确保模板没有在不应该的地方打印额外的空格！
