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


## 使用LLMs进行生成

[[open-in-colab]]

LLMs，即大语言模型，是文本生成背后的关键组成部分。简单来说，它们包含经过大规模预训练的transformer模型，用于根据给定的输入文本预测下一个词（或更准确地说，下一个`token`）。由于它们一次只预测一个`token`，因此除了调用模型之外，您需要执行更复杂的操作来生成新的句子——您需要进行自回归生成。

自回归生成是在给定一些初始输入，通过迭代调用模型及其自身的生成输出来生成文本的推理过程，。在🤗 Transformers中，这由[`~generation.GenerationMixin.generate`]方法处理，所有具有生成能力的模型都可以使用该方法。

本教程将向您展示如何：

* 使用LLM生成文本
* 避免常见的陷阱
* 帮助您充分利用LLM下一步指导

在开始之前，请确保已安装所有必要的库：


```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## 生成文本

一个用于[因果语言建模](tasks/language_modeling)训练的语言模型，将文本`tokens`序列作为输入，并返回下一个`token`的概率分布。


<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"LLM的前向传递"</figcaption>
</figure>

使用LLM进行自回归生成的一个关键方面是如何从这个概率分布中选择下一个`token`。这个步骤可以随意进行，只要最终得到下一个迭代的`token`。这意味着可以简单的从概率分布中选择最可能的`token`，也可以复杂的在对结果分布进行采样之前应用多种变换，这取决于你的需求。

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"自回归生成迭代地从概率分布中选择下一个token以生成文本"</figcaption>
</figure>

上述过程是迭代重复的，直到达到某个停止条件。理想情况下，停止条件由模型决定，该模型应学会在何时输出一个结束序列（`EOS`）标记。如果不是这种情况，生成将在达到某个预定义的最大长度时停止。

正确设置`token`选择步骤和停止条件对于让你的模型按照预期的方式执行任务至关重要。这就是为什么我们为每个模型都有一个[~generation.GenerationConfig]文件，它包含一个效果不错的默认生成参数配置，并与您模型一起加载。

让我们谈谈代码！

<Tip>

如果您对基本的LLM使用感兴趣，我们高级的[`Pipeline`](pipeline_tutorial)接口是一个很好的起点。然而，LLMs通常需要像`quantization`和`token选择步骤的精细控制`等高级功能，这最好通过[`~generation.GenerationMixin.generate`]来完成。使用LLM进行自回归生成也是资源密集型的操作，应该在GPU上执行以获得足够的吞吐量。

</Tip>

首先，您需要加载模型。

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

您将会注意到在`from_pretrained`调用中的两个标志：

- `device_map`确保模型被移动到您的GPU(s)上
- `load_in_4bit`应用[4位动态量化](main_classes/quantization)来极大地减少资源需求

还有其他方式来初始化一个模型，但这是一个开始使用LLM很好的起点。

接下来，你需要使用一个[tokenizer](tokenizer_summary)来预处理你的文本输入。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

`model_inputs`变量保存着分词后的文本输入以及注意力掩码。尽管[`~generation.GenerationMixin.generate`]在未传递注意力掩码时会尽其所能推断出注意力掩码，但建议尽可能传递它以获得最佳结果。

在对输入进行分词后，可以调用[`~generation.GenerationMixin.generate`]方法来返回生成的`tokens`。生成的`tokens`应该在打印之前转换为文本。

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, orange, purple, pink,'
```

最后，您不需要一次处理一个序列！您可以批量输入，这将在小延迟和低内存成本下显著提高吞吐量。您只需要确保正确地填充您的输入（详见下文）。

```py
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model_inputs = tokenizer(
...     ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['A list of colors: red, blue, green, yellow, orange, purple, pink,',
'Portugal is a country in southwestern Europe, on the Iber']
```

就是这样！在几行代码中，您就可以利用LLM的强大功能。


## 常见陷阱

有许多[生成策略](generation_strategies)，有时默认值可能不适合您的用例。如果您的输出与您期望的结果不匹配，我们已经创建了一个最常见的陷阱列表以及如何避免它们。

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

### 生成的输出太短/太长

如果在[`~generation.GenerationConfig`]文件中没有指定，`generate`默认返回20个tokens。我们强烈建议在您的`generate`调用中手动设置`max_new_tokens`以控制它可以返回的最大新tokens数量。请注意，LLMs（更准确地说，仅[解码器模型](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)）也将输入提示作为输出的一部分返回。

```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### 错误的生成模式

默认情况下，除非在[`~generation.GenerationConfig`]文件中指定，否则`generate`会在每个迭代中选择最可能的token（贪婪解码）。对于您的任务，这可能是不理想的；像聊天机器人或写作文章这样的创造性任务受益于采样。另一方面，像音频转录或翻译这样的基于输入的任务受益于贪婪解码。通过将`do_sample=True`启用采样，您可以在这篇[博客文章](https://huggingface.co/blog/how-to-generate)中了解更多关于这个话题的信息。

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(42)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.  Specifically, I am an indoor-only cat.  I'
```

### 错误的填充位置

LLMs是[仅解码器](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)架构，意味着它们会持续迭代您的输入提示。如果您的输入长度不相同，则需要对它们进行填充。由于LLMs没有接受过从`pad tokens`继续训练，因此您的输入需要左填充。确保在生成时不要忘记传递注意力掩码！

```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails to capture the logic.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

### 错误的提示

一些模型和任务期望某种输入提示格式才能正常工作。当未应用此格式时，您将获得悄然的性能下降：模型能工作，但不如预期提示那样好。有关提示的更多信息，包括哪些模型和任务需要小心，可在[指南](tasks/prompting)中找到。让我们看一个使用[聊天模板](chat_templating)的聊天LLM示例：

```python
>>> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
>>> model = AutoModelForCausalLM.from_pretrained(
...     "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", load_in_4bit=True
... )
>>> set_seed(0)
>>> prompt = """How many helicopters can a human eat in one sitting? Reply as a thug."""
>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> input_length = model_inputs.input_ids.shape[1]
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=20)
>>> print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"I'm not a thug, but i can tell you that a human cannot eat"
>>> # Oh no, it did not follow our instruction to reply as a thug! Let's see what happens when we write
>>> # a better prompt and use the right template for this model (through `tokenizer.apply_chat_template`)

>>> set_seed(0)
>>> messages = [
...     {
...         "role": "system",
...         "content": "You are a friendly chatbot who always responds in the style of a thug",
...     },
...     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
... ]
>>> model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
>>> input_length = model_inputs.shape[1]
>>> generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=20)
>>> print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
'None, you thug. How bout you try to focus on more useful questions?'
>>> # As we can see, it followed a proper thug style 😎
```

## 更多资源

虽然自回归生成过程相对简单，但要充分利用LLM可能是一个具有挑战性的任务，因为很多组件复杂且密切关联。以下是帮助您深入了解LLM使用和理解的下一步：

### 高级生成用法

1. [指南](generation_strategies)，介绍如何控制不同的生成方法、如何设置生成配置文件以及如何进行输出流式传输；
2. [指南](chat_templating)，介绍聊天LLMs的提示模板；
3. [指南](tasks/prompting)，介绍如何充分利用提示设计；
4. API参考文档，包括[`~generation.GenerationConfig`]、[`~generation.GenerationMixin.generate`]和[与生成相关的类](internal/generation_utils)。

### LLM排行榜

1. [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), 侧重于开源模型的质量;
2. [Open LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard), 侧重于LLM的吞吐量.

### 延迟、吞吐量和内存利用率

1. [指南](llm_tutorial_optimization),如何优化LLMs以提高速度和内存利用；
2. [指南](main_classes/quantization), 关于`quantization`，如bitsandbytes和autogptq的指南，教您如何大幅降低内存需求。

### 相关库

1. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference), 一个面向生产的LLM服务器；
2. [`optimum`](https://github.com/huggingface/optimum), 一个🤗 Transformers的扩展，优化特定硬件设备的性能
