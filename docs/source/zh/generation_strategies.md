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

# 生成策略

解码策略决定了模型应该如何选择下一个生成的token。有许多类型的解码策略，选择合适的策略对生成文本的质量有显著影响。

本指南将帮助你理解 Transformers 中可用的不同解码策略，以及如何和何时使用它们。

## 基础解码方法

这些是成熟的解码方法，应该作为文本生成任务的起点。

### 贪婪搜索

贪婪搜索是默认的解码策略。它在每一步选择最可能的下一个token。除非在 [`GenerationConfig`] 中指定，否则此策略最多生成20个新token。

贪婪搜索适用于输出相对较短且不需要创造性的任务。然而，当生成较长的序列时，它会开始重复自己，效果会变差。

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# 显式设置为默认长度，因为Llama2的生成长度是4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

### 采样

采样，或多项式采样，根据模型整个词汇表上的概率分布随机选择一个token（而不是像贪婪搜索那样选择最可能的token）。这意味着每个具有非零概率的token都有机会被选中。采样策略可以减少重复，并生成更有创意和多样性的输出。

通过设置 `do_sample=True` 和 `num_beams=1` 来启用多项式采样。

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# 显式设置为100，因为Llama2的生成长度是4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

### 束搜索

束搜索在每个时间步保持跟踪多个生成的序列（称为"束"）。在一定数量的步骤后，它选择*整体*概率最高的序列。与贪婪搜索不同，这种策略可以"向前看"，即使初始token的概率较低，也能选择整体概率更高的序列。它最适合基于输入的任务，如描述图像或语音识别。你也可以在束搜索中使用 `do_sample=True` 在每一步进行采样，但束搜索仍会在步骤之间贪婪地剪除低概率序列。

> [!TIP]
> 查看 [束搜索可视化工具](https://huggingface.co/spaces/m-ric/beam_search_visualizer) 来了解束搜索的工作原理。

通过 `num_beams` 参数启用束搜索（应大于1，否则等同于贪婪搜索）。

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# 显式设置为100，因为Llama2的生成长度是4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

## 自定义生成方法

自定义生成方法可以实现特殊的行为，例如：

- 如果模型不确定，让它继续思考；
- 如果模型卡住了，回滚生成；
- 使用自定义逻辑处理特殊token；
- 使用专门的 KV 缓存；

我们通过模型仓库启用自定义生成方法，假设有特定的模型标签和文件结构（见下面的小节）。此功能是 [自定义建模代码](./models.md#custom-models) 的扩展，因此同样需要设置 `trust_remote_code=True`。

如果模型仓库包含自定义生成方法，最简单的尝试方法是加载模型并用它生成：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

# `transformers-community/custom_generate_example` 包含 `Qwen/Qwen2.5-0.5B-Instruct` 的副本，但
# 带有自定义生成代码 -> 调用 `generate` 会使用自定义生成方法！
tokenizer = AutoTokenizer.from_pretrained("transformers-community/custom_generate_example")
model = AutoModelForCausalLM.from_pretrained(
    "transformers-community/custom_generate_example", device_map="auto", trust_remote_code=True
)

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# 自定义生成方法是一个最小的贪婪解码实现。它还会在运行时打印自定义消息。
gen_out = model.generate(**inputs)
# 你现在应该能看到它的自定义消息，"✨ using a custom generation method ✨"
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

具有自定义生成方法的模型仓库有一个特殊属性：它们的生成方法可以通过 [`~GenerationMixin.generate`] 的 `custom_generate` 参数从**任何**模型加载。这意味着任何人都可以创建和共享他们的自定义生成方法，使其能与任何 Transformers 模型一起工作，而无需用户安装额外的 Python 包。

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# `custom_generate` 用 `transformers-community/custom_generate_example` 中定义的
# 自定义生成方法替换原始的 `generate`
gen_out = model.generate(**inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

你应该阅读包含自定义生成策略的仓库的 `README.md` 文件，以了解新参数和输出类型的差异（如果有的话）。否则，你可以假设它的工作方式与基础 [`~GenerationMixin.generate`] 方法相同。

> [!TIP]
> 你可以通过 [搜索自定义标签](https://huggingface.co/models?other=custom_generate) `custom_generate` 来找到所有自定义生成方法。

以 Hub 仓库 [transformers-community/custom_generate_example](https://huggingface.co/transformers-community/custom_generate_example) 为例。`README.md` 说明它有一个额外的输入参数 `left_padding`，可以在提示词前添加若干个填充token。

```py
gen_out = model.generate(
    **inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True, left_padding=5
)
print(tokenizer.batch_decode(gen_out)[0])
'<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>The quick brown fox jumps over the lazy dog.\n\nThe sentence "The quick'
```

如果自定义方法有你的环境不满足的固定 Python 依赖要求，你会收到关于缺少依赖的异常。例如，[transformers-community/custom_generate_bad_requirements](https://huggingface.co/transformers-community/custom_generate_bad_requirements) 在其 `custom_generate/requirements.txt` 文件中定义了一组不可能的依赖要求，如果你尝试运行它，会看到以下错误消息。

```text
ImportError: Missing requirements in your local environment for `transformers-community/custom_generate_bad_requirements`:
foo (installed: None)
bar==0.0.0 (installed: None)
torch>=99.0 (installed: 2.6.0)
```

相应地更新你的 Python 依赖将消除此错误消息。

### 创建自定义生成方法

要创建新的生成方法，你需要创建一个新的 [**模型**](https://huggingface.co/new) 仓库并向其中推送一些文件。

1. 你为其设计生成方法的模型。
2. `custom_generate/generate.py`，包含你自定义生成方法的所有逻辑。
3. `custom_generate/requirements.txt`，用于可选地添加新的 Python 依赖和/或锁定特定版本以正确使用你的方法。
4. `README.md`，你应该在这里添加 `custom_generate` 标签，并记录你自定义方法的任何新参数或输出类型差异。

添加所有必需文件后，你的仓库应该如下所示：

```text
your_repo/
├── README.md          # 包含 'custom_generate' 标签
├── config.json
├── ...
└── custom_generate/
    ├── generate.py
    └── requirements.txt
```

#### 添加基础模型

你自定义生成方法的起点是一个与其他任何模型仓库一样的模型仓库。要添加到此仓库的模型应该是你为其设计方法的模型，它旨在成为一个可工作的自包含模型-生成对的一部分。当加载此仓库中的模型时，你的自定义生成方法将覆盖 `generate`。不用担心——如上一节所述，你的生成方法仍然可以与任何其他 Transformers 模型一起加载。

如果你只是想复制现有模型，可以这样做：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("source/model_repo")
model = AutoModelForCausalLM.from_pretrained("source/model_repo")
tokenizer.save_pretrained("your/generation_method", push_to_hub=True)
model.save_pretrained("your/generation_method", push_to_hub=True)
```

#### generate.py

这是你生成方法的核心。它*必须*包含一个名为 `generate` 的方法，且该方法*必须*以 `model` 参数作为其第一个参数。`model` 是模型实例，这意味着你可以访问模型中的所有属性和方法，包括 [`GenerationMixin`] 中定义的方法（如基础 `generate` 方法）。

> [!WARNING]
> `generate.py` 必须放在名为 `custom_generate` 的文件夹中，而不是仓库的根目录。此功能的文件路径是硬编码的。

在底层，当基础 [`~GenerationMixin.generate`] 方法被调用并带有 `custom_generate` 参数时，它首先检查其 Python 依赖要求（如果有），然后在 `generate.py` 中定位自定义 `generate` 方法，最后调用自定义 `generate`。所有接收到的参数和 `model` 都会转发到你的自定义 `generate` 方法，但用于触发自定义生成的参数（`trust_remote_code` 和 `custom_generate`）除外。

这意味着你的 `generate` 可以混合使用原始参数和自定义参数（以及不同的输出类型），如下所示。

```py
import torch

def generate(model, input_ids, generation_config=None, left_padding=None, **kwargs):
    generation_config = generation_config or model.generation_config  # 默认使用模型生成配置
    cur_length = input_ids.shape[1]
    max_length = generation_config.max_length or cur_length + generation_config.max_new_tokens

    # 自定义参数示例：在提示词前添加 `left_padding`（整数）个填充token
    if left_padding is not None:
        if not isinstance(left_padding, int) or left_padding < 0:
            raise ValueError(f"left_padding 必须是大于0的整数，但得到的是 {left_padding}")

        pad_token = kwargs.pop("pad_token", None) or generation_config.pad_token_id or model.config.pad_token_id
        if pad_token is None:
            raise ValueError("pad_token 未定义")
        batch_size = input_ids.shape[0]
        pad_tensor = torch.full(size=(batch_size, left_padding), fill_value=pad_token).to(input_ids.device)
        input_ids = torch.cat((pad_tensor, input_ids), dim=1)
        cur_length = input_ids.shape[1]

    # 简单的贪婪解码循环
    while cur_length < max_length:
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
        cur_length += 1

    return input_ids
```

遵循以下推荐做法以确保你的自定义生成方法按预期工作。

- 随意重用原始 [`~GenerationMixin.generate`] 中的验证和输入准备逻辑。
- 如果在 `model` 中使用任何私有方法/属性，请在 requirements 中固定 `transformers` 版本。
- 考虑添加模型验证、输入验证，甚至单独的测试文件，以帮助用户在其环境中对你的代码进行健全性检查。

你的自定义 `generate` 方法可以从 `custom_generate` 文件夹相对导入代码。例如，如果你有一个 `utils.py` 文件，可以这样导入：

```py
from .utils import some_function
```

仅支持从同级 `custom_generate` 文件夹的相对导入。父级/兄弟文件夹导入无效。`custom_generate` 参数也可以在本地与包含 `custom_generate` 结构的任何目录一起使用。这是开发自定义生成方法的推荐工作流程。

#### requirements.txt

你可以在 `custom_generate` 文件夹内的 `requirements.txt` 文件中可选地指定额外的 Python 依赖要求。这些会在运行时检查，如果缺少会抛出异常，提示用户相应地更新其环境。

#### README.md

模型仓库根目录的 `README.md` 通常描述其中的模型。然而，由于仓库的重点是自定义生成方法，我们强烈建议将其重点转向描述自定义生成方法。除了方法的描述外，我们建议记录与原始 [`~GenerationMixin.generate`] 的任何输入和/或输出差异。这样，用户可以专注于新内容，并依赖 Transformers 文档了解通用实现细节。

为了便于发现，我们强烈建议你为仓库添加 `custom_generate` 标签。为此，你的 `README.md` 文件顶部应如下例所示。推送文件后，你应该能在仓库中看到该标签！

```text
---
library_name: transformers
tags:
  - custom_generate
---

(你的 markdown 内容在这里)
```

推荐做法：

- 记录 [`~GenerationMixin.generate`] 的输入和输出差异。
- 添加自包含的示例以便快速实验。
- 描述软性要求，例如该方法是否仅适用于特定模型系列。

### 重用 `generate` 的输入准备

如果你正在添加新的解码循环，你可能希望保留 `generate` 中的输入准备（批次扩展、注意力掩码、logits 处理器、停止条件等）。你也可以传递一个**可调用对象**给 `custom_generate`，以重用 [`~GenerationMixin.generate`] 的完整准备流程，同时仅覆盖解码循环。

```py
def custom_loop(model, input_ids, attention_mask, logits_processor, stopping_criteria, generation_config, **model_kwargs):
    next_tokens = input_ids
    while input_ids.shape[1] < stopping_criteria[0].max_length:
        logits = model(next_tokens, attention_mask=attention_mask, **model_kwargs).logits
        next_token_logits = logits_processor(input_ids, logits[:, -1, :])
        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)
    return input_ids

output = model.generate(
    **inputs,
    custom_generate=custom_loop,
    max_new_tokens=10,
)
```

> [!TIP]
> 如果你发布 `custom_generate` 仓库，你的 `generate` 实现本身可以定义一个可调用对象并将其传递给 `model.generate()`。这让你可以自定义解码循环，同时仍然受益于 Transformers 内置的输入准备逻辑。

### 查找自定义生成方法

你可以通过 [搜索自定义标签](https://huggingface.co/models?other=custom_generate) `custom_generate` 来找到所有自定义生成方法。除了标签外，我们还策划了两个 `custom_generate` 方法集合：

- [自定义生成方法 - 社区](https://huggingface.co/collections/transformers-community/custom-generation-methods-community-6888fb1da0efbc592d3a8ab6) -- 社区贡献的强大方法集合；
- [自定义生成方法 - 教程](https://huggingface.co/collections/transformers-community/custom-generation-methods-tutorials-6823589657a94940ea02cfec) -- 以前是 `transformers` 一部分的方法的参考实现集合，以及 `custom_generate` 的教程。

## 资源

阅读 [如何生成文本：使用不同的解码方法进行 Transformers 语言生成](https://huggingface.co/blog/how-to-generate) 博客文章，了解常见解码策略的工作原理解释。
