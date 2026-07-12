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

# 使用 🤗 PEFT 加载adapters

[[open-in-colab]]

[参数高效微调（PEFT）方法](https://huggingface.co/blog/peft)在微调过程中冻结预训练模型的参数，并在其顶部添加少量可训练参数（adapters）。adapters被训练以学习特定任务的信息。这种方法已被证明非常节省内存，同时具有较低的计算使用量，同时产生与完全微调模型相当的结果。

使用PEFT训练的adapters通常比完整模型小一个数量级，使其方便共享、存储和加载。

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">与完整尺寸的模型权重（约为700MB）相比，存储在Hub上的OPTForCausalLM模型的adapter权重仅为~6MB。</figcaption>
</div>

如果您对学习更多关于🤗 PEFT库感兴趣，请查看[文档](https://huggingface.co/docs/peft/index)。


## 设置

首先安装 🤗 PEFT：

```bash
pip install peft
```

如果你想尝试全新的特性，你可能会有兴趣从源代码安装这个库：

```bash
pip install git+https://github.com/huggingface/peft.git
```
## 支持的 PEFT 模型

Transformers原生支持一些PEFT方法，这意味着你可以加载本地存储或在Hub上的adapter权重，并使用几行代码轻松运行或训练它们。以下是受支持的方法：

- [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://huggingface.co/papers/2303.10512)

如果你想使用其他PEFT方法，例如提示学习或提示微调，或者关于通用的 🤗 PEFT库，请参阅[文档](https://huggingface.co/docs/peft/index)。

## 加载 PEFT adapter

要从huggingface的Transformers库中加载并使用PEFTadapter模型，请确保Hub仓库或本地目录包含一个`adapter_config.json`文件和adapter权重，如上例所示。然后，您可以使用`AutoModelFor`类加载PEFT adapter模型。例如，要为因果语言建模加载一个PEFT adapter模型：

1. 指定PEFT模型id
2. 将其传递给[`AutoModelForCausalLM`]类

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

你可以使用`AutoModelFor`类或基础模型类（如`OPTForCausalLM`或`LlamaForCausalLM`）来加载一个PEFT adapter。


</Tip>

您也可以通过`load_adapter`方法来加载 PEFT adapter。

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## 基于8bit或4bit进行加载

`bitsandbytes`集成支持8bit和4bit精度数据类型，这对于加载大模型非常有用，因为它可以节省内存（请参阅`bitsandbytes`[指南](./quantization#bitsandbytes-integration)以了解更多信息）。要有效地将模型分配到您的硬件，请在[`~PreTrainedModel.from_pretrained`]中添加`load_in_8bit`或`load_in_4bit`参数，并将`device_map="auto"`设置为：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

## 添加新的adapter

你可以使用[`~peft.PeftModel.add_adapter`]方法为一个已有adapter的模型添加一个新的adapter，只要新adapter的类型与当前adapter相同即可。例如，如果你有一个附加到模型上的LoRA adapter：

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```


添加一个新的adapter：

```py
# attach new adapter with same config
model.add_adapter(lora_config, adapter_name="adapter_2")
```
现在您可以使用[`~peft.PeftModel.set_adapter`]来设置要使用的adapter。

```py
# use adapter_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# use adapter_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## 启用和禁用adapters
一旦您将adapter添加到模型中，您可以启用或禁用adapter模块。要启用adapter模块：


```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# to initiate with random weights
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```
要禁用adapter模块：

```py
model.disable_adapters()
output = model.generate(**inputs)
```
## 训练一个 PEFT adapter

PEFT适配器受[`Trainer`]类支持，因此您可以为您的特定用例训练适配器。它只需要添加几行代码即可。例如，要训练一个LoRA adapter：


<Tip>

如果你不熟悉如何使用[`Trainer`]微调模型，请查看[微调预训练模型](training)教程。

</Tip>

1. 使用任务类型和超参数定义adapter配置（参见[`~peft.LoraConfig`]以了解超参数的详细信息）。

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2. 将adapter添加到模型中。

```py
model.add_adapter(peft_config)
```

3. 现在可以将模型传递给[`Trainer`]了！

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

要保存训练好的adapter并重新加载它：

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer
-   8-bit / 4-bit examples ?
-->
