<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# GGUF 和 Transformers 的交互

GGUF文件格式用于存储模型，以便通过[GGML](https://github.com/ggerganov/ggml)和其他依赖它的库进行推理，例如非常流行的[llama.cpp](https://github.com/ggerganov/llama.cpp)或[whisper.cpp](https://github.com/ggerganov/whisper.cpp)。

该文件格式[由抱抱脸支持](https://huggingface.co/docs/hub/en/gguf)，可用于快速检查文件中张量和元数据。

该文件格式是一种“单文件格式”，通常单个文件就包含了配置属性、分词器词汇表和其他属性，同时还有模型中要加载的所有张量。这些文件根据文件的量化类型有不同的格式。我们在[这里](https://huggingface.co/docs/hub/en/gguf#quantization-types)进行了简要介绍。

## 在 Transformers 中的支持

我们在 transformers 中添加了加载 gguf 文件的功能，这样可以对 GGUF 模型进行进一步的训练或微调，然后再将模型转换回 GGUF 格式，以便在 ggml 生态系统中使用。加载模型时，我们首先将其反量化为 FP32，然后再加载权重以在 PyTorch 中使用。

>    [!注意]
>    目前这个功能还处于探索阶段，欢迎大家贡献力量，以便在不同量化类型和模型架构之间更好地完善这一功能。

目前，支持的模型架构和量化类型如下：

### 支持的量化类型

根据分享在 Hub 上的较为热门的量化文件，初步支持以下量化类型：

- F32
- F16
- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q5_K
- Q6_K
- IQ1_S
- IQ1_M
- IQ2_XXS
- IQ2_XS
- IQ2_S
- IQ3_XXS
- IQ3_S
- IQ4_XS
- IQ4_NL

>    [!注意]
>    为了支持 gguf 反量化，需要安装 `gguf>=0.10.0`。

### 支持的模型架构

目前支持以下在 Hub 上非常热门的模型架构：

- LLaMa
- Mistral
- Qwen2
- Qwen2Moe
- Phi3
- Bloom
- Falcon
- StableLM
- GPT2
- Starcoder2

## 使用示例

为了在`transformers`中加载`gguf`文件，你需要在 `from_pretrained`方法中为分词器和模型指定 `gguf_file`参数。下面是从同一个文件中加载分词器和模型的示例：

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
```

现在，你就已经可以结合 PyTorch 生态系统中的一系列其他工具，来使用完整的、未量化的模型了。

为了将模型转换回`gguf`文件，我们建议使用`llama.cpp`中的[`convert-hf-to-gguf.py`文件](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py)。

以下是如何补充上面的脚本，以保存模型并将其导出回 `gguf`的示例：

```py
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```