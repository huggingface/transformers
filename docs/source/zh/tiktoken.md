<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
``
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Transformers与Tiktonken的互操作性

在🤗 transformers中，当使用`from_pretrained`方法从Hub加载模型时，如果模型包含tiktoken格式的`tokenizer.model`文件，框架可以无缝支持tiktoken模型文件，并自动将其转换为我们的[快速词符化器](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)。

### 已知包含`tiktoken.model`文件发布的模型：
    - gpt2
    - llama3

## 使用示例

为了在transformers中正确加载`tiktoken`文件，请确保`tiktoken.model`文件是tiktoken格式的，并且会在加载`from_pretrained`时自动加载。以下展示如何从同一个文件中加载词符化器(tokenizer)和模型：

```py
from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original")
```
## 创建tiktoken词符化器(tokenizer)

`tokenizer.model`文件中不包含任何额外的词符(token)或模式字符串(pattern strings)的信息。如果这些信息很重要，需要将词符化器(tokenizer)转换为适用于[`PreTrainedTokenizerFast`]类的`tokenizer.json`格式。

使用[tiktoken.get_encoding](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken/registry.py#L63)生成`tokenizer.model`文件，再使用[`convert_tiktoken_to_fast`]函数将其转换为`tokenizer.json`文件。

```py

from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# You can load your custom encoding or the one provided by OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")
```

生成的`tokenizer.json`文件将被保存到指定的目录，并且可以通过[`PreTrainedTokenizerFast`]类来加载。

```py
tokenizer = PreTrainedTokenizerFast.from_pretrained("config/save/dir")
```
