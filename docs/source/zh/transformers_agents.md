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

# Transformers Agents

<Tip warning={true}>

`Transformers Agents`是一个实验性的随时可能发生变化的API。由于API或底层模型可能发生变化，`agents`返回的结果也会有所不同。

</Tip>

Transformers版本`v4.29.0`基于`tools`和`agents`概念构建。您可以在[此Colab链接](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj)中进行测试。

简而言之，它在`Transformers`之上提供了一个自然语言API：我们定义了一组经过筛选的`tools`，并设计了一个`agents`来解读自然语言并使用这些工具。它具有很强的可扩展性；我们筛选了一些相关的`tools`，但我们将向您展示如何通过社区开发的`tool`轻松地扩展系统。

让我们从一些可以通过这个新API实现的示例开始。在处理多模态任务时它尤其强大，因此让我们快速试着生成图像并大声朗读文本。


```py
agent.run("Caption the following image", image=image)
```

| **输入**                                                                                                                      | **输出**                            |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |

---

```py
agent.run("Read the following text out loud", text=text)
```
| **输入**                            | **输出**                                                                                                                                                                                                               |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio> 

---

```py
agent.run(
    "In the following `document`, where will the TRRF Scientific Advisory Council Meeting take place?",
    document=document,
)
```
| **输入**                                                                                                                   | **输出**     |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | ballroom foyer |

## 快速入门

要使用 `agent.run`，您需要实例化一个`agent`，它是一个大型语言模型（LLM）。我们支持OpenAI模型以及来自BigCode和OpenAssistant的开源替代方案。OpenAI模型性能更好（但需要您拥有OpenAI API密钥，因此无法免费使用），Hugging Face为BigCode和OpenAssistant模型提供了免费访问端点。

一开始请安装`agents`附加模块，以安装所有默认依赖项。

```bash
pip install transformers[agents]
```

要使用OpenAI模型，您可以在安装`openai`依赖项后实例化一个`OpenAiAgent`：

```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

要使用BigCode或OpenAssistant，请首先登录以访问Inference API：

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

然后，实例化`agent`：

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

此示例使用了目前Hugging Face免费提供的推理API。如果你有自己的推理端点用于此模型（或其他模型），你可以用你的URL替换上面的URL。

<Tip>

StarCoder和OpenAssistant可以免费使用，并且在简单任务上表现出色。然而，当处理更复杂的提示时就不再有效。如果你遇到这样的问题，我们建议尝试使用OpenAI模型，尽管遗憾的是它不是开源的，但它在目前情况下表现更好。

</Tip>

现在，您已经可以开始使用了！让我们深入了解您现在可以使用的两个API。

### 单次执行(run)

单次执行方法是使用`agent`的 `~Agent.run`：

```py
agent.run("Draw me a picture of rivers and lakes.")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>

它会自动选择适合您要执行的任务的`tool`（或`tools`），并以适当的方式运行它们。它可以在同一指令中执行一个或多个任务（尽管您的指令越复杂，`agent`失败的可能性就越大）。


```py
agent.run("Draw me a picture of the sea then transform the picture to add an island")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>

每个 [`~Agent.run`] 操作都是独立的，因此您可以多次连续运行 [`~Agent.run`]并执行不同的任务。

请注意，您的 `agent` 只是一个大型语言模型，因此您略有变化的提示可能会产生完全不同的结果。重要的是尽可能清晰地解释您要执行的任务。我们在[这里](../en/custom_tools#writing-good-user-inputs)更深入地讨论了如何编写良好的提示。

如果您想在多次执行之间保持同一状态或向`agent`传递非文本对象，可以通过指定`agent`要使用的变量来实现。例如，您可以生成有关河流和湖泊的第一幅图像，并要求模型通过执行以下操作向该图片添加一个岛屿：

```python
picture = agent.run("Generate a picture of rivers and lakes.")
updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
```

<Tip>

当模型无法理解您的请求和库中的工具时，这可能会有所帮助。例如：

```py
agent.run("Draw me the picture of a capybara swimming in the sea")
```

在这种情况下，模型可以以两种方式理解您的请求：
- 使用`text-to-image` 生成在大海中游泳的大水獭
- 或者，使用`text-to-image`生成大水獭，然后使用`image-transformation`工具使其在大海中游泳

如果您想强制使用第一种情景，可以通过将提示作为参数传递给它来实现：


```py
agent.run("Draw me a picture of the `prompt`", prompt="a capybara swimming in the sea")
```

</Tip>


### 基于交流的执行 (chat)

基于交流的执行（chat）方式是使用 [`~Agent.chat`]：

```py
agent.chat("Generate a picture of rivers and lakes")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("Transform the picture so that there is a rock in there")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

当您希望在不同指令之间保持同一状态时，这会是一个有趣的方法。它更适合用于单个指令，而不是复杂的多步指令（`~Agent.run` 方法更适合处理这种情况）。

这种方法也可以接受参数，以便您可以传递非文本类型或特定提示。

### ⚠️ 远程执行

出于演示目的以便适用于所有设置，我们为发布版本的少数默认工具创建了远程执行器。这些工具是使用推理终端（inference endpoints）创建的。

目前我们已将其关闭，但为了了解如何自行设置远程执行器工具，我们建议阅读[自定义工具指南](./custom_tools)。

### 这里发生了什么？什么是`tools`，什么是`agents`？


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png">


#### Agents

这里的`Agents`是一个大型语言模型，我们通过提示它以访问特定的工具集。

大型语言模型在生成小代码示例方面表现出色，因此这个API利用这一特点，通过提示LLM生成一个使用`tools`集合的小代码示例。然后，根据您给`Agents`的任务和`tools`的描述来完成此提示。这种方式让它能够访问工具的文档，特别是它们的期望输入和输出，以生成相关的代码。

#### Tools

`Tools`非常简单：它们是有名称和描述的单个函数。然后，我们使用这些`tools`的描述来提示代理。通过提示，我们向`agent`展示如何使用`tool`来执行查询语言中请求的操作。

这是使用全新`tools`而不是`pipelines`，因为`agent`编写的代码更好，具有非常原子化的`tools`。`pipelines`经常被重构，并且通常将多个任务合并为一个。`tools`旨在专注于一个非常简单的任务。

#### 代码执行？

然后，这段代码基于`tools`的输入被我们的小型Python解释器执行。我们听到你在后面大声呼喊“任意代码执行！”，但让我们解释为什么情况并非如此。

只能您提供的`tools`和打印函数可以被执行，因此您已经受到了执行的限制。如果仅限于 Hugging Face 工具，那么您应该是安全的。

然后，我们不允许任何属性查找或导入（无论如何都不需要将输入/输出传递给一小组函数），因此所有最明显的攻击（并且您需要提示LLM无论如何输出它们）不应该是一个问题。如果你想超级安全，你可以使用附加参数 return_code=True 执行 run() 方法，在这种情况下，`agent`将只返回要执行的代码，你可以决定是否执行。

如果`agent`生成的代码存在任何尝试执行非法操作的行为，或者代码中出现了常规Python错误，执行将停止。


### 一组经过精心筛选的`tools`

我们确定了一组可以赋予这些`agent`强大能力的`tools`。以下是我们在`transformers`中集成的`tools`的更新列表：

- **文档问答**：给定一个图像格式的文档（例如PDF），回答该文档上的问题（[Donut](../en/model_doc/donut)）
- **文本问答**：给定一段长文本和一个问题，回答文本中的问题（[Flan-T5](../en/model_doc/flan-t5)）
- **无条件图像字幕**：为图像添加字幕！（[BLIP](../en/model_doc/blip)）
- **图像问答**：给定一张图像，回答该图像上的问题（[VILT](../en/model_doc/vilt)）
- **图像分割**：给定一张图像和一个提示，输出该提示的分割掩模（[CLIPSeg](../en/model_doc/clipseg)）
- **语音转文本**：给定一个人说话的音频录音，将演讲内容转录为文本（[Whisper](../en/model_doc/whisper)）
- **文本转语音**：将文本转换为语音（[SpeechT5](../en/model_doc/speecht5)）
- **Zero-Shot文本分类**：给定一个文本和一个标签列表，确定文本最符合哪个标签（[BART](../en/model_doc/bart)）
- **文本摘要**：总结长文本为一两句话（[BART](../en/model_doc/bart)）
- **翻译**：将文本翻译为指定语言（[NLLB](../en/model_doc/nllb)）

这些`tools`已在transformers中集成，并且也可以手动使用，例如：

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### 自定义工具

尽管我们确定了一组经过筛选的`tools`，但我们坚信，此实现提供的主要价值在于能够快速创建和共享自定义`tool`。

通过将工具的代码上传到Hugging Face空间或模型repository，您可以直接通过`agent`使用`tools`。我们已经添加了一些**与transformers无关**的`tools`到[`huggingface-tools`组织](https://huggingface.co/huggingface-tools)中：

- **文本下载器**：从Web URL下载文本
- **文本到图像**：根据提示生成图像，利用`stable diffusion`
- **图像转换**：根据初始图像和提示修改图像，利用`instruct pix2pix stable diffusion`
- **文本到视频**：根据提示生成小视频，利用`damo-vilab`

从一开始就一直在使用的文本到图像`tool`是一个远程`tool `，位于[*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)！我们将继续在此组织和其他组织上发布此类`tool`，以进一步增强此实现。

`agents`默认可以访问存储在[`huggingface-tools`](https://huggingface.co/huggingface-tools)上的`tools`。我们将在后续指南中解释如何编写和共享自定义`tools`，以及如何利用Hub上存在的任何自定义`tools`。

### 代码生成

到目前为止，我们已经展示了如何使用`agents`来为您执行操作。但是，`agents`仅使用非常受限Python解释器执行的代码。如果您希望在不同的环境中使用生成的代码，可以提示`agents`返回代码，以及`tools`的定义和准确的导入信息。

例如，以下指令

```python
agent.run("Draw me a picture of rivers and lakes", return_code=True)
```

返回以下代码

```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

然后你就可以调整并执行代码