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

# Agents和工具

<Tip warning={true}>

Transformers Agents是一个实验性的API，它随时可能发生变化。由于API或底层模型容易发生变化，因此由agents返回的结果可能会有所不同。


</Tip>

要了解更多关于agents和工具的信息，请确保阅读[介绍指南](../transformers_agents)。此页面包含底层类的API文档。


## Agents

我们提供三种类型的agents：[`HfAgent`]使用开源模型的推理端点，[`LocalAgent`]使用您在本地选择的模型，[`OpenAiAgent`]使用OpenAI封闭模型。


### HfAgent

[[autodoc]] HfAgent

### LocalAgent

[[autodoc]] LocalAgent

### OpenAiAgent

[[autodoc]] OpenAiAgent

### AzureOpenAiAgent

[[autodoc]] AzureOpenAiAgent

### Agent

[[autodoc]] Agent 
    - chat 
    - run 
    - prepare_for_new_chat

## 工具

### load_tool

[[autodoc]] load_tool

### Tool

[[autodoc]] Tool

### PipelineTool

[[autodoc]] PipelineTool

### RemoteTool

[[autodoc]] RemoteTool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## Agent类型

Agents可以处理工具之间任何类型的对象；工具是多模态的，可以接受和返回文本、图像、音频、视频等类型。为了增加工具之间的兼容性，以及正确地在ipython（jupyter、colab、ipython notebooks等）中呈现这些返回值，我们实现了这些类型的包装类。

被包装的对象应该继续按照最初的行为方式运作；文本对象应该仍然像字符串一样运作，图像对象应该仍然像`PIL.Image`一样运作。

这些类型有三个特定目的：

- 对类型调用 `to_raw` 应该返回底层对象
- 对类型调用 `to_string` 应该将对象作为字符串返回：在`AgentText`的情况下可能是字符串，但在其他情况下可能是对象序列化版本的路径
- 在ipython内核中显示它应该正确显示对象

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
