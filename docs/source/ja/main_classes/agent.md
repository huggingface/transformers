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

# Agents & Tools

<Tip warning={true}>

Transformers Agentsは実験的なAPIであり、いつでも変更される可能性があります。エージェントが返す結果は、APIまたはその基礎となるモデルが変更される可能性があるため、異なることがあります。

</Tip>

エージェントとツールについて詳しく学ぶには、[導入ガイド](../transformers_agents)を読んでください。このページには基本となるクラスのAPIドキュメントが含まれています。

## Agents

私たちは3つのタイプのエージェントを提供しています：[`HfAgent`]はオープンソースモデルの推論エンドポイントを使用し、[`LocalAgent`]はお好きなモデルをローカルで使用し、[`OpenAiAgent`]はOpenAIのクローズドモデルを使用します。

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

## Tools

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

## Agent Types

エージェントはツールとの間でさまざまなオブジェクトを処理できます。ツールは完全にマルチモーダルであり、テキスト、画像、音声、ビデオなどのさまざまなタイプのデータを受け入れて返すことができます。ツール間の互換性を高めるため、またこれらの返り値をipython（jupyter、colab、ipythonノートブックなど）で正しく表示するために、これらのタイプをラップするクラスを実装しています。

ラップされたオブジェクトは、元の動作を継続する必要があります。テキストオブジェクトは依然として文字列として振る舞うべきであり、画像オブジェクトは依然として`PIL.Image`として振る舞うべきです。

これらのタイプには次の3つの具体的な目的があります：

- タイプ上で`to_raw`を呼び出すと、基になるオブジェクトを返すべきです。
- タイプ上で`to_string`を呼び出すと、オブジェクトを文字列として返すべきです。`AgentText`の場合は文字列になりますが、他のインスタンスの場合はオブジェクトのシリアル化バージョンのパスになります。
- ipythonカーネルで表示する場合、オブジェクトを表示するべきです。

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
