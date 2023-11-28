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

# エージェントとツール

<Tip warning={true}>

Transformers Agents は実験的な API であり、いつでも変更される可能性があります。エージェントから返される結果
API または基礎となるモデルは変更される傾向があるため、変更される可能性があります。

</Tip>

エージェントとツールの詳細については、[入門ガイド](../transformers_agents) を必ずお読みください。このページ
基礎となるクラスの API ドキュメントが含まれています。

## エージェント

私たちは 3 種類のエージェントを提供します。[`HfAgent`] はオープンソース モデルの推論エンドポイントを使用し、[`LocalAgent`] は選択したモデルをローカルで使用し、[`OpenAiAgent`] は OpenAI クローズド モデルを使用します。

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

## エージェントの種類

エージェントはツール間であらゆる種類のオブジェクトを処理できます。ツールは完全にマルチモーダルであるため、受け取りと返品が可能です
テキスト、画像、オーディオ、ビデオなどのタイプ。ツール間の互換性を高めるためだけでなく、
これらの戻り値を ipython (jupyter、colab、ipython ノートブックなど) で正しくレンダリングするには、ラッパー クラスを実装します。
このタイプの周り。

ラップされたオブジェクトは最初と同じように動作し続けるはずです。テキストオブジェクトは依然として文字列または画像として動作する必要があります
オブジェクトは依然として `PIL.Image` として動作するはずです。

これらのタイプには、次の 3 つの特定の目的があります。

- 型に対して `to_raw` を呼び出すと、基になるオブジェクトが返されるはずです
- 型に対して `to_string` を呼び出すと、オブジェクトを文字列として返す必要があります。`AgentText` の場合は文字列になる可能性があります。
  ただし、他のインスタンスのオブジェクトのシリアル化されたバージョンのパスになります。
- ipython カーネルで表示すると、オブジェクトが正しく表示されるはずです

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
