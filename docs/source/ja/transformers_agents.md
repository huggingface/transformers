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

Transformers Agentsは、いつでも変更される可能性のある実験的なAPIです。エージェントが返す結果は、APIまたは基礎となるモデルが変更される可能性があるため、異なることがあります。

</Tip>

Transformersバージョンv4.29.0は、*ツール*と*エージェント*のコンセプトを基に構築されています。この[colab](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj)で試すことができます。

要するに、これはtransformersの上に自然言語APIを提供するものです：私たちは一連の厳選されたツールを定義し、自然言語を解釈し、これらのツールを使用するエージェントを設計します。これは設計上拡張可能です。私たちはいくつかの関連するツールを厳選しましたが、コミュニティによって開発された任意のツールを使用するためにシステムを簡単に拡張できる方法も示します。

この新しいAPIで何ができるかのいくつかの例から始めましょう。特に多モーダルなタスクに関して強力ですので、画像を生成したりテキストを読み上げたりするのに最適です。

上記のテキストの上に、日本語の翻訳を提供します。


```py
agent.run("Caption the following image", image=image)
```

| **Input**                                                                                                                   | **Output**                        |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |

---

```py
agent.run("Read the following text out loud", text=text)
```
| **Input**                                                                                                               | **Output**                                   |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio>

---

```py
agent.run(
    "In the following `document`, where will the TRRF Scientific Advisory Council Meeting take place?",
    document=document,
)
```
| **Input**                                                                                                                   | **Output**     |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | ballroom foyer |

## Quickstart

`agent.run`を使用する前に、エージェントをインスタンス化する必要があります。エージェントは、大規模な言語モデル（LLM）です。
OpenAIモデルとBigCode、OpenAssistantからのオープンソースの代替モデルをサポートしています。OpenAIモデルはパフォーマンスが優れていますが、OpenAIのAPIキーが必要であり、無料で使用することはできません。一方、Hugging FaceはBigCodeとOpenAssistantモデルのエンドポイントへの無料アクセスを提供しています。

まず、デフォルトの依存関係をすべてインストールするために`agents`のエクストラをインストールしてください。


```bash
pip install transformers[agents]
```

OpenAIモデルを使用するには、`openai`の依存関係をインストールした後、`OpenAiAgent`をインスタンス化します。


```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

BigCodeまたはOpenAssistantを使用するには、まずログインしてInference APIにアクセスしてください。

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

次に、エージェントをインスタンス化してください。

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

これは、Hugging Faceが現在無料で提供している推論APIを使用しています。このモデル（または別のモデル）の独自の推論エンドポイントをお持ちの場合は、上記のURLエンドポイントをご自分のURLエンドポイントで置き換えることができます。

<Tip>

StarCoderとOpenAssistantは無料で利用でき、シンプルなタスクには非常に優れた性能を発揮します。ただし、より複雑なプロンプトを処理する際には、チェックポイントが十分でないことがあります。そのような場合には、現時点ではオープンソースではないものの、パフォーマンスが向上する可能性のあるOpenAIモデルを試してみることをお勧めします。

</Tip>

これで準備が整いました！これから、あなたが利用できる2つのAPIについて詳しく説明します。

### Single execution (run)

単一実行メソッドは、エージェントの [`~Agent.run`] メソッドを使用する場合です。


```py
agent.run("Draw me a picture of rivers and lakes.")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>


これは、実行したいタスクに適したツール（またはツール）を自動的に選択し、適切に実行します。1つまたは複数のタスクを同じ命令で実行することができます（ただし、命令が複雑であるほど、エージェントが失敗する可能性が高くなります）。


```py
agent.run("Draw me a picture of the sea then transform the picture to add an island")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>

[`~Agent.run`] 操作は独立して実行できますので、異なるタスクで何度も実行することができます。

注意点として、あなたの `agent` は単なる大規模な言語モデルであるため、プロンプトのわずかな変更でも完全に異なる結果が得られる可能性があります。したがって、実行したいタスクをできるだけ明確に説明することが重要です。良いプロンプトの書き方については、[こちら](custom_tools#writing-good-user-inputs) で詳しく説明しています。

実行ごとに状態を保持したり、テキスト以外のオブジェクトをエージェントに渡したりする場合は、エージェントが使用する変数を指定することができます。例えば、最初の川や湖の画像を生成し、その画像に島を追加するようにモデルに指示するには、次のように行うことができます：

```python
picture = agent.run("Generate a picture of rivers and lakes.")
updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
```

<Tip>

これは、モデルがあなたのリクエストを理解できない場合や、ツールを混同する場合に役立つことがあります。例えば：

```py
agent.run("Draw me the picture of a capybara swimming in the sea")
```

ここでは、モデルは2つの方法で解釈できます：
- `text-to-image`に海で泳ぐカピバラを生成させる
- または、`text-to-image`でカピバラを生成し、それを海で泳がせるために`image-transformation`ツールを使用する

最初のシナリオを強制したい場合は、プロンプトを引数として渡すことができます：


```py
agent.run("Draw me a picture of the `prompt`", prompt="a capybara swimming in the sea")
```

</Tip>


### Chat-based execution (チャット)

エージェントは、[`~Agent.chat`] メソッドを使用することで、チャットベースのアプローチも可能です。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("Transform the picture so that there is a rock in there")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

これは、指示をまたいで状態を保持したい場合に便利なアプローチで、単一の指示に比べて複雑な指示を処理するのは難しいかもしれません（その場合は [`~Agent.run`] メソッドの方が適しています）。

このメソッドは、非テキスト型の引数や特定のプロンプトを渡したい場合にも使用できます。

### ⚠️ Remote execution

デモンストレーションの目的やすべてのセットアップで使用できるように、リリースのためにいくつかのデフォルトツール用のリモート実行ツールも作成しました。これらは [推論エンドポイント](https://huggingface.co/inference-endpoints) を使用して作成されます。

これらは現在オフになっていますが、リモート実行ツールを自分で設定する方法については、[カスタムツールガイド](./custom_tools) を読むことをお勧めします。

### What's happening here? What are tools, and what are agents?

![エージェントとツールのダイアグラム](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png)

#### Agents

ここでの「エージェント」とは、大規模な言語モデルのことであり、特定の一連のツールにアクセスできるようにプロンプトを設定しています。

LLM（大規模言語モデル）は、コードの小さなサンプルを生成するのにかなり優れており、このAPIは、エージェントに特定のツールセットを使用してタスクを実行するコードの小さなサンプルを生成させることに利用しています。このプロンプトは、エージェントにタスクとツールの説明を提供することで、エージェントが使用しているツールのドキュメントにアクセスし、関連するコードを生成できるようになります。

#### Tools

ツールは非常に単純で、名前と説明からなる単一の関数です。それから、これらのツールの説明を使用してエージェントをプロンプトします。プロンプトを通じて、エージェントに、ツールを使用してクエリで要求されたタスクをどのように実行するかを示します。特に、ツールの期待される入力と出力を示します。

これは新しいツールを使用しており、パイプラインではなくツールを使用しています。なぜなら、エージェントは非常に原子的なツールでより良いコードを生成するからです。パイプラインはよりリファクタリングされ、しばしば複数のタスクを組み合わせています。ツールは非常に単純なタスクに焦点を当てることを意図しています。

#### Code-execution?!

このコードは、ツールとツールと一緒に渡される入力のセットで、当社の小規模なPythonインタープリタで実行されます。すでに提供されたツールとprint関数しか呼び出すことができないため、実行できることはすでに制限されています。Hugging Faceのツールに制限されているため、安全だと考えても問題ありません。

さらに、属性の検索やインポートは許可しておらず（それらは渡された入力/出力を処理するためには必要ないはずです）、最も明らかな攻撃は問題ありません（エージェントにそれらを出力するようにプロンプトする必要があります）。超安全な側に立ちたい場合は、追加の引数 return_code=True を指定して run() メソッドを実行できます。その場合、エージェントは実行するコードを返すだけで、実行するかどうかはあなた次第です。

実行は、違法な操作を試みる行またはエージェントが生成したコードに通常のPythonエラーがある場合に停止します。

### A curated set of tools

私たちは、このようなエージェントを強化できるツールのセットを特定します。以下は、`transformers`に統合されたツールの更新されたリストです：

- **ドキュメント質問応答**: 画像形式のドキュメント（PDFなど）が与えられた場合、このドキュメントに関する質問に回答します（[Donut](./model_doc/donut)）
- **テキスト質問応答**: 長いテキストと質問が与えられた場合、テキスト内の質問に回答します（[Flan-T5](./model_doc/flan-t5)）
- **無条件の画像キャプション**: 画像にキャプションを付けます！（[BLIP](./model_doc/blip)）
- **画像質問応答**: 画像が与えられた場合、その画像に関する質問に回答します（[VILT](./model_doc/vilt)）
- **画像セグメンテーション**: 画像とプロンプトが与えられた場合、そのプロンプトのセグメンテーションマスクを出力します（[CLIPSeg](./model_doc/clipseg)）
- **音声からテキストへの変換**: 人の話し声のオーディオ録音が与えられた場合、その音声をテキストに転記します（[Whisper](./model_doc/whisper)）
- **テキストから音声への変換**: テキストを音声に変換します（[SpeechT5](./model_doc/speecht5)）
- **ゼロショットテキスト分類**: テキストとラベルのリストが与えられた場合、テキストが最も対応するラベルを識別します（[BART](./model_doc/bart)）
- **テキスト要約**: 長いテキストを1つまたは数文に要約します（[BART](./model_doc/bart)）
- **翻訳**: テキストを指定された言語に翻訳します（[NLLB](./model_doc/nllb)）

これらのツールはtransformersに統合されており、手動でも使用できます。たとえば、次のように使用できます：

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### Custom tools

私たちは、厳選されたツールのセットを特定する一方、この実装が提供する主要な価値は、カスタムツールを迅速に作成して共有できる能力だと強く信じています。

ツールのコードをHugging Face Spaceまたはモデルリポジトリにプッシュすることで、エージェントと直接連携してツールを活用できます。[`huggingface-tools` organization](https://huggingface.co/huggingface-tools)には、**transformers非依存**のいくつかのツールが追加されました：

- **テキストダウンローダー**: ウェブURLからテキストをダウンロードするためのツール
- **テキストから画像へ**: プロンプトに従って画像を生成するためのツール。安定した拡散を活用します
- **画像変換**: 初期画像とプロンプトを指定して画像を変更するためのツール。instruct pix2pixの安定した拡散を活用します
- **テキストからビデオへ**: プロンプトに従って小さなビデオを生成するためのツール。damo-vilabを活用します

最初から使用しているテキストから画像へのツールは、[*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)にあるリモートツールです！今後も、この組織および他の組織にさらにこのようなツールをリリースし、この実装をさらに強化していきます。

エージェントはデフォルトで[`huggingface-tools`](https://huggingface.co/huggingface-tools)にあるツールにアクセスできます。
ツールの作成と共有方法、またHubに存在するカスタムツールを活用する方法についての詳細は、[次のガイド](custom_tools)で説明しています。

### Code generation

これまで、エージェントを使用してあなたのためにアクションを実行する方法を示しました。ただし、エージェントはコードを生成するだけで、非常に制限されたPythonインタープリタを使用して実行します。生成されたコードを異なる環境で使用したい場合、エージェントにコードを返すように指示できます。ツールの定義と正確なインポートも含めて。

例えば、以下の命令：
```python
agent.run("Draw me a picture of rivers and lakes", return_code=True)
```

次のコードを返します
```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

その後、自分で変更して実行できます。
