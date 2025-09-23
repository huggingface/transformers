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


# LLM prompting guide

[[open-in-colab]]

Falcon、LLaMA などの大規模言語モデルは、事前にトレーニングされたトランスフォーマー モデルであり、最初は予測するようにトレーニングされています。
入力テキストが与えられた場合の次のトークン。通常、数十億のパラメータがあり、何兆ものパラメータでトレーニングされています。
長期間のトークン。その結果、これらのモデルは非常に強力で多用途になり、次のようなことが可能になります。
自然言語プロンプトでモデルに指示することで、すぐに複数の NLP タスクを解決できます。

最適な出力を保証するためにこのようなプロンプトを設計することは、多くの場合「プロンプト エンジニアリング」と呼ばれます。プロンプトエンジニアリングとは、
かなりの量の実験を必要とする反復プロセス。自然言語ははるかに柔軟で表現力豊かです
ただし、プログラミング言語よりもあいまいさが生じる可能性があります。同時に、自然言語によるプロンプト
変化にはかなり敏感です。プロンプトにわずかな変更を加えただけでも、出力が大幅に異なる場合があります。

すべてのケースに適合するプロンプトを作成するための正確なレシピはありませんが、研究者はいくつかの最良のレシピを考案しました。
最適な結果をより一貫して達成するのに役立つ実践。

このガイドでは、より優れた LLM プロンプトを作成し、さまざまな NLP タスクを解決するのに役立つプロンプト エンジニアリングのベスト プラクティスについて説明します。
次のことを学びます:

- [プロンプトの基本](#basics-of-prompting)
- [LLM プロンプトのベスト プラクティス](#best-practices-of-llm-prompting)
- [高度なプロンプト テクニック: 数回のプロンプトと思考の連鎖](#advanced-prompting-techniques)
- [プロンプトを表示する代わりに微調整する場合](#prompting-vs-fine-tuning)

<Tip>

迅速なエンジニアリングは、LLM 出力最適化プロセスの一部にすぎません。もう 1 つの重要な要素は、
最適なテキスト生成戦略。 LLM が生成時に後続の各トークンを選択する方法をカスタマイズできます。
トレーニング可能なパラメータを一切変更せずにテキストを作成します。テキスト生成パラメータを微調整することで、
生成されたテキストに繰り返しが含まれているため、より一貫性があり人間らしい響きになります。
テキスト生成戦略とパラメーターはこのガイドの範囲外ですが、これらのトピックについて詳しくは、次のトピックを参照してください。
次のガイド:
 
* [LLM による生成](../llm_tutorial)
* [テキスト生成戦略](../generation_strategies)

</Tip>

## Basics of prompting

### Types of models 

最新の LLM の大部分は、デコーダ専用のトランスフォーマーです。例としては、[LLaMA](../model_doc/llama)、
[Llama2](../model_doc/llama2)、[Falcon](../model_doc/falcon)、[GPT2](../model_doc/gpt2)。ただし、遭遇する可能性があります
エンコーダ デコーダ トランスフォーマ LLM も同様です。たとえば、[Flan-T5](../model_doc/flan-t5) や [BART](../model_doc/bart) です。

エンコーダ デコーダ スタイルのモデルは通常、出力が入力に**大きく**依存する生成タスクで使用されます。
たとえば、翻訳と要約です。デコーダ専用モデルは、他のすべてのタイプの生成タスクに使用されます。

パイプラインを使用して LLM でテキストを生成する場合、使用している LLM のタイプを知ることが重要です。
異なるパイプラインを使用します。

`text-generation`パイプラインを使用してデコーダのみのモデルで推論を実行します。

```python
>>> from transformers import pipeline
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT

>>> generator = pipeline('text-generation', model = 'openai-community/gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)
[{'generated_text': "Hello, I'm a language model expert, so I'm a big believer in the concept that I know very well and then I try to look into"}]
```

エンコーダー/デコーダーを使用して推論を実行するには、`text2text-generation` パイプラインを使用します。

```python
>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)
[{'generated_text': 'Je suis très heureuse de vous rencontrer.'}]
```

### Base vs instruct/chat models

🤗 Hub で利用できる最近の LLM チェックポイントのほとんどには、base と instruct (または chat) の 2 つのバージョンがあります。例えば、
[`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b) および [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b) -指示する)。

基本モデルは、最初のプロンプトが与えられたときにテキストを完成させるのには優れていますが、NLP タスクには理想的ではありません。
指示に従う必要がある場合、または会話で使用する場合に使用します。ここで、指示 (チャット) バージョンが登場します。
これらのチェックポイントは、命令と会話データに基づいて事前トレーニングされたベース バージョンをさらに微調整した結果です。
この追加の微調整により、多くの NLP タスクにとってより適切な選択肢になります。

[`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct) で使用できるいくつかの簡単なプロンプトを示してみましょう。
いくつかの一般的な NLP タスクを解決します。

### NLP tasks 

まず、環境をセットアップしましょう。

```bash
pip install -q transformers accelerate
```

次に、適切なパイプライン (`text_generation`) を使用してモデルをロードしましょう。

```python
>>> from transformers import pipeline, AutoTokenizer
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipe = pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     dtype=torch.bfloat16,
...     device_map="auto",
... )
```

<Tip>

Falcon モデルは `bfloat16` データ型を使用してトレーニングされたため、同じものを使用することをお勧めします。これには、最近の
CUDA のバージョンに準拠しており、最新のカードで最適に動作します。

</Tip>

パイプライン経由でモデルをロードしたので、プロンプトを使用して NLP タスクを解決する方法を見てみましょう。

#### Text classification

テキスト分類の最も一般的な形式の 1 つはセンチメント分析であり、「ポジティブ」、「ネガティブ」、「ネガティブ」などのラベルを割り当てます。
または、一連のテキストに対して「中立」です。与えられたテキスト (映画レビュー) を分類するようにモデルに指示するプロンプトを作成してみましょう。
まず指示を与え、次に分類するテキストを指定します。そのままにしておくのではなく、
応答の先頭にも追加します - `"Sentiment: "`:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Classify the text into neutral, negative or positive. 
... Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
... Sentiment:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

その結果、出力には、手順で提供したリストの分類ラベルが含まれており、それは正しいラベルです。

<Tip>

プロンプトに加えて、`max_new_tokens`パラメータを渡していることに気づくかもしれません。トークンの数を制御します。
モデルが生成します。これは、学習できる多くのテキスト生成パラメーターの 1 つです。
[テキスト生成戦略](../generation_strategies) ガイドを参照してください。

</Tip>

#### Named Entity Recognition

固有表現認識 (NER) は、テキスト内の人物、場所、組織などの固有表現を検索するタスクです。
プロンプトの指示を変更して、LLM にこのタスクを実行させましょう。ここでは`return_full_text = False`も設定しましょう
出力にプロンプ​​トが含​​まれないようにします。

```python
>>> torch.manual_seed(1) # doctest: +IGNORE_RESULT
>>> prompt = """Return a list of named entities in the text.
... Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
... Named entities:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=15,
...     return_full_text = False,    
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
- Golden State Warriors
- San Francisco
```

ご覧のとおり、モデルは指定されたテキストから 2 つの名前付きエンティティを正しく識別しました。

#### Translation

LLM が実行できるもう 1 つのタスクは翻訳です。このタスクにはエンコーダー/デコーダー モデルを使用することを選択できますが、ここでは
例を簡単にするために、きちんとした仕事をする Falcon-7b-instruct を使い続けます。もう一度、方法は次のとおりです
テキストの一部を英語からイタリア語に翻訳するようにモデルに指示する基本的なプロンプトを作成できます。

```python
>>> torch.manual_seed(2) # doctest: +IGNORE_RESULT
>>> prompt = """Translate the English text to Italian.
... Text: Sometimes, I've believed as many as six impossible things before breakfast.
... Translation:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=20,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
A volte, ho creduto a sei impossibili cose prima di colazione.
```

ここでは、出力生成時にモデルがもう少し柔軟になるように `do_sample=True` と `top_k=10` を追加しました。

#### Text summarization

翻訳と同様に、テキストの要約も、出力が入力に**大きく**依存する生成タスクです。
エンコーダ/デコーダ モデルの方が良い選択になる可能性があります。ただし、デコーダ スタイルのモデルもこのタスクに使用できます。
以前は、プロンプトの先頭に指示を配置していました。ただし、プロンプトの最後で、
指示を与えるのに適した場所でもあります。通常、命令はどちらかの端に配置することをお勧めします。

```python
>>> torch.manual_seed(3) # doctest: +IGNORE_RESULT
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
... Write a summary of the above text.
... Summary:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
Permaculture is an ecological design mimicking natural ecosystems to meet basic needs and prepare for climate change. It is based on traditional knowledge and scientific understanding.
```

#### Question answering

質問応答タスクの場合、プロンプトを次の論理コンポーネントに構造化できます: 指示、コンテキスト、質問、
先頭の単語またはフレーズ (`"Answer:"`) を使用して、モデルを操作して答えの生成を開始します。


```python
>>> torch.manual_seed(4) # doctest: +IGNORE_RESULT
>>> prompt = """Answer the question using the context below.
... Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
... Question: What modern tool is used to make gazpacho?
... Answer:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Modern tools are used, such as immersion blenders
```

#### Reasoning

LLM にとって推論は最も困難なタスクの 1 つであり、良い結果を達成するには、多くの場合、次のような高度なプロンプト テクニックを適用する必要があります。
[Chain-of-thought](#chain-of-thought)。

基本的なプロンプトを使用して、単純な算術タスクに関するモデル推論を作成できるかどうか試してみましょう。

```python
>>> torch.manual_seed(5) # doctest: +IGNORE_RESULT
>>> prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
There are a total of 5 groups, so there are 5 x 4=20 students in the class.
```

正しい！もう少し複雑さを増やして、基本的なプロンプトで問題を解決できるかどうかを確認してみましょう。

```python
>>> torch.manual_seed(6) # doctest: +IGNORE_RESULT
>>> prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
The total number of muffins now is 21
```

これは間違った答えです。12 である必要があります。この場合、プロンプトが基本的すぎるか、選択内容が原因である可能性があります。
結局のところ、Falcon の最小バージョンを選択しました。あらゆるサイズのモデルでは推論が困難ですが、より大きなモデルでは
モデルのパフォーマンスが向上する可能性があります。

## Best practices of LLM prompting

ガイドのこのセクションでは、プロンプトの結果を改善する傾向にあるベスト プラクティスのリストをまとめました。

* 使用するモデルを選択する場合は、最新かつ最も機能的なモデルの方がパフォーマンスが向上する可能性があります。
* シンプルで短いプロンプトから始めて、そこから繰り返します。
* 指示はプロンプトの最初または最後に入力してください。大規模なコンテキストを扱う場合、モデルはさまざまな最適化を適用して、アテンションの複雑さが二次的に拡大するのを防ぎます。これにより、モデルはプロンプトの途中よりも最初または最後に注意を払うようになります。
* 指示と、それが適用されるテキストを明確に区別してください。これについては、次のセクションで詳しく説明します。
* タスクと望ましい結果 (その形式、長さ、スタイル、言語など) について具体的かつ説明的にします。
* 曖昧な説明や指示は避けてください。
*「何をしてはいけないか」という指示ではなく、「何をすべきか」という指示を優先します。
* 最初の単語を書いて (またはモデルの最初の文を始めて)、出力を正しい方向に「導き」ます。
* [Few-shot prompting](#few-shot-prompting) や [Chain-of-thought](#chain-of-thought) などの高度なテクニックを使用します。
* さまざまなモデルでプロンプトをテストして、その堅牢性を評価します。
* プロンプトのバージョンを確認し、パフォーマンスを追跡します。

## Advanced prompting techniques

### Few-shot prompting

上記のセクションの基本的なプロンプトは、「ゼロショット」プロンプトの例です。つまり、モデルにはすでに与えられています。
指示とコンテキストはありますが、解決策を含む例はありません。通常、命令データセットに基づいて微調整された LLM
このような「ゼロショット」タスクでも優れたパフォーマンスを発揮します。ただし、タスクがより複雑であったり微妙な点があったりする場合があります。
出力には、命令だけではモデルが理解できないいくつかの要件があります。この場合、次のことができます。
少数ショット プロンプトと呼ばれるテクニックを試してください。

少数ショット プロンプトでは、モデルにパフォーマンスを向上させるためのより多くのコンテキストを提供するプロンプト内の例が提供されます。
例では、例のパターンに従って出力を生成するようにモデルを条件付けします。

以下に例を示します。

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
... Date: 04/12/1961
... Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
... Date:"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=8,
...     do_sample=True,
...     top_k=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

上記のコード スニペットでは、モデルへの目的の出力を示すために 1 つの例を使用しました。したがって、これは、
「ワンショット」プロンプト。ただし、タスクの複雑さに応じて、複数の例を使用する必要がある場合があります。

数回のプロンプト手法の制限:
- LLM は例のパターンを理解できますが、これらの手法は複雑な推論タスクではうまく機能しません。
- 少数ショットのプロンプトでは、長いプロンプトを作成する必要があります。大量のトークンを含むプロンプトでは、計算量と待ち時間が増加する可能性があります。プロンプトの長さにも制限があります。
- 多くの例を与えると、モデルが学習するつもりのなかったパターンを学習することがあります。 3番目の映画レビューはいつも否定的だということ。

### Chain-of-thought

思考連鎖 (CoT) プロンプトは、モデルを微調整して中間推論ステップを生成し、改善する手法です。
複雑な推論タスクの結果。

モデルを操作して推論ステップを生成するには、2 つの方法があります。
- 質問に対する詳細な回答を含む例を示し、問題に対処する方法をモデルに示すことで、数回のプロンプトを表示します。
- 「ステップごとに考えてみましょう」または「深呼吸して、問題をステップごとに解決してください」などのフレーズを追加してモデルに推論を指示します。

[推論セクション](#reasoning) のマフィンの例に CoT テクニックを適用し、より大きなモデルを使用すると、
[HuggingChat](https://huggingface.co/chat/)で遊べる(`tiiuae/falcon-180B-chat`)など、
推論結果は大幅に改善されます。

```text
Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
Therefore, you now have 12 muffins.
```

## Prompting vs fine-tuning

プロンプトを最適化することで優れた結果を達成できますが、モデルを微調整するかどうかについてはまだ思案するかもしれません。
あなたの場合にはもっとうまくいくでしょう。より小規模なモデルを微調整することが好ましいオプションである場合のいくつかのシナリオを次に示します。

- ドメインが LLM が事前にトレーニングされたものと大きく異なっており、広範なプロンプト最適化では十分な結果が得られませんでした。
- モデルが低リソース言語で適切に動作する必要があります。
- 厳格な規制の下にある機密データでモデルをトレーニングする必要があります。
- コスト、プライバシー、インフラストラクチャ、またはその他の制限により、小規模なモデルを使用する必要があります。

上記のすべての例で、十分な大きさのファイルをすでに持っているか、簡単に入手できるかを確認する必要があります。
ドメイン固有のデータセットを合理的なコストでモデルを微調整できます。十分な時間とリソースも必要になります
モデルを微調整します。

上記の例が当てはまらない場合は、プロンプトを最適化する方が有益であることがわかります。
