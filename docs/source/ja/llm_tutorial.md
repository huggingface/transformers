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

# Generation with LLMs

[[open-in-colab]]

LLM、またはLarge Language Models（大規模言語モデル）は、テキスト生成の鍵となる要素です。要するに、これらは大規模な事前訓練済みトランスフォーマーモデルで、与えられた入力テキストに基づいて次の単語（または、より正確にはトークン）を予測するように訓練されています。トークンを1つずつ予測するため、モデルを呼び出すだけでは新しい文を生成するために何かより精巧なことをする必要があります。自己回帰生成を行う必要があります。

自己回帰生成は、推論時の手続きで、いくつかの初期入力を与えた状態で、モデルを反復的に呼び出す手法です。🤗 Transformersでは、これは[`~generation.GenerationMixin.generate`]メソッドによって処理され、これは生成能力を持つすべてのモデルで利用可能です。

このチュートリアルでは、以下のことを示します：

* LLMを使用してテキストを生成する方法
* 一般的な落とし穴を回避する方法
* LLMを最大限に活用するための次のステップ

始める前に、必要なライブラリがすべてインストールされていることを確認してください：


```bash
pip install transformers bitsandbytes>=0.39.0 -q
```

## Generate text

[因果言語モデリング](tasks/language_modeling)のためにトレーニングされた言語モデルは、テキストトークンのシーケンスを入力として受け取り、次のトークンの確率分布を返します。

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"Forward pass of an LLM"</figcaption>
</figure>


LLM（Language Model）による自己回帰生成の重要な側面の1つは、この確率分布から次のトークンを選択する方法です。このステップでは、次のイテレーションのためのトークンが得られる限り、何でも可能です。これは、確率分布から最も可能性の高いトークンを選択するだけのシンプルな方法から、結果の分布からサンプリングする前に数々の変換を適用するほど複雑な方法まで、あらゆる方法が考えられます。


<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"Autoregressive generation iteratively selects the next token from a probability distribution to generate text"</figcaption>
</figure>

上記のプロセスは、ある停止条件が満たされるまで反復的に繰り返されます。理想的には、停止条件はモデルによって指示され、モデルは終了シーケンス（`EOS`）トークンを出力するタイミングを学習すべきです。これがそうでない場合、生成はあらかじめ定義された最大長に達したときに停止します。

トークン選択ステップと停止条件を適切に設定することは、モデルがタスクで期待どおりに振る舞うために重要です。それが、各モデルに関連付けられた [`~generation.GenerationConfig`] ファイルがある理由であり、これには優れたデフォルトの生成パラメータ化が含まれ、モデルと一緒に読み込まれます。

コードについて話しましょう！

<Tip>

基本的なLLMの使用に興味がある場合、高レベルの [`Pipeline`](pipeline_tutorial) インターフェースが良い出発点です。ただし、LLMはしばしば量子化やトークン選択ステップの細かい制御などの高度な機能が必要であり、これは [`~generation.GenerationMixin.generate`] を介して最良に行われます。LLMとの自己回帰生成はリソースが多く必要であり、適切なスループットのためにGPUで実行する必要があります。

</Tip>

<!-- TODO: llama 2（またはより新しい一般的なベースライン）が利用可能になったら、例を更新する -->
まず、モデルを読み込む必要があります。


```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

`from_pretrained` 呼び出しで2つのフラグがあることに注意してください：

- `device_map` はモデルをあなたのGPUに移動させます
- `load_in_4bit` は[4ビットの動的量子化](main_classes/quantization)を適用してリソース要件を大幅に削減します

モデルを初期化する他の方法もありますが、これはLLMを始めるための良い基準です。

次に、[トークナイザ](tokenizer_summary)を使用してテキスト入力を前処理する必要があります。


```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```


`model_inputs` 変数は、トークン化されたテキスト入力とアテンションマスクを保持しています。 [`~generation.GenerationMixin.generate`] は、アテンションマスクが渡されていない場合でも、最善の努力をしてそれを推測しようとしますが、できる限り渡すことをお勧めします。最適な結果を得るためです。

最後に、[`~generation.GenerationMixin.generate`] メソッドを呼び出して生成されたトークンを取得し、それを表示する前にテキストに変換する必要があります。


```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```

これで完了です！わずかなコード行数で、LLM（Large Language Model）のパワーを活用できます。

## Common pitfalls

[生成戦略](generation_strategies)はたくさんあり、デフォルトの値があなたのユースケースに適していないことがあります。出力が期待通りでない場合、最も一般的な落とし穴とその回避方法のリストを作成しました。

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

### Generated output is too short/long

[`~generation.GenerationConfig`] ファイルで指定されていない場合、`generate` はデフォルトで最大で 20 トークンまで返します。我々は `generate` コールで `max_new_tokens` を手動で設定することを強くお勧めします。これにより、返される新しいトークンの最大数を制御できます。LLM（正確には、[デコーダー専用モデル](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)）も出力の一部として入力プロンプトを返すことに注意してください。

```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### Incorrect generation mode

デフォルトでは、 [`~generation.GenerationConfig`] ファイルで指定されていない限り、`generate` は各イテレーションで最も可能性の高いトークンを選択します（貪欲デコーディング）。タスクに応じて、これは望ましくないことがあります。チャットボットやエッセイのような創造的なタスクでは、サンプリングが有益です。一方、音声の転写や翻訳のような入力に基づくタスクでは、貪欲デコーディングが有益です。`do_sample=True` でサンプリングを有効にできます。このトピックについての詳細は、この[ブログポスト](https://huggingface.co/blog/how-to-generate)で学ぶことができます。

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(0)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.\nI just need to be. I am always.\nEvery time'
```

### Wrong padding side

LLM（Large Language Models）は[デコーダー専用](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)のアーキテクチャであり、入力プロンプトを繰り返し処理することを意味します。入力が同じ長さでない場合、それらをパディングする必要があります。LLMはパッドトークンからの続きを学習していないため、入力は左パディングする必要があります。また、生成に対して注目マスクを渡し忘れないようにしてください！


```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
''

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

## Further resources

オートリグレッシブ生成プロセスは比較的簡単ですが、LLMを最大限に活用することは多くの要素が絡むため、挑戦的な試みとなります。LLMの使用と理解をさらに深めるための次のステップについては以下のリソースをご覧ください。

<!-- TODO: 新しいガイドで完了 -->
### Advanced generate usage

1. [ガイド](generation_strategies)：異なる生成方法を制御する方法、生成構成ファイルの設定方法、出力のストリーミング方法についてのガイド;
2. [`~generation.GenerationConfig`]、[`~generation.GenerationMixin.generate`]、および[生成関連クラス](internal/generation_utils)に関するAPIリファレンス。

### LLM leaderboards

1. [Open LLM リーダーボード](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)：オープンソースモデルの品質に焦点を当てたリーダーボード;
2. [Open LLM-Perf リーダーボード](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)：LLMのスループットに焦点を当てたリーダーボード。

### Latency and throughput

1. [ガイド](main_classes/quantization)：ダイナミッククオンタイズに関するガイド。これによりメモリ要件を劇的に削減する方法が示されています。

### Related libraries

1. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference)：LLM用の本番向けサーバー;
2. [`optimum`](https://github.com/huggingface/optimum)：特定のハードウェアデバイス向けに最適化された🤗 Transformersの拡張。
