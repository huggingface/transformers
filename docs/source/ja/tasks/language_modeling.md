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

# Causal language modeling


[[open-in-colab]]


言語モデリングには、因果的モデリングとマスクされた言語モデリングの 2 つのタイプがあります。このガイドでは、因果関係のある言語モデリングについて説明します。
因果言語モデルはテキスト生成によく使用されます。これらのモデルは、次のようなクリエイティブなアプリケーションに使用できます。
独自のテキスト アドベンチャーを選択するか、Copilot や CodeParrot などのインテリジェントなコーディング アシスタントを選択します。

<Youtube id="Vpjb1lu0MDk"/>


因果言語モデリングは、一連のトークン内の次のトークンを予測します。モデルは、次のトークンにのみ対応できます。
左。これは、モデルが将来のトークンを認識できないことを意味します。 GPT-2 は因果的言語モデルの一例です。

このガイドでは、次の方法を説明します。

1. [ELI5](https:/) の [r/askscience](https://www.reddit.com/r/askscience/) サブセットで [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) を微調整します。 /huggingface.co/datasets/eli5) データセット。
2. 微調整したモデルを推論に使用します。

<Tip>

このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/text-generation) を確認することをお勧めします。u

</Tip>

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install transformers datasets evaluate
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load ELI5 dataset


まず、ELI5 データセットの r/askscience サブセットの小さいサブセットを 🤗 データセット ライブラリからロードします。
 これにより、完全なデータセットのトレーニングにさらに時間を費やす前に、実験してすべてが機能することを確認する機会が得られます。

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5", split="train_asks[:5000]")
```

[`~datasets.Dataset.train_test_split`] メソッドを使用して、データセットの `train_asks` をトレイン セットとテスト セットに分割します。

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

次に、例を見てみましょう。

```py
>>> eli5["train"][0]
{'answers': {'a_id': ['c3d1aib', 'c3d4lya'],
  'score': [6, 3],
  'text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
   "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"]},
 'answers_urls': {'url': []},
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls': {'url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg']},
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls': {'url': []}}
```

これは多くのことのように見えるかもしれませんが、実際に関心があるのは`text`フィールドだけです。言語モデリングの優れている点
タスクでは、次の単語がラベル * であるため、ラベル (教師なしタスクとも呼ばれます) は必要ありません。


## Preprocess

<Youtube id="ma1TrR7gE7I"/>


次のステップは、`text`サブフィールドを処理するために DistilGPT2 トークナイザーをロードすることです。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
```

上の例からわかるように、`text`フィールドは実際には`answers`内にネストされています。つまり、次のことが必要になります。
[` flatten`](https://huggingface.co/docs/datasets/process.html#flatten) メソッドを使用して、ネストされた構造から `text` サブフィールドを抽出します。

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'answers.a_id': ['c3d1aib', 'c3d4lya'],
 'answers.score': [6, 3],
 'answers.text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
  "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"],
 'answers_urls.url': [],
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls.url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg'],
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls.url': []}
```

`answers`接頭辞で示されるように、各サブフィールドは個別の列になり、`text`フィールドはリストになりました。その代わり
各文を個別にトークン化する場合は、リストを文字列に変換して、それらをまとめてトークン化できるようにします。

以下は、各例の文字列のリストを結合し、結果をトークン化する最初の前処理関数です。

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

この前処理関数をデータセット全体に適用するには、🤗 Datasets [`~datasets.Dataset.map`] メソッドを使用します。 `map` 関数を高速化するには、`batched=True` を設定してデータセットの複数の要素を一度に処理し、`num_proc` でプロセスの数を増やします。不要な列を削除します。

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

このデータセットにはトークン シーケンスが含まれていますが、その一部はモデルの最大入力長よりも長くなります。

2 番目の前処理関数を使用して、
- すべてのシーケンスを連結します
- 連結されたシーケンスを`block_size`で定義された短いチャンクに分割します。これは、最大入力長より短く、GPU RAM に十分な長さである必要があります。

```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

Apply the `group_texts` function over the entire dataset:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

次に、[`DataCollat​​orForLanguageModeling`] を使用してサンプルのバッチを作成します。 *動的にパディング*する方が効率的です。
データセット全体を最大長までパディングするのではなく、照合中にバッチ内の文を最長の長さにします。

<frameworkcontent>
<pt>

シーケンス終了トークンをパディング トークンとして使用し、`mlm=False` を設定します。これは、入力を 1 要素分右にシフトしたラベルとして使用します。

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

</pt>
<tf>
シーケンス終了トークンをパディング トークンとして使用し、`mlm=False` を設定します。これは、入力を 1 要素分右にシフトしたラベルとして使用します。

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
```

</tf>
</frameworkcontent>


## Train

<frameworkcontent>
<pt>
<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[基本チュートリアル](../training#train-with-pytorch-trainer) を参照してください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForCausalLM`] を使用して DistilGPT2 をロードします。


```py
>>> from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。
2. トレーニング引数をモデル、データセット、データ照合器とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_clm-model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.evaluate`] メソッドを使用してモデルを評価し、その複雑さを取得します。

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 49.61
```

次に、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

Keras を使用したモデルの微調整に慣れていない場合は、[基本チュートリアル](../training#train-a-tensorflow-model-with-keras) をご覧ください。

</Tip>
TensorFlow でモデルを微調整するには、オプティマイザー関数、学習率スケジュール、およびいくつかのトレーニング ハイパーパラメーターをセットアップすることから始めます。

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

次に、[`TFAutoModelForCausalLM`] を使用して DistilGPT2 をロードできます。

```py
>>> from transformers import TFAutoModelForCausalLM

>>> model = TFAutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`] を使用して、データセットを `tf.data.Dataset` 形式に変換します。

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

[`compile`](https://keras.io/api/models/model_training_apis/#compile-method) を使用してトレーニング用のモデルを設定します。 Transformers モデルにはすべてデフォルトのタスク関連の損失関数があるため、次の場合を除き、損失関数を指定する必要はないことに注意してください。

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

これは、モデルとトークナイザーを [`~transformers.PushToHubCallback`] でプッシュする場所を指定することで実行できます。



```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_clm-model",
...     tokenizer=tokenizer,
... )
```

ついに、モデルのトレーニングを開始する準備が整いました。トレーニングおよび検証データセット、エポック数、コールバックを指定して [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) を呼び出し、モデルを微調整します。



```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
```

トレーニングが完了すると、モデルは自動的にハブにアップロードされ、誰でも使用できるようになります。

</tf>
</frameworkcontent>

<Tip>

因果言語モデリング用にモデルを微調整する方法のより詳細な例については、対応するドキュメントを参照してください。
[PyTorch ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
または [TensorFlow ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。

</Tip>

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

テキストを生成するプロンプトを考え出します。

```py
>>> prompt = "Somatic hypermutation allows the immune system to"
```

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用してテキスト生成用の`pipeline`をインスタンス化し、それにテキストを渡します。


```py
>>> from transformers import pipeline

>>> generator = pipeline("text-generation", model="my_awesome_eli5_clm-model")
>>> generator(prompt)
[{'generated_text': "Somatic hypermutation allows the immune system to be able to effectively reverse the damage caused by an infection.\n\n\nThe damage caused by an infection is caused by the immune system's ability to perform its own self-correcting tasks."}]
```

<frameworkcontent>
<pt>


テキストをトークン化し、「input_ids」を PyTorch テンソルとして返します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_eli5_clm-model")
>>> inputs = tokenizer(prompt, return_tensors="pt").input_ids
```

[`~generation.GenerationMixin.generate`] メソッドを使用してテキストを生成します。
さまざまなテキスト生成戦略と生成を制御するためのパラメーターの詳細については、[テキスト生成戦略](../generation_strategies) ページを参照してください。

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

生成されたトークン ID をデコードしてテキストに戻します。

```py
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Somatic hypermutation allows the immune system to react to drugs with the ability to adapt to a different environmental situation. In other words, a system of 'hypermutation' can help the immune system to adapt to a different environmental situation or in some cases even a single life. In contrast, researchers at the University of Massachusetts-Boston have found that 'hypermutation' is much stronger in mice than in humans but can be found in humans, and that it's not completely unknown to the immune system. A study on how the immune system"]
```

</pt>
<tf>

テキストをトークン化し、`input_ids`を TensorFlow テンソルとして返します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_eli5_clm-model")
>>> inputs = tokenizer(prompt, return_tensors="tf").input_ids
```

[`~transformers.generation_tf_utils.TFGenerationMixin.generate`] メソッドを使用して要約を作成します。さまざまなテキスト生成戦略と生成を制御するためのパラメーターの詳細については、[テキスト生成戦略](../generation_strategies) ページを参照してください。

```py
>>> from transformers import TFAutoModelForCausalLM

>>> model = TFAutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
>>> outputs = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

生成されたトークン ID をデコードしてテキストに戻します。

```py
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Somatic hypermutation allows the immune system to detect the presence of other viruses as they become more prevalent. Therefore, researchers have identified a high proportion of human viruses. The proportion of virus-associated viruses in our study increases with age. Therefore, we propose a simple algorithm to detect the presence of these new viruses in our samples as a sign of improved immunity. A first study based on this algorithm, which will be published in Science on Friday, aims to show that this finding could translate into the development of a better vaccine that is more effective for']
```

</tf>
</frameworkcontent>
