<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Translation

[[open-in-colab]]

<Youtube id="1JvfrvZgi6c"/>

翻訳では、一連のテキストをある言語から別の言語に変換します。これは、シーケンス間問題として定式化できるいくつかのタスクの 1 つであり、翻訳や要約など、入力から何らかの出力を返すための強力なフレームワークです。翻訳システムは通常、異なる言語のテキスト間の翻訳に使用されますが、音声、またはテキストから音声への変換や音声からテキストへの変換など、音声間の組み合わせにも使用できます。

このガイドでは、次の方法を説明します。

1. [OPUS Books](https://huggingface.co/datasets/opus_books) データセットの英語-フランス語サブセットの [T5](https://huggingface.co/google-t5/t5-small) を微調整して、英語のテキストを次の形式に翻訳します。フランス語。
2. 微調整されたモデルを推論に使用します。

<Tip>

このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/translation) を確認することをお勧めします。

</Tip>

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install transformers datasets evaluate sacrebleu
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```
## Load OPUS Books dataset

まず、🤗 データセット ライブラリから [OPUS Books](https://huggingface.co/datasets/opus_books) データセットの英語とフランス語のサブセットを読み込みます。

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

[`~datasets.Dataset.train_test_split`] メソッドを使用して、データセットをトレイン セットとテスト セットに分割します。


```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

次に、例を見てみましょう。

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`: テキストの英語とフランス語の翻訳。

## Preprocess

<Youtube id="XAR8jnZZuUs"/>

次のステップでは、T5 トークナイザーをロードして英語とフランス語の言語ペアを処理します。

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

作成する前処理関数は次のことを行う必要があります。

1. T5 がこれが翻訳タスクであることを認識できるように、入力の前にプロンプ​​トを付けます。複数の NLP タスクが可能な一部のモデルでは、特定のタスクのプロンプトが必要です。
2. 英語の語彙で事前トレーニングされたトークナイザーを使用してフランス語のテキストをトークン化することはできないため、入力 (英語) とターゲット (フランス語) を別々にトークン化します。
3. `max_length`パラメータで設定された最大長を超えないようにシーケンスを切り詰めます。
```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.Dataset.map`] メソッドを使用します。 `batched=True` を設定してデータセットの複数の要素を一度に処理することで、`map` 関数を高速化できます。

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

次に、[`DataCollat​​orForSeq2Seq`] を使用してサンプルのバッチを作成します。データセット全体を最大長までパディングするのではなく、照合中にバッチ内の最長の長さまで文を *動的にパディング* する方が効率的です。


```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用して、評価メソッドをすばやくロードできます。このタスクでは、[SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) メトリクスをロードします (🤗 Evaluate [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照してください) ) メトリクスの読み込みと計算方法の詳細については、次を参照してください)。

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

次に、予測とラベルを [`~evaluate.EvaluationModule.compute`] に渡して SacreBLEU スコアを計算する関数を作成します。
```py
>>> import numpy as np


>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```
これで`compute_metrics`関数の準備が整いました。トレーニングをセットアップするときにこの関数に戻ります。

## Train

<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[ここ](../training#train-with-pytorch-trainer) の基本的なチュートリアルをご覧ください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForSeq2SeqLM`] を使用して T5 をロードします。

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

この時点で残っているステップは 3 つだけです。

1. [`Seq2SeqTrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`Trainer`] は SacreBLEU メトリクスを評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数をモデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Seq2SeqTrainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。


```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```

<Tip>

翻訳用にモデルを微調整する方法の詳細な例については、対応するドキュメントを参照してください。
[PyTorch ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)
または [TensorFlow ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)。

</Tip>

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

別の言語に翻訳したいテキストを考え出します。 T5 の場合、作業中のタスクに応じて入力に接頭辞を付ける必要があります。英語からフランス語に翻訳する場合は、以下に示すように入力に接頭辞を付ける必要があります。

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

テキストをトークン化し、`input_ids` を PyTorch テンソルとして返します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

[`~generation.GenerationMixin.generate`] メソッドを使用して翻訳を作成します。さまざまなテキスト生成戦略と生成を制御するためのパラメーターの詳細については、[Text Generation](../main_classes/text_generation) API を確認してください。


```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

生成されたトークン ID をデコードしてテキストに戻します。


```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```

