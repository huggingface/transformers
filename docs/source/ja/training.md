<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-tune a pretrained model

[[open-in-colab]]

事前学習済みモデルを使用すると、計算コストを削減し、炭素排出量を減少させ、ゼロからモデルをトレーニングする必要なしに最新のモデルを使用できる利点があります。
🤗 Transformersは、さまざまなタスクに対応した数千もの事前学習済みモデルへのアクセスを提供します。
事前学習済みモデルを使用する場合、それを特定のタスクに合わせたデータセットでトレーニングします。これはファインチューニングとして知られ、非常に強力なトレーニング技術です。
このチュートリアルでは、事前学習済みモデルを選択したディープラーニングフレームワークでファインチューニングする方法について説明します：

* 🤗 Transformersの[`Trainer`]を使用して事前学習済みモデルをファインチューニングする。
* TensorFlowとKerasを使用して事前学習済みモデルをファインチューニングする。
* ネイティブのPyTorchを使用して事前学習済みモデルをファインチューニングする。

<a id='data-processing'></a>

## Prepare a dataset

<Youtube id="_BZearw7f0w"/>

事前学習済みモデルをファインチューニングする前に、データセットをダウンロードしてトレーニング用に準備する必要があります。
前のチュートリアルでは、トレーニングデータの処理方法を説明しましたが、これからはそれらのスキルを活かす機会があります！

まず、[Yelp Reviews](https://huggingface.co/datasets/yelp_review_full)データセットを読み込んでみましょう：

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

トークナイザがテキストを処理し、可変のシーケンス長を処理するためのパディングと切り捨て戦略を含める必要があることをご存知の通り、
データセットを1つのステップで処理するには、🤗 Datasets の [`map`](https://huggingface.co/docs/datasets/process#map) メソッドを使用して、
データセット全体に前処理関数を適用します：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)

>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

お好みで、実行時間を短縮するためにフルデータセットの小さなサブセットを作成することができます：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Train

この時点で、使用したいフレームワークに対応するセクションに従う必要があります。右側のサイドバーのリンクを使用して、ジャンプしたいフレームワークに移動できます。
そして、特定のフレームワークのすべてのコンテンツを非表示にしたい場合は、そのフレームワークのブロック右上にあるボタンを使用してください！

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## Train with Pytorch Trainer

🤗 Transformersは、🤗 Transformersモデルのトレーニングを最適化した[`Trainer`]クラスを提供し、独自のトレーニングループを手動で記述せずにトレーニングを開始しやすくしています。
[`Trainer`] APIは、ログ記録、勾配累積、混合精度など、さまざまなトレーニングオプションと機能をサポートしています。

まず、モデルをロードし、予想されるラベルの数を指定します。Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields)から、5つのラベルがあることがわかります：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

一部の事前学習済みの重みが使用されず、一部の重みがランダムに初期化された警告が表示されることがあります。心配しないでください、これは完全に正常です！
BERTモデルの事前学習済みのヘッドは破棄され、ランダムに初期化された分類ヘッドで置き換えられます。この新しいモデルヘッドをシーケンス分類タスクでファインチューニングし、事前学習モデルの知識をそれに転送します。

</Tip>

### Training Hyperparameters

次に、トレーニングオプションをアクティベートするためのすべてのハイパーパラメータと、調整できるハイパーパラメータを含む[`TrainingArguments`]クラスを作成します。
このチュートリアルでは、デフォルトのトレーニング[ハイパーパラメータ](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)を使用して開始できますが、最適な設定を見つけるためにこれらを実験しても構いません。

トレーニングのチェックポイントを保存する場所を指定します：

```python
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Evaluate

[`Trainer`]はトレーニング中に自動的にモデルのパフォーマンスを評価しません。メトリクスを計算して報告する関数を[`Trainer`]に渡す必要があります。
[🤗 Evaluate](https://huggingface.co/docs/evaluate/index)ライブラリでは、[`evaluate.load`]関数を使用して読み込むことができるシンプルな[`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy)関数が提供されています（詳細については[こちらのクイックツアー](https://huggingface.co/docs/evaluate/a_quick_tour)を参照してください）：

```python
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

`metric`の`~evaluate.compute`を呼び出して、予測の正確度を計算します。 `compute`に予測を渡す前に、予測をロジットに変換する必要があります（すべての🤗 Transformersモデルはロジットを返すことを覚えておいてください）：

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

評価メトリクスをファインチューニング中に監視したい場合、トレーニング引数で `eval_strategy` パラメータを指定して、各エポックの終了時に評価メトリクスを報告します：

```python
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### Trainer

モデル、トレーニング引数、トレーニングおよびテストデータセット、評価関数を使用して[`Trainer`]オブジェクトを作成します：

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

その後、[`~transformers.Trainer.train`]を呼び出してモデルを微調整します：

```python
>>> trainer.train()
```

</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## Kerasを使用してTensorFlowモデルをトレーニングする

Keras APIを使用して🤗 TransformersモデルをTensorFlowでトレーニングすることもできます！

### Loading Data from Keras

🤗 TransformersモデルをKeras APIでトレーニングする場合、データセットをKerasが理解できる形式に変換する必要があります。
データセットが小さい場合、データセット全体をNumPy配列に変換してKerasに渡すことができます。
複雑なことをする前に、まずそれを試してみましょう。

まず、データセットを読み込みます。GLUEベンチマークからCoLAデータセットを使用します
([GLUE Banchmark](https://huggingface.co/datasets/glue))、これは単純なバイナリテキスト分類タスクです。今のところトレーニング分割のみを使用します。

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # 今のところトレーニング分割のみを使用します
```

次に、トークナイザをロードし、データをNumPy配列としてトークン化します。ラベルは既に`0`と`1`のリストであるため、トークン化せずに直接NumPy配列に変換できます！

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# トークナイザはBatchEncodingを返しますが、それをKeras用に辞書に変換します
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # ラベルはすでに0と1の配列です
```

最後に、モデルをロードし、[`compile`](https://keras.io/api/models/model_training_apis/#compile-method) と [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) メソッドを実行します。
注意点として、Transformersモデルはすべてデフォルトでタスクに関連した損失関数を持っているため、指定しなくても構いません（指定する場合を除く）：

```python
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# モデルをロードしてコンパイルする
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# ファインチューニングには通常、学習率を下げると良いです
model.compile(optimizer=Adam(3e-5))  # 損失関数の指定は不要です！

model.fit(tokenized_data, labels)
```

<Tip>

モデルを`compile()`する際に`loss`引数を渡す必要はありません！Hugging Faceモデルは、この引数を空白のままにしておくと、タスクとモデルアーキテクチャに適した損失を自動的に選択します。
必要に応じて自分で損失を指定してオーバーライドすることもできます！

</Tip>

このアプローチは、小規模なデータセットには適していますが、大規模なデータセットに対しては問題になることがあります。なぜなら、トークナイズされた配列とラベルはメモリに完全に読み込まれる必要があり、またNumPyは「ジャギー」な配列を処理しないため、トークナイズされた各サンプルを全体のデータセット内で最も長いサンプルの長さにパディングする必要があります。
これにより、配列がさらに大きくなり、すべてのパディングトークンがトレーニングを遅くする原因になります！

### Loading data as a tf.data.Dataset

トレーニングを遅くせずにデータを読み込むには、データを`tf.data.Dataset`として読み込むことができます。独自の`tf.data`パイプラインを作成することもできますが、これを行うための便利な方法が2つあります：

- [`~TFPreTrainedModel.prepare_tf_dataset`]: これはほとんどの場合で推奨する方法です。モデル上のメソッドなので、モデルを検査してモデル入力として使用可能な列を自動的に把握し、他の列を破棄してより単純で高性能なデータセットを作成できます。
- [`~datasets.Dataset.to_tf_dataset`]: このメソッドはより低レベルで、データセットがどのように作成されるかを正確に制御する場合に便利です。`columns`と`label_cols`を指定して、データセットに含める列を正確に指定できます。

[`~TFPreTrainedModel.prepare_tf_dataset`]を使用する前に、次のコードサンプルに示すように、トークナイザの出力をデータセットに列として追加する必要があります：

```py
def tokenize_dataset(data):
    # 返された辞書のキーはデータセットに列として追加されます
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

Hugging Faceのデータセットはデフォルトでディスクに保存されるため、これによりメモリの使用量が増えることはありません！
列が追加されたら、データセットからバッチをストリームし、各バッチにパディングを追加できます。これにより、
データセット全体にパディングを追加する場合と比べて、パディングトークンの数が大幅に削減されます。

```python
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

上記のコードサンプルでは、トークナイザを`prepare_tf_dataset`に渡して、バッチを正しく読み込む際に正しくパディングできるようにする必要があります。
データセットのすべてのサンプルが同じ長さであり、パディングが不要な場合は、この引数をスキップできます。
パディング以外の複雑な処理を行う必要がある場合（例：マスク言語モデリングのためのトークンの破損など）、
代わりに`collate_fn`引数を使用して、サンプルのリストをバッチに変換し、必要な前処理を適用する関数を渡すことができます。
このアプローチを実際に使用した例については、
[examples](https://github.com/huggingface/transformers/tree/main/examples)や
[notebooks](https://huggingface.co/docs/transformers/notebooks)をご覧ください。

`tf.data.Dataset`を作成したら、以前と同様にモデルをコンパイルし、適合させることができます：

```python
model.compile(optimizer=Adam(3e-5))  # 損失引数は不要です！

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>

<a id='pytorch_native'></a>

## Train in native Pytorch

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`]はトレーニングループを処理し、1行のコードでモデルをファインチューニングできるようにします。
トレーニングループを独自に記述したいユーザーのために、🤗 TransformersモデルをネイティブのPyTorchでファインチューニングすることもできます。

この時点で、ノートブックを再起動するか、以下のコードを実行してメモリを解放する必要があるかもしれません：

```py
del model
del trainer
torch.cuda.empty_cache()
```

1. モデルは生のテキストを入力として受け取らないため、`text` 列を削除します：

```py
>>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
```

2. `label`列を`labels`に名前を変更します。モデルは引数の名前を`labels`と期待しています：

```py
>>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

3. データセットの形式をリストではなくPyTorchテンソルを返すように設定します：

```py
>>> tokenized_datasets.set_format("torch")
```

以前に示したように、ファインチューニングを高速化するためにデータセットの小さなサブセットを作成します：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

トレーニングデータセットとテストデータセット用の`DataLoader`を作成して、データのバッチをイテレートできるようにします：

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

ロードするモデルと期待されるラベルの数を指定してください：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Optimizer and learning rate scheduler

モデルをファインチューニングするためのオプティマイザと学習率スケジューラーを作成しましょう。
PyTorchから[`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)オプティマイザを使用します：

```python
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

デフォルトの学習率スケジューラを[`Trainer`]から作成する：

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

最後に、GPUを利用できる場合は `device` を指定してください。それ以外の場合、CPUでのトレーニングは数時間かかる可能性があり、数分で完了することができます。

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

クラウドGPUが利用できない場合、[Colaboratory](https://colab.research.google.com/)や[SageMaker StudioLab](https://studiolab.sagemaker.aws/)などのホストされたノートブックを使用して無料でGPUにアクセスできます。

</Tip>

さて、トレーニングの準備が整いました！ 🥳

### トレーニングループ

トレーニングの進捗を追跡するために、[tqdm](https://tqdm.github.io/)ライブラリを使用してトレーニングステップの数に対して進行状況バーを追加します：

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Evaluate

[`Trainer`]に評価関数を追加したのと同様に、独自のトレーニングループを作成する際にも同様の操作を行う必要があります。
ただし、各エポックの最後にメトリックを計算および報告する代わりに、今回は[`~evaluate.add_batch`]を使用してすべてのバッチを蓄積し、最後にメトリックを計算します。

```python
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## 追加リソース

さらなるファインチューニングの例については、以下を参照してください：

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) には、PyTorchとTensorFlowで一般的なNLPタスクをトレーニングするスクリプトが含まれています。

- [🤗 Transformers Notebooks](notebooks) には、特定のタスクにモデルをファインチューニングする方法に関するさまざまなノートブックが含まれています。
