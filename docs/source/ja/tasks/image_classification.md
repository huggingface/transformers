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


# Image classification

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

画像分類では、画像にラベルまたはクラスを割り当てます。テキストや音声の分類とは異なり、入力は
画像を構成するピクセル値。損傷の検出など、画像分類には多くの用途があります
自然災害の後、作物の健康状態を監視したり、病気の兆候がないか医療画像をスクリーニングしたりするのに役立ちます。

このガイドでは、次の方法を説明します。

1. [Food-101](https://huggingface.co/datasets/food101) データセットの [ViT](model_doc/vit) を微調整して、画像内の食品を分類します。
2. 微調整したモデルを推論に使用します。

<Tip>

このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/image-classification) を確認することをお勧めします。

</Tip>

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install transformers datasets evaluate
```

Hugging Face アカウントにログインして、モデルをアップロードしてコミュニティと共有することをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load Food-101 dataset

Datasets、🤗 データセット ライブラリから Food-101 データセットの小さいサブセットを読み込みます。これにより、次の機会が得られます
完全なデータセットのトレーニングにさらに時間を費やす前に、実験してすべてが機能することを確認してください。

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

[`~datasets.Dataset.train_test_split`] メソッドを使用して、データセットの `train` 分割をトレイン セットとテスト セットに分割します。


```py
>>> food = food.train_test_split(test_size=0.2)
```

次に、例を見てみましょう。

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

データセット内の各例には 2 つのフィールドがあります。

- `image`: 食品の PIL 画像
- `label`: 食品のラベルクラス

モデルがラベル ID からラベル名を取得しやすくするために、ラベル名をマップする辞書を作成します。
整数への変換、またはその逆:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

これで、ラベル ID をラベル名に変換できるようになりました。

```py
>>> id2label[str(79)]
'prime_rib'
```

## Preprocess

次のステップでは、ViT 画像プロセッサをロードして画像をテンソルに処理します。

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```


<frameworkcontent>
<pt>

いくつかの画像変換を画像に適用して、モデルの過学習に対する堅牢性を高めます。ここでは torchvision の [`transforms`](https://pytorch.org/vision/stable/transforms.html) モジュールを使用しますが、任意の画像ライブラリを使用することもできます。

画像のランダムな部分をトリミングし、サイズを変更し、画像の平均と標準偏差で正規化します。


```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

次に、変換を適用し、画像の `pixel_values` (モデルへの入力) を返す前処理関数を作成します。

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.Dataset.with_transform`] メソッドを使用します。変換は、データセットの要素を読み込むときにオンザフライで適用されます。

```py
>>> food = food.with_transform(transforms)
```

次に、[`DefaultDataCollat​​or`] を使用してサンプルのバッチを作成します。 🤗 Transformers の他のデータ照合器とは異なり、`DefaultDataCollat​​or` はパディングなどの追加の前処理を適用しません。

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
</frameworkcontent>

## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。すぐにロードできます
🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用した評価方法。このタスクでは、ロードします
[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 指標 (詳細については、🤗 評価 [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照してくださいメトリクスをロードして計算する方法):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

次に、予測とラベルを [`~evaluate.EvaluationModule.compute`] に渡して精度を計算する関数を作成します。

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

これで `compute_metrics`関数の準備が整いました。トレーニングを設定するときにこの関数に戻ります。

## Train

<frameworkcontent>
<pt>
<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[こちら](../training#train-with-pytorch-trainer) の基本的なチュートリアルをご覧ください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForImageClassification`] を使用して ViT をロードします。ラベルの数と予想されるラベルの数、およびラベル マッピングを指定します。

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

この時点で残っているステップは 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。 `image` 列が削除されるため、未使用の列を削除しないことが重要です。 `image` 列がないと、`pixel_values` を作成できません。この動作を防ぐには、`remove_unused_columns=False`を設定してください。他に必要なパラメータは、モデルの保存場所を指定する `output_dir` だけです。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`Trainer`] は精度を評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数を、モデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

画像分類用のモデルを微調整する方法の詳細な例については、対応する [PyTorch ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)

</Tip>

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

推論を実行したい画像を読み込みます。

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/>
</div>

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用して画像分類用の`pipeline`をインスタンス化し、それに画像を渡します。

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

必要に応じて、`pipeline`の結果を手動で複製することもできます。

<frameworkcontent>
<pt>

画像プロセッサをロードして画像を前処理し、`input`を PyTorch テンソルとして返します。

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

入力をモデルに渡し、ロジットを返します。

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

最も高い確率で予測されたラベルを取得し、モデルの `id2label` マッピングを使用してラベルに変換します。

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt>
</frameworkcontent>
