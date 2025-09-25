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

# Semantic segmentation

[[open-in-colab]]

<Youtube id="dKE8SIt9C-w"/>

セマンティック セグメンテーションでは、画像の個々のピクセルにラベルまたはクラスを割り当てます。セグメンテーションにはいくつかのタイプがありますが、セマンティック セグメンテーションの場合、同じオブジェクトの一意のインスタンス間の区別は行われません。両方のオブジェクトに同じラベルが付けられます (たとえば、`car-1`と`car-2`の代わりに`car`)。セマンティック セグメンテーションの一般的な現実世界のアプリケーションには、歩行者や重要な交通情報を識別するための自動運転車のトレーニング、医療画像内の細胞と異常の識別、衛星画像からの環境変化の監視などが含まれます。

このガイドでは、次の方法を説明します。

1. [SceneParse150](https://huggingface.co/datasets/scene_parse_150) データセットの [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) を微調整します。
2. 微調整したモデルを推論に使用します。

<Tip>

このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/image-segmentation) を確認することをお勧めします。

</Tip>

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install -q datasets transformers evaluate
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SceneParse150 dataset

まず、SceneParse150 データセットの小さいサブセットを 🤗 データセット ライブラリから読み込みます。これにより、完全なデータセットのトレーニングにさらに時間を費やす前に、実験してすべてが機能することを確認する機会が得られます。

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

[`~datasets.Dataset.train_test_split`] メソッドを使用して、データセットの `train` 分割をトレイン セットとテスト セットに分割します。


```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

次に、例を見てみましょう。

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}
```

- `image`: シーンの PIL イメージ。
- `annotation`: セグメンテーション マップの PIL イメージ。モデルのターゲットでもあります。
- `scene_category`: "kitchen"や"office"などの画像シーンを説明するカテゴリ ID。このガイドでは、`image`と`annotation`のみが必要になります。どちらも PIL イメージです。

また、ラベル ID をラベル クラスにマップする辞書を作成することもできます。これは、後でモデルを設定するときに役立ちます。ハブからマッピングをダウンロードし、`id2label` および `label2id` ディクショナリを作成します。

```py
>>> import json
>>> from pathlib import Path
>>> from huggingface_hub import hf_hub_download

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

## Preprocess

次のステップでは、SegFormer 画像プロセッサをロードして、モデルの画像と注釈を準備します。このデータセットのような一部のデータセットは、バックグラウンド クラスとしてゼロインデックスを使用します。ただし、実際には背景クラスは 150 個のクラスに含まれていないため、`do_reduce_labels=True`を設定してすべてのラベルから 1 つを引く必要があります。ゼロインデックスは `255` に置き換えられるため、SegFormer の損失関数によって無視されます。

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
```


モデルを過学習に対してより堅牢にするために、画像データセットにいくつかのデータ拡張を適用するのが一般的です。このガイドでは、[torchvision](https://pytorch.org/vision/stable/index.html) の [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) 関数を使用します。 ) を使用して画像の色のプロパティをランダムに変更しますが、任意の画像ライブラリを使用することもできます。

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

次に、モデルの画像と注釈を準備するための 2 つの前処理関数を作成します。これらの関数は、画像を`pixel_values`に変換し、注釈を`labels`に変換します。トレーニング セットの場合、画像を画像プロセッサに提供する前に `jitter` が適用されます。テスト セットの場合、テスト中にデータ拡張が適用されないため、画像プロセッサは`images`を切り取って正規化し、`ラベル`のみを切り取ります。

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

データセット全体に`jitter`を適用するには、🤗 Datasets [`~datasets.Dataset.set_transform`] 関数を使用します。変換はオンザフライで適用されるため、高速で消費するディスク容量が少なくなります。

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```


## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用して、評価メソッドをすばやくロードできます。このタスクでは、[Mean Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy) (IoU) メトリックをロードします (🤗 Evaluate [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照して、メトリクスをロードして計算する方法の詳細を確認してください)。

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

次に、メトリクスを [`~evaluate.EvaluationModule.compute`] する関数を作成します。予測を次のように変換する必要があります
最初にロジットを作成し、次に [`~evaluate.EvaluationModule.compute`] を呼び出す前にラベルのサイズに一致するように再形成します。


```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if type(value) is np.ndarray:
...                 metrics[key] = value.tolist()
...         return metrics
```



これで`compute_metrics`関数の準備が整いました。トレーニングをセットアップするときにこの関数に戻ります。

## Train
<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[ここ](../training#finetune-with-trainer) の基本的なチュートリアルをご覧ください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForSemanticSegmentation`] を使用して SegFormer をロードし、ラベル ID とラベル クラス間のマッピングをモデルに渡します。

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。 `image` 列が削除されるため、未使用の列を削除しないことが重要です。 `image` 列がないと、`pixel_values` を作成できません。この動作を防ぐには、`remove_unused_columns=False`を設定してください。他に必要なパラメータは、モデルの保存場所を指定する `output_dir` だけです。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`Trainer`] は IoU メトリックを評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数を、モデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     eval_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

推論のために画像をロードします。

```py
>>> image = ds[0]["image"]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="Image of bedroom"/>
</div>


推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用して画像セグメンテーション用の `pipeline`をインスタンス化し、それに画像を渡します。

```py
>>> from transformers import pipeline

>>> segmenter = pipeline("image-segmentation", model="my_awesome_seg_model")
>>> segmenter(image)
[{'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062690>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A50>},
 {'score': None,
  'label': 'floor',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062B50>},
 {'score': None,
  'label': 'ceiling',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A10>},
 {'score': None,
  'label': 'bed ',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E90>},
 {'score': None,
  'label': 'windowpane',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062390>},
 {'score': None,
  'label': 'cabinet',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062550>},
 {'score': None,
  'label': 'chair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062D90>},
 {'score': None,
  'label': 'armchair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E10>}]
```

必要に応じて、`pipeline`の結果を手動で複製することもできます。画像を画像プロセッサで処理し、`pixel_values` を GPU に配置します。

```py
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

入力をモデルに渡し、`logits`を返します。


```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

次に、ロジットを元の画像サイズに再スケールします。

```py
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```


結果を視覚化するには、[データセット カラー パレット](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) を、それぞれをマップする `ade_palette()` としてロードします。クラスを RGB 値に変換します。次に、画像と予測されたセグメンテーション マップを組み合わせてプロットできます。

```py
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  # convert to BGR

>>> img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>
