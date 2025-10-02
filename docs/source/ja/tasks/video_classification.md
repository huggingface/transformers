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

# Video classification

[[open-in-colab]]


ビデオ分類は、ビデオ全体にラベルまたはクラスを割り当てるタスクです。ビデオには、各ビデオに 1 つのクラスのみが含まれることが期待されます。ビデオ分類モデルはビデオを入力として受け取り、ビデオがどのクラスに属するかについての予測を返します。これらのモデルを使用して、ビデオの内容を分類できます。ビデオ分類の実際のアプリケーションはアクション/アクティビティ認識であり、フィットネス アプリケーションに役立ちます。また、視覚障害のある人にとって、特に通勤時に役立ちます。

このガイドでは、次の方法を説明します。

1. [UCF101](https://www.crcv.ucf.edu/) のサブセットで [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) を微調整します。 data/UCF101.php) データセット。
2. 微調整したモデルを推論に使用します。

> [!TIP]
> このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/video-classification) を確認することをお勧めします。

始める前に、必要なライブラリがすべてインストールされていることを確認してください。
```bash
pip install -q pytorchvideo transformers evaluate
```

[PyTorchVideo](https://pytorchvideo.org/) (`pytorchvideo` と呼ばれます) を使用してビデオを処理し、準備します。

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load UCF101 dataset

まず、[UCF-101 データセット](https://www.crcv.ucf.edu/data/UCF101.php) のサブセットをロードします。これにより、完全なデータセットのトレーニングにさらに時間を費やす前に、実験してすべてが機能することを確認する機会が得られます。

```py
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

サブセットをダウンロードした後、圧縮アーカイブを抽出する必要があります。

```py
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...      t.extractall(".")
```

大まかに言うと、データセットは次のように構成されています。

```bash
UCF101_subset/
    train/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    val/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    test/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
```

(`sorted`)された ビデオ パスは次のように表示されます。


```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

同じグループ/シーンに属するビデオ クリップがあり、ビデオ ファイル パスではグループが`g`で示されていることがわかります。たとえば、`v_ApplyEyeMakeup_g07_c04.avi`や`v_ApplyEyeMakeup_g07_c06.avi`などです。

検証と評価の分割では、[データ漏洩](https://www.kaggle.com/code/alexisbcook/data-leakage) を防ぐために、同じグループ/シーンからのビデオ クリップを使用しないでください。このチュートリアルで使用しているサブセットでは、この情報が考慮されています。

次に、データセット内に存在するラベルのセットを取得します。また、モデルを初期化するときに役立つ 2 つの辞書を作成します。

* `label2id`: クラス名を整数にマップします。
* `id2label`: 整数をクラス名にマッピングします。


```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# Unique classes: ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress'].
```

個性的なクラスが10種類あります。トレーニング セットには、クラスごとに 30 個のビデオがあります。

## Load a model to fine-tune

事前トレーニングされたチェックポイントとそれに関連する画像プロセッサからビデオ分類モデルをインスタンス化します。モデルのエンコーダーには事前トレーニングされたパラメーターが付属しており、分類ヘッドはランダムに初期化されます。画像プロセッサは、データセットの前処理パイプラインを作成するときに役立ちます。

```py
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
... )
```

モデルのロード中に、次の警告が表示される場合があります。

```bash
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of VideoMAEForVideoClassification were not initialized from the model checkpoint at MCG-NJU/videomae-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

この警告は、一部の重み (たとえば、`classifier`層の重みとバイアス) を破棄し、他のいくつかの重み (新しい`classifier`層の重みとバイアス) をランダムに初期化していることを示しています。この場合、これは予想されることです。事前にトレーニングされた重みを持たない新しい頭部を追加しているため、推論に使用する前にこのモデルを微調整する必要があるとライブラリが警告します。これはまさに私たちが行おうとしているものです。する。

**注意** [このチェックポイント](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) は、同様のダウンストリームで微調整されてチェックポイントが取得されたため、このタスクのパフォーマンスが向上することに注意してください。かなりのドメインの重複があるタスク。 `MCG-NJU/videomae-base-finetuned-kinetics` を微調整して取得した [このチェックポイント](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset) を確認できます。 -キネティクス`。

## Prepare the datasets for training

ビデオの前処理には、[PyTorchVideo ライブラリ](https://pytorchvideo.org/) を利用します。まず、必要な依存関係をインポートします。


```py
>>> import pytorchvideo.data

>>> from pytorchvideo.transforms import (
...     ApplyTransformToKey,
...     Normalize,
...     RandomShortSideScale,
...     RemoveKey,
...     ShortSideScale,
...     UniformTemporalSubsample,
... )

>>> from torchvision.transforms import (
...     Compose,
...     Lambda,
...     RandomCrop,
...     RandomHorizontalFlip,
...     Resize,
... )
```

トレーニング データセットの変換には、均一な時間サブサンプリング、ピクセル正規化、ランダム クロッピング、およびランダムな水平反転を組み合わせて使用​​します。検証および評価のデータセット変換では、ランダムなトリミングと水平反転を除き、同じ変換チェーンを維持します。これらの変換の詳細については、[PyTorchVideo の公式ドキュメント](https://pytorchvideo.org) を参照してください。

事前トレーニングされたモデルに関連付けられた`image_processor`を使用して、次の情報を取得します。

* ビデオ フレームのピクセルが正規化される画像の平均値と標準偏差。
* ビデオ フレームのサイズが変更される空間解像度。

まず、いくつかの定数を定義します。

```py
>>> mean = image_processor.image_mean
>>> std = image_processor.image_std
>>> if "shortest_edge" in image_processor.size:
...     height = width = image_processor.size["shortest_edge"]
>>> else:
...     height = image_processor.size["height"]
...     width = image_processor.size["width"]
>>> resize_to = (height, width)

>>> num_frames_to_sample = model.config.num_frames
>>> sample_rate = 4
>>> fps = 30
>>> clip_duration = num_frames_to_sample * sample_rate / fps
```

次に、データセット固有の変換とデータセットをそれぞれ定義します。トレーニングセットから始めます:


```py
>>> train_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     RandomShortSideScale(min_size=256, max_size=320),
...                     RandomCrop(resize_to),
...                     RandomHorizontalFlip(p=0.5),
...                 ]
...             ),
...         ),
...     ]
... )

>>> train_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "train"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
...     decode_audio=False,
...     transform=train_transform,
... )
```

同じ一連のワークフローを検証セットと評価セットに適用できます。


```py
>>> val_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     Resize(resize_to),
...                 ]
...             ),
...         ),
...     ]
... )

>>> val_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "val"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )

>>> test_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "test"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )
```

**注意**: 上記のデータセット パイプラインは、[公式 PyTorchVideo サンプル](https://pytorchvideo.org/docs/tutorial_classification#dataset) から取得したものです。 [`pytorchvideo.data.Ucf101()`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.Ucf101) 関数を使用しています。 UCF-101 データセット。内部では、[`pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.LabeledVideoDataset) オブジェクトを返します。 `LabeledVideoDataset` クラスは、PyTorchVideo データセット内のすべてのビデオの基本クラスです。したがって、PyTorchVideo で既製でサポートされていないカスタム データセットを使用したい場合は、それに応じて `LabeledVideoDataset` クラスを拡張できます。詳細については、`data`API [ドキュメント](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html)を参照してください。また、データセットが同様の構造 (上に示したもの) に従っている場合は、`pytorchvideo.data.Ucf101()` を使用すると問題なく動作するはずです。

`num_videos` 引数にアクセスすると、データセット内のビデオの数を知ることができます。



```py
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300, 30, 75)
```

## Visualize the preprocessed video for better debugging

```py
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """Un-normalizes the image pixels."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """Prepares a GIF from a video tensor.
...
...     The video tensor is expected to have the following shape:
...     (num_frames, num_channels, height, width).
...     """
...     frames = []
...     for video_frame in video_tensor:
...         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
...         frames.append(frame_unnormalized)
...     kargs = {"duration": 0.25}
...     imageio.mimsave(filename, frames, "GIF", **kargs)
...     return filename

>>> def display_gif(video_tensor, gif_name="sample.gif"):
...     """Prepares and displays a GIF from a video tensor."""
...     video_tensor = video_tensor.permute(1, 0, 2, 3)
...     gif_filename = create_gif(video_tensor, gif_name)
...     return Image(filename=gif_filename)

>>> sample_video = next(iter(train_dataset))
>>> video_tensor = sample_video["video"]
>>> display_gif(video_tensor)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif.gif" alt="Person playing basketball"/>
</div>

## Train the model

🤗 Transformers の [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) をモデルのトレーニングに利用します。 `Trainer`をインスタンス化するには、トレーニング構成と評価メトリクスを定義する必要があります。最も重要なのは [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments) で、これはトレーニングを構成するためのすべての属性を含むクラスです。モデルのチェックポイントを保存するために使用される出力フォルダー名が必要です。また、🤗 Hub 上のモデル リポジトリ内のすべての情報を同期するのにも役立ちます。

トレーニング引数のほとんどは一目瞭然ですが、ここで非常に重要なのは`remove_unused_columns=False`です。これにより、モデルの呼び出し関数で使用されない機能が削除されます。デフォルトでは`True`です。これは、通常、未使用の特徴列を削除し、モデルの呼び出し関数への入力を解凍しやすくすることが理想的であるためです。ただし、この場合、`pixel_values` (モデルが入力で期待する必須キーです) を作成するには、未使用の機能 (特に`video`) が必要です。

```py
>>> from transformers import TrainingArguments, Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name,
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=batch_size,
...     per_device_eval_batch_size=batch_size,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
... )
```

`pytorchvideo.data.Ucf101()` によって返されるデータセットは `__len__` メソッドを実装していません。そのため、`TrainingArguments`をインスタンス化するときに`max_steps`を定義する必要があります。

次に、予測からメトリクスを計算する関数を定義する必要があります。これは、これからロードする`metric`を使用します。必要な前処理は、予測されたロジットの argmax を取得することだけです。

```py
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

**評価に関する注意事項**:

[VideoMAE 論文](https://huggingface.co/papers/2203.12602) では、著者は次の評価戦略を使用しています。彼らはテスト ビデオからのいくつかのクリップでモデルを評価し、それらのクリップにさまざまなクロップを適用して、合計スコアを報告します。ただし、単純さと簡潔さを保つために、このチュートリアルではそれを考慮しません。

また、サンプルをまとめてバッチ処理するために使用される `collat​​e_fn` を定義します。各バッチは、`pixel_values` と `labels` という 2 つのキーで構成されます。


```py
>>> def collate_fn(examples):
...     # permute to (num_frames, num_channels, height, width)
...     pixel_values = torch.stack(
...         [example["video"].permute(1, 0, 2, 3) for example in examples]
...     )
...     labels = torch.tensor([example["label"] for example in examples])
...     return {"pixel_values": pixel_values, "labels": labels}
```

次に、これらすべてをデータセットとともに`Trainer`に渡すだけです。

```py
>>> trainer = Trainer(
...     model,
...     args,
...     train_dataset=train_dataset,
...     eval_dataset=val_dataset,
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
...     data_collator=collate_fn,
... )
```

すでにデータを前処理しているのに、なぜトークナイザーとして`image_processor`を渡したのか不思議に思うかもしれません。これは、イメージ プロセッサ構成ファイル (JSON として保存) もハブ上のリポジトリにアップロードされるようにするためだけです。

次に、`train` メソッドを呼び出してモデルを微調整します。

```py
>>> train_results = trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

推論のためにビデオをロードします。

```py
>>> sample_test_video = next(iter(test_dataset))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif" alt="Teams playing basketball"/>
</div>

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline). で使用することです。モデルを使用してビデオ分類用の` pipeline`をインスタンス化し、それにビデオを渡します。


```py
>>> from transformers import pipeline

>>> video_cls = pipeline(model="my_awesome_video_cls_model")
>>> video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
[{'score': 0.9272987842559814, 'label': 'BasketballDunk'},
 {'score': 0.017777055501937866, 'label': 'BabyCrawling'},
 {'score': 0.01663011871278286, 'label': 'BalanceBeam'},
 {'score': 0.009560945443809032, 'label': 'BandMarching'},
 {'score': 0.0068979403004050255, 'label': 'BaseballPitch'}]
```

必要に応じて、`pipeline`の結果を手動で複製することもできます。

```py
>>> def run_inference(model, video):
...     # (num_frames, num_channels, height, width)
...     perumuted_sample_test_video = video.permute(1, 0, 2, 3)
...     inputs = {
...         "pixel_values": perumuted_sample_test_video.unsqueeze(0),
...         "labels": torch.tensor(
...             [sample_test_video["label"]]
...         ),  # this can be skipped if you don't have labels available.
...     }

...     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...     inputs = {k: v.to(device) for k, v in inputs.items()}
...     model = model.to(device)

...     # forward pass
...     with torch.no_grad():
...         outputs = model(**inputs)
...         logits = outputs.logits

...     return logits
```

次に、入力をモデルに渡し、`logits `を返します。

```py
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

`logits` をデコードすると、次のようになります。

```py
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```
