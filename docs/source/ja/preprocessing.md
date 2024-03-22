<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdown形式ですが、特定のMDXに類似したドキュメントビルダーの構文を含んでおり、
Markdownビューアーで正しく表示されないことがあります。

-->

# Preprocess

[[open-in-colab]]

データセットでモデルをトレーニングする前に、それをモデルの期待する入力形式に前処理する必要があります。
データがテキスト、画像、またはオーディオであるかどうかにかかわらず、それらはテンソルのバッチに変換して組み立てる必要があります。
🤗 Transformersは、データをモデル用に準備するのに役立つ前処理クラスのセットを提供しています。
このチュートリアルでは、次のことを学びます：

* テキストの場合、[Tokenizer](./main_classes/tokenizer)を使用してテキストをトークンのシーケンスに変換し、トークンの数値表現を作成し、それらをテンソルに組み立てる方法。
* 音声とオーディオの場合、[Feature extractor](./main_classes/feature_extractor)を使用してオーディオ波形から連続的な特徴を抽出し、それらをテンソルに変換する方法。
* 画像入力の場合、[ImageProcessor](./main_classes/image)を使用して画像をテンソルに変換する方法。
* マルチモーダル入力の場合、[Processor](./main_classes/processors)を使用してトークナイザと特徴抽出器または画像プロセッサを組み合わせる方法。

<Tip>

`AutoProcessor`は常に動作し、使用するモデルに適切なクラスを自動的に選択します。
トークナイザ、画像プロセッサ、特徴抽出器、またはプロセッサを使用しているかにかかわらず、動作します。

</Tip>

始める前に、🤗 Datasetsをインストールして、いくつかのデータセットを試すことができるようにしてください：

```bash
pip install datasets
```

## Natural Language Processing

<Youtube id="Yffk5aydLzg"/>

テキストデータの前処理に使用する主要なツールは、[トークナイザ](main_classes/tokenizer)です。トークナイザは、一連のルールに従ってテキストを*トークン*に分割します。トークンは数値に変換され、その後テンソルに変換され、モデルの入力となります。モデルが必要とする追加の入力は、トークナイザによって追加されます。

<Tip>

事前学習済みモデルを使用する予定の場合、関連する事前学習済みトークナイザを使用することが重要です。これにより、テキストが事前学習コーパスと同じ方法で分割され、事前学習中に通常*ボキャブ*として参照される対応するトークンインデックスを使用します。

</Tip>

[`AutoTokenizer.from_pretrained`]メソッドを使用して事前学習済みトークナイザをロードして、開始しましょう。これにより、モデルが事前学習された*ボキャブ*がダウンロードされます：

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

次に、テキストをトークナイザに渡します：

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

トークナイザは、重要な3つの項目を持つ辞書を返します：

* [input_ids](glossary#input-ids) は文中の各トークンに対応するインデックスです。
* [attention_mask](glossary#attention-mask) はトークンがアテンションを受ける必要があるかどうかを示します。
* [token_type_ids](glossary#token-type-ids) は複数のシーケンスがある場合、トークンがどのシーケンスに属しているかを識別します。

`input_ids` をデコードして入力を返します：

```python
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] 魔法使いの事に干渉するな、彼らは微妙で怒りっぽい。 [SEP]'
```

如何にお分かりいただけるかと思いますが、トークナイザはこの文章に2つの特別なトークン、`CLS`（クラシファイア）と`SEP`（セパレータ）を追加しました。
すべてのモデルが特別なトークンを必要とするわけではありませんが、必要な場合、トークナイザは自動的にそれらを追加します。

複数の文章を前処理する場合、トークナイザにリストとして渡してください：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### Pad

文章は常に同じ長さではないことがあり、これはテンソル（モデルの入力）が均一な形状を持つ必要があるため問題となります。
パディングは、短い文に特別な「パディングトークン」を追加して、テンソルを長いシーケンスに合わせるための戦略です。

バッチ内の短いシーケンスを最長のシーケンスに合わせるために、`padding`パラメータを`True`に設定します：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

1番目と3番目の文は、短いために`0`でパディングされています。

### Truncation

逆のスペクトルでは、時折、モデルが処理するのに長すぎるシーケンスがあるかもしれません。この場合、シーケンスを短縮する必要があります。

モデルが受け入れる最大の長さにシーケンスを切り詰めるには、`truncation`パラメータを`True`に設定します：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

<Tip>

異なるパディングと切り詰めの引数について詳しくは、[パディングと切り詰め](./pad_truncation)のコンセプトガイドをご覧ください。

</Tip>

### Build tensors

最後に、トークナイザがモデルに供給される実際のテンソルを返すように設定します。

`return_tensors`パラメータを`pt`（PyTorch用）または`tf`（TensorFlow用）に設定します：

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>,
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>,
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

## Audio

オーディオタスクの場合、データセットをモデル用に準備するために[特徴抽出器](main_classes/feature_extractor)が必要です。
特徴抽出器は生のオーディオデータから特徴を抽出し、それらをテンソルに変換するために設計されています。

[PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)データセットをロードして（データセットのロード方法の詳細については🤗 [Datasetsチュートリアル](https://huggingface.co/docs/datasets/load_hub)を参照）、
オーディオデータセットで特徴抽出器をどのように使用できるかを確認してみましょう：

```python
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

アクセスして`audio`列の最初の要素を確認します。`audio`列を呼び出すと、自動的にオーディオファイルが読み込まれ、リサンプリングされます：

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

これにより、3つのアイテムが返されます：

* `array` は読み込まれた音声信号で、1Dの配列として読み込まれます。必要に応じてリサンプリングされることもあります。
* `path` は音声ファイルの場所を指します。
* `sampling_rate` は音声信号内のデータポイントが1秒間にいくつ測定されるかを示します。

このチュートリアルでは、[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)モデルを使用します。
モデルカードを確認すると、Wav2Vec2が16kHzのサンプリングされた音声オーディオで事前学習されていることがわかります。
モデルの事前学習に使用されたデータセットのサンプリングレートと、あなたのオーディオデータのサンプリングレートが一致することが重要です。
データのサンプリングレートが異なる場合、データをリサンプリングする必要があります。

1. 🤗 Datasetsの [`~datasets.Dataset.cast_column`] メソッドを使用して、サンプリングレートを16kHzにアップサンプリングします：

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. 再び `audio` 列を呼び出してオーディオファイルをリサンプルします：

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

次に、入力を正規化しパディングするために特徴抽出器をロードします。テキストデータをパディングする場合、短いシーケンスには `0` が追加されます。同じ考え方がオーディオデータにも適用されます。特徴抽出器は `array` に `0` を追加します（これは無音として解釈されます）。

[`AutoFeatureExtractor.from_pretrained`]を使用して特徴抽出器をロードします：

```python
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

オーディオ `array` を特徴抽出器に渡します。特徴抽出器で発生する可能性のある無音エラーをより良くデバッグするために、特徴抽出器に `sampling_rate` 引数を追加することをお勧めします。

```python
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

同様に、トークナイザと同様に、バッチ内の可変シーケンスを処理するためにパディングまたは切り詰めを適用できます。次に、これらの2つのオーディオサンプルのシーケンス長を確認してみましょう：

```python
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

この関数は、データセットを前処理してオーディオサンプルの長さを同じにするためのものです。最大サンプル長を指定し、特徴抽出器はシーケンスをそれに合わせてパディングまたは切り詰めます。

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

`preprocess_function`をデータセットの最初の数例に適用します：

```python
>>> processed_dataset = preprocess_function(dataset[:5])
```

サンプルの長さは現在同じで、指定された最大長と一致しています。これで処理されたデータセットをモデルに渡すことができます！

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## Computer Vision

コンピュータビジョンタスクでは、モデル用にデータセットを準備するための[画像プロセッサ](main_classes/image_processor)が必要です。
画像の前処理には、画像をモデルが期待する入力形式に変換するためのいくつかのステップが含まれています。これらのステップには、リサイズ、正規化、カラーチャネルの補正、および画像をテンソルに変換するなどが含まれます。

<Tip>

画像の前処理は、通常、画像の増強の形式に従います。画像の前処理と画像の増強の両方は画像データを変換しますが、異なる目的があります：

* 画像の増強は、過学習を防ぎ、モデルの堅牢性を向上させるのに役立つ方法で画像を変更します。データを増強する方法は無限で、明るさや色の調整、クロップ、回転、リサイズ、ズームなど、様々な方法があります。ただし、増強操作によって画像の意味が変わらないように注意する必要があります。
* 画像の前処理は、画像がモデルの期待する入力形式と一致することを保証します。コンピュータビジョンモデルをファインチューニングする場合、画像はモデルが最初にトレーニングされたときとまったく同じ方法で前処理する必要があります。

画像の増強には任意のライブラリを使用できます。画像の前処理には、モデルに関連付けられた`ImageProcessor`を使用します。

</Tip>

コンピュータビジョンのデータセットで画像プロセッサを使用する方法を示すために、[food101](https://huggingface.co/datasets/food101)データセットをロードします（データセットのロード方法の詳細については🤗[Datasetsチュートリアル](https://huggingface.co/docs/datasets/load_hub)を参照）：

<Tip>

データセットがかなり大きいため、🤗 Datasetsの`split`パラメータを使用してトレーニングデータの小さなサンプルのみをロードします！

</Tip>

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

次に、🤗 Datasetsの [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=image#datasets.Image) 機能で画像を見てみましょう：

```python
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

AutoImageProcessorを[`AutoImageProcessor.from_pretrained`]を使用してロードします：

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

1. まず、画像の拡張を追加しましょう。好きなライブラリを使用できますが、このチュートリアルではtorchvisionの[`transforms`](https://pytorch.org/vision/stable/transforms.html)モジュールを使用します。別のデータ拡張ライブラリを使用したい場合は、[Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb)または[Kornia notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)で詳細を学ぶことができます。

   ここでは、[`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html)を使用していくつかの変換を連鎖させます - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)と[`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)。
   サイズの変更に関しては、`image_processor`から画像サイズの要件を取得できます。
   一部のモデルでは、正確な高さと幅が必要ですが、他のモデルでは`shortest_edge`のみが定義されています。

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. モデルは[`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values)を入力として受け取ります。
`ImageProcessor`は画像の正規化と適切なテンソルの生成を処理できます。
一連の画像に対する画像拡張と画像前処理を組み合わせ、`pixel_values`を生成する関数を作成します：

```python
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

上記の例では、画像のサイズ変更を既に画像増強変換で行っているため、`do_resize=False`を設定しました。
適切な `image_processor` からの `size` 属性を活用しています。画像増強中に画像のサイズ変更を行わない場合は、このパラメータを省略してください。
デフォルトでは、`ImageProcessor` がサイズ変更を処理します。

画像を増強変換の一部として正規化したい場合は、`image_processor.image_mean` と `image_processor.image_std` の値を使用してください。
</Tip>

3. 次に、🤗 Datasetsの[`set_transform`](https://huggingface.co/docs/datasets/process#format-transform)を使用して、変換をリアルタイムで適用します：

```python
>>> dataset.set_transform(transforms)
```

4. 画像にアクセスすると、画像プロセッサが `pixel_values` を追加したことがわかります。これで処理済みのデータセットをモデルに渡すことができます！

```python
>>> dataset[0].keys()
```

以下は、変換が適用された後の画像の外観です。 画像はランダムに切り抜かれ、その色の特性も異なります。

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

オブジェクト検出、意味セグメンテーション、インスタンスセグメンテーション、およびパノプティックセグメンテーションなどのタスクの場合、`ImageProcessor`は
ポスト処理メソッドを提供します。これらのメソッドは、モデルの生の出力を境界ボックスやセグメンテーションマップなどの意味のある予測に変換します。

</Tip>

### Pad

一部の場合、たとえば、[DETR](./model_doc/detr)をファインチューニングする場合、モデルはトレーニング時にスケールの変更を適用します。
これにより、バッチ内の画像のサイズが異なる場合があります。[`DetrImageProcessor`]から[`DetrImageProcessor.pad`]を使用し、
カスタムの`collate_fn`を定義して画像を一緒にバッチ処理できます。

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## Multi Modal

マルチモーダル入力を使用するタスクの場合、モデル用にデータセットを準備するための[プロセッサ](main_classes/processors)が必要です。プロセッサは、トークナイザや特徴量抽出器などの2つの処理オブジェクトを結合します。

自動音声認識（ASR）のためのプロセッサの使用方法を示すために、[LJ Speech](https://huggingface.co/datasets/lj_speech)データセットをロードします（データセットのロード方法の詳細については🤗 [Datasets チュートリアル](https://huggingface.co/docs/datasets/load_hub)を参照）：

```python
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

ASR（自動音声認識）の場合、主に `audio` と `text` に焦点を当てているため、他の列を削除できます：

```python
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

次に、`audio`と`text`の列を見てみましょう：

```python
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

常に、オーディオデータセットのサンプリングレートを、モデルの事前学習に使用されたデータセットのサンプリングレートと一致させるように[リサンプル](preprocessing#audio)する必要があります！

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

プロセッサを [`AutoProcessor.from_pretrained`] を使用してロードします：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. `array`内に含まれるオーディオデータを`input_values`に処理し、`text`を`labels`にトークン化する関数を作成します：

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. サンプルに`prepare_dataset`関数を適用します：

```py
>>> prepare_dataset(lj_speech[0])
```
