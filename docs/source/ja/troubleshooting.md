<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Troubleshoot

時にはエラーが発生することがありますが、私たちはここにいます！このガイドでは、私たちがよく見る最も一般的な問題と、それらを解決する方法について説明します。ただし、このガイドはすべての 🤗 Transformers の問題の包括的なコレクションではありません。問題をトラブルシューティングするための詳細なヘルプが必要な場合は、以下の方法を試してみてください：

<Youtube id="S2EEG3JIt2A"/>

1. [フォーラム](https://discuss.huggingface.co/)で助けを求める。 [初心者向け](https://discuss.huggingface.co/c/beginners/5) または [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9) など、質問を投稿できる特定のカテゴリがあります。問題が解決される可能性を最大限にするために、再現可能なコードを含む良い説明的なフォーラム投稿を書くことを確認してください！

<Youtube id="_PAli-V4wj0"/>

2. バグがライブラリに関連する場合は、🤗 Transformers リポジトリで [Issue](https://github.com/huggingface/transformers/issues/new/choose) を作成してください。バグを説明するためのできるだけ多くの情報を含めるように心がけ、何が問題で、どのように修正できるかをより良く理解できるようにしてください。

3. より古いバージョンの 🤗 Transformers を使用している場合は、[Migration](migration) ガイドを確認してください。バージョン間で重要な変更が導入されているためです。

トラブルシューティングとヘルプの詳細については、Hugging Faceコースの [第8章](https://huggingface.co/course/chapter8/1?fw=pt) を参照してください。

## Firewalled environments

一部のクラウド上のGPUインスタンスやイントラネットセットアップは、外部接続に対してファイアウォールで保護されているため、接続エラーが発生することがあります。スクリプトがモデルの重みやデータセットをダウンロードしようとすると、ダウンロードが途中で止まり、次のメッセージとタイムアウトエラーが表示されます：

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```


この場合、接続エラーを回避するために[オフラインモード](installation#offline-mode)で🤗 Transformersを実行してみてください。

## CUDA out of memory

数百万のパラメータを持つ大規模なモデルのトレーニングは、適切なハードウェアなしでは課題です。GPUのメモリが不足するとよくあるエラーの1つは次のとおりです：

以下はメモリ使用量を減らすために試すことができるいくつかの解決策です：

- [`TrainingArguments`]の中で [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) の値を減らす。
- [`TrainingArguments`]の中で [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) を使用して、全体的なバッチサイズを効果的に増やすことを試す。

<Tip>

メモリ節約のテクニックについての詳細は、[ガイド](performance)を参照してください。

</Tip>

## Unable to load a saved TensorFlow model

TensorFlowの[model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)メソッドは、モデル全体 - アーキテクチャ、重み、トレーニング設定 - を1つのファイルに保存します。しかし、モデルファイルを再度読み込む際にエラーが発生することがあります。これは、🤗 Transformersがモデルファイル内のすべてのTensorFlow関連オブジェクトを読み込まないためです。TensorFlowモデルの保存と読み込みに関する問題を回避するために、次のことをお勧めします：

- モデルの重みを`h5`ファイル拡張子で保存し、[`~TFPreTrainedModel.from_pretrained`]を使用してモデルを再読み込みする：

```py
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- Save the model with [`~TFPretrainedModel.save_pretrained`] and load it again with [`~TFPreTrainedModel.from_pretrained`]:

```py
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError

もう一つよくあるエラーは、特に新しくリリースされたモデルの場合に遭遇することがある `ImportError` です：


```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

これらのエラータイプに関しては、最新バージョンの 🤗 Transformers がインストールされていることを確認して、最新のモデルにアクセスできるようにしてください：

```bash
pip install transformers --upgrade
```

## CUDA error: device-side assert triggered

時々、デバイスコードでエラーが発生したという一般的な CUDA エラーに遭遇することがあります。

```
RuntimeError: CUDA error: device-side assert triggered
```

より具体的なエラーメッセージを取得するために、まずはCPU上でコードを実行してみることをお勧めします。以下の環境変数をコードの冒頭に追加して、CPUに切り替えてみてください：

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

GPUからより良いトレースバックを取得する別のオプションは、次の環境変数をコードの先頭に追加することです。これにより、エラーの発生源を指すトレースバックが得られます：

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```


## Incorrect output when padding tokens aren't masked

一部のケースでは、`input_ids`にパディングトークンが含まれている場合、出力の`hidden_state`が正しくないことがあります。デモンストレーションのために、モデルとトークナイザーをロードします。モデルの`pad_token_id`にアクセスして、その値を確認できます。一部のモデルでは`pad_token_id`が`None`になることもありますが、常に手動で設定することができます。


```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
>>> model.config.pad_token_id
0
```

以下の例は、パディングトークンをマスクせずに出力を表示したものです：

```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

以下は、第2のシーケンスの実際の出力です：

```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

大抵の場合、モデルには `attention_mask` を提供して、パディングトークンを無視し、このような無音のエラーを回避する必要があります。これにより、2番目のシーケンスの出力が実際の出力と一致するようになります。

<Tip>

デフォルトでは、トークナイザは、トークナイザのデフォルトに基づいて `attention_mask` を自動で作成します。

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗 Transformersは、提供されるパディングトークンをマスクするために自動的に`attention_mask`を作成しません。その理由は以下の通りです：

- 一部のモデルにはパディングトークンが存在しない場合があるためです。
- 一部のユースケースでは、ユーザーがパディングトークンにアテンションを向けることを望む場合があるためです。

## ValueError: Unrecognized configuration class XYZ for this kind of AutoModel

一般的に、事前学習済みモデルのインスタンスをロードするためには[`AutoModel`]クラスを使用することをお勧めします。このクラスは、設定に基づいて与えられたチェックポイントから正しいアーキテクチャを自動的に推測およびロードできます。モデルをロードする際にこの`ValueError`が表示される場合、Autoクラスは与えられたチェックポイントの設定から、ロードしようとしているモデルの種類へのマッピングを見つけることができなかったことを意味します。最も一般的には、特定のタスクをサポートしないチェックポイントがある場合にこのエラーが発生します。
例えば、質問応答のためのGPT2が存在しない場合、次の例でこのエラーが表示されます：

上記のテキストを日本語に翻訳し、Markdownファイルとしてフォーマットしました。


```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
