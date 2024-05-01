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

# Create a custom architecture

[`AutoClass`](model_doc/auto)は、モデルのアーキテクチャを自動的に推論し、事前学習済みの設定と重みをダウンロードします。一般的には、チェックポイントに依存しないコードを生成するために`AutoClass`を使用することをお勧めします。ただし、特定のモデルパラメータに対する制御をより詳細に行いたいユーザーは、いくつかの基本クラスからカスタム🤗 Transformersモデルを作成できます。これは、🤗 Transformersモデルを研究、トレーニング、または実験する興味があるユーザーに特に役立つかもしれません。このガイドでは、`AutoClass`を使用しないカスタムモデルの作成について詳しく説明します。次の方法を学びます：

- モデルの設定をロードおよびカスタマイズする。
- モデルアーキテクチャを作成する。
- テキスト用の遅いトークナイザと高速トークナイザを作成する。
- ビジョンタスク用の画像プロセッサを作成する。
- オーディオタスク用の特徴抽出器を作成する。
- マルチモーダルタスク用のプロセッサを作成する。

## Configuration

[設定](main_classes/configuration)は、モデルの特定の属性を指します。各モデルの設定には異なる属性があります。たとえば、すべてのNLPモデルには、`hidden_size`、`num_attention_heads`、`num_hidden_layers`、および`vocab_size`属性が共通してあります。これらの属性は、モデルを構築するための注意ヘッドの数や隠れ層の数を指定します。

[DistilBERT](model_doc/distilbert)をより詳しく調べるために、[`DistilBertConfig`]にアクセスしてその属性を調べてみましょう：

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`]は、基本の[`DistilBertModel`]を構築するために使用されるすべてのデフォルト属性を表示します。
すべての属性はカスタマイズ可能で、実験のためのスペースを提供します。例えば、デフォルトのモデルをカスタマイズして以下のようなことができます：

- `activation`パラメータで異なる活性化関数を試す。
- `attention_dropout`パラメータで注意確率の高いドロップアウト率を使用する。


```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

事前学習済みモデルの属性は、[`~PretrainedConfig.from_pretrained`] 関数で変更できます：

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

Once you are satisfied with your model configuration, you can save it with [`PretrainedConfig.save_pretrained`]. Your configuration file is stored as a JSON file in the specified save directory.

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

設定ファイルを再利用するには、[`~PretrainedConfig.from_pretrained`]を使用してそれをロードします：

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

カスタム構成ファイルを辞書として保存することも、カスタム構成属性とデフォルトの構成属性の違いだけを保存することもできます！詳細については[configuration](main_classes/configuration)のドキュメンテーションをご覧ください。

</Tip>

## Model

次のステップは、[モデル](main_classes/models)を作成することです。モデル（アーキテクチャとも緩く言われることがあります）は、各レイヤーが何をしているか、どの操作が行われているかを定義します。構成からの `num_hidden_layers` のような属性はアーキテクチャを定義するために使用されます。
すべてのモデルは [`PreTrainedModel`] をベースクラスとし、入力埋め込みのリサイズやセルフアテンションヘッドのプルーニングなど、共通のメソッドがいくつかあります。
さらに、すべてのモデルは [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)、[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)、または [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) のいずれかのサブクラスでもあります。つまり、モデルはそれぞれのフレームワークの使用法と互換性があります。

<frameworkcontent>
<pt>
モデルにカスタム構成属性をロードします：

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

これにより、事前トレーニング済みの重みではなくランダムな値を持つモデルが作成されます。
これは、トレーニングが行われるまで、まだ有用なものとして使用することはできません。
トレーニングはコストと時間がかかるプロセスです。
通常、トレーニングに必要なリソースの一部しか使用せず、より速くより良い結果を得るために事前学習済みモデルを使用することが良いでしょう。

[`~PreTrainedModel.from_pretrained`]を使用して事前学習済みモデルを作成します：


```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

事前学習済みの重みをロードする際、モデルが🤗 Transformersによって提供されている場合、デフォルトのモデル設定が自動的にロードされます。ただし、必要に応じてデフォルトのモデル設定属性の一部またはすべてを独自のもので置き換えることができます。

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```
</pt>
<tf>
モデルにカスタム設定属性をロードしてください：

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> tf_model = TFDistilBertModel(my_config)
```

これにより、事前学習済みの重みではなくランダムな値を持つモデルが作成されます。
このモデルを有用な目的にはまだ使用することはできません。トレーニングはコストがかかり、時間がかかるプロセスです。
一般的には、トレーニングに必要なリソースの一部しか使用せずに、より速く優れた結果を得るために事前学習済みモデルを使用することが良いでしょう。

[`~TFPreTrainedModel.from_pretrained`]を使用して事前学習済みモデルを作成します：


```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

事前学習済みの重みをロードする際、モデルが🤗 Transformersによって提供されている場合、デフォルトのモデル構成が自動的にロードされます。ただし、必要であればデフォルトのモデル構成属性の一部またはすべてを独自のもので置き換えることもできます：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```
</tf>
</frameworkcontent>


### Model heads

この時点で、ベースのDistilBERTモデルがあり、これは隠れた状態を出力します。隠れた状態はモデルのヘッドへの入力として渡され、最終的な出力を生成します。🤗 Transformersは、モデルがそのタスクをサポートしている限り、各タスクに対応する異なるモデルヘッドを提供します（つまり、DistilBERTを翻訳のようなシーケンス対シーケンスタスクに使用することはできません）。

<frameworkcontent>
<pt>
たとえば、[`DistilBertForSequenceClassification`]は、シーケンス分類ヘッドを持つベースのDistilBERTモデルです。シーケンス分類ヘッドは、プールされた出力の上にある線形層です。

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

新しいタスクにこのチェックポイントを簡単に再利用するには、異なるモデルヘッドに切り替えます。
質問応答タスクの場合、[`DistilBertForQuestionAnswering`] モデルヘッドを使用します。
質問応答ヘッドはシーケンス分類ヘッドと類似していますが、隠れ状態の出力の上に線形層があります。

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

</pt>
<tf>
例えば、[`TFDistilBertForSequenceClassification`]は、シーケンス分類ヘッドを持つベースのDistilBERTモデルです。シーケンス分類ヘッドは、プールされた出力の上にある線形層です。

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

別のタスクにこのチェックポイントを簡単に再利用することができ、異なるモデルヘッドに切り替えるだけです。
質問応答タスクの場合、[`TFDistilBertForQuestionAnswering`]モデルヘッドを使用します。
質問応答ヘッドはシーケンス分類ヘッドと似ていますが、隠れ状態の出力の上に線形層があるだけです。


```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## Tokenizer

テキストデータをモデルで使用する前に必要な最後のベースクラスは、生のテキストをテンソルに変換するための[トークナイザ](main_classes/tokenizer)です。
🤗 Transformersで使用できる2つのタイプのトークナイザがあります：

- [`PreTrainedTokenizer`]: トークナイザのPython実装です。
- [`PreTrainedTokenizerFast`]: Rustベースの[🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/)ライブラリからのトークナイザです。
このトークナイザのタイプは、そのRust実装により、特にバッチトークナイゼーション中に高速です。
高速なトークナイザは、トークンを元の単語または文字にマッピングする*オフセットマッピング*などの追加メソッドも提供します。

両方のトークナイザは、エンコードとデコード、新しいトークンの追加、特別なトークンの管理など、共通のメソッドをサポートしています。

<Tip warning={true}>

すべてのモデルが高速なトークナイザをサポートしているわけではありません。
モデルが高速なトークナイザをサポートしているかどうかを確認するには、この[表](index#supported-frameworks)をご覧ください。

</Tip>

独自のトークナイザをトレーニングした場合、*ボキャブラリー*ファイルからトークナイザを作成できます。

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

カスタムトークナイザーから生成される語彙は、事前学習済みモデルのトークナイザーが生成する語彙とは異なることを覚えておくことは重要です。
事前学習済みモデルを使用する場合は、事前学習済みモデルの語彙を使用する必要があります。そうしないと、入力が意味をなさなくなります。
[`DistilBertTokenizer`]クラスを使用して、事前学習済みモデルの語彙を持つトークナイザーを作成します:


```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

[`DistilBertTokenizerFast`]クラスを使用して高速なトークナイザを作成します：

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip>

デフォルトでは、[`AutoTokenizer`]は高速なトークナイザを読み込もうとします。`from_pretrained`内で`use_fast=False`を設定することで、この動作を無効にすることができます。

</Tip>

## Image Processor

画像プロセッサはビジョン入力を処理します。これは基本クラス [`~image_processing_utils.ImageProcessingMixin`] を継承しています。

使用するには、使用しているモデルに関連付けられた画像プロセッサを作成します。
たとえば、画像分類に[ViT](model_doc/vit)を使用する場合、デフォルトの [`ViTImageProcessor`] を作成します。

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

カスタマイズを必要としない場合、モデルのデフォルトの画像プロセッサパラメータをロードするには、単純に`from_pretrained`メソッドを使用してください。

</Tip>

[`ViTImageProcessor`]のパラメータを変更して、カスタムの画像プロセッサを作成できます：


```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

## Feature Extractor

フィーチャー抽出器は音声入力を処理します。これは基本的な [`~feature_extraction_utils.FeatureExtractionMixin`] クラスから継承され、音声入力を処理するための [`SequenceFeatureExtractor`] クラスからも継承されることがあります。

使用するには、モデルに関連付けられたフィーチャー抽出器を作成します。たとえば、音声分類に [Wav2Vec2](model_doc/wav2vec2) を使用する場合、デフォルトの [`Wav2Vec2FeatureExtractor`] を作成します。


```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

<Tip>

カスタマイズを行わない場合、モデルのデフォルトの特徴抽出器パラメーターをロードするには、単に `from_pretrained` メソッドを使用してください。

</Tip>

[`Wav2Vec2FeatureExtractor`] のパラメーターを変更して、カスタム特徴抽出器を作成できます:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor(sampling_rate=8000, do_normalize=False)
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": false,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 8000
}
```

## Processor

マルチモーダルタスクをサポートするモデルに対して、🤗 Transformersは便利なプロセッサクラスを提供しています。
このプロセッサクラスは、特徴量抽出器やトークナイザなどの処理クラスを便利にラップし、単一のオブジェクトに結合します。
たとえば、自動音声認識タスク（ASR）用に[`Wav2Vec2Processor`]を使用してみましょう。
ASRは音声をテキストに転写するタスクであり、音声入力を処理するために特徴量抽出器とトークナイザが必要です。

音声入力を処理する特徴量抽出器を作成します：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

テキスト入力を処理するトークナイザを作成します:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

[`Wav2Vec2Processor`]で特徴量抽出器とトークナイザを組み合わせます：


```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

二つの基本クラス - 設定とモデル - および追加の前処理クラス（トークナイザ、画像プロセッサ、特徴抽出器、またはプロセッサ）を使用することで、🤗 Transformers がサポートするモデルのいずれかを作成できます。これらの基本クラスは設定可能で、必要な特性を使用できます。モデルをトレーニング用に簡単にセットアップしたり、既存の事前学習済みモデルを微調整することができます。
