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

# Sharing custom models

🤗 Transformersライブラリは、簡単に拡張できるように設計されています。すべてのモデルはリポジトリの特定のサブフォルダに完全にコード化されており、抽象化はありません。したがって、モデリングファイルをコピーして調整することが簡単です。

新しいモデルを書いている場合、ゼロから始める方が簡単かもしれません。このチュートリアルでは、カスタムモデルとその設定をどのように書き、Transformers内で使用できるようにし、コードに依存する共同体と共有する方法を説明します。ライブラリに存在しない場合でも、誰でも使用できるようにします。

これを実証するために、[timmライブラリ](https://github.com/rwightman/pytorch-image-models)のResNetクラスを[`PreTrainedModel`]にラップすることによって、ResNetモデルを使用します。

## Writing a custom configuration

モデルに取り組む前に、まずその設定を書きましょう。モデルの設定は、モデルを構築するために必要なすべての情報を含むオブジェクトです。次のセクションで見るように、モデルは初期化するために`config`しか受け取ることができないため、そのオブジェクトができるだけ完全である必要があります。

この例では、ResNetクラスのいくつかの引数を取得し、調整したいかもしれないとします。異なる設定は、異なるタイプのResNetを提供します。その後、これらの引数を確認した後、それらの引数を単に格納します。

```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

重要なことを3つ覚えておくべきポイントは次のとおりです：
- `PretrainedConfig` を継承する必要があります。
- あなたの `PretrainedConfig` の `__init__` は任意の kwargs を受け入れる必要があります。
- これらの `kwargs` は親クラスの `__init__` に渡す必要があります。

継承は、🤗 Transformers ライブラリのすべての機能を取得できるようにするためです。他の2つの制約は、
`PretrainedConfig` が設定しているフィールド以外にも多くのフィールドを持っていることから来ています。
`from_pretrained` メソッドで設定を再ロードする場合、これらのフィールドはあなたの設定に受け入れられ、
その後、親クラスに送信される必要があります。

設定の `model_type` を定義すること（ここでは `model_type="resnet"`）は、
自動クラスにモデルを登録したい場合を除いては必須ではありません（最後のセクションを参照）。

これで、ライブラリの他のモデル設定と同様に、設定を簡単に作成して保存できます。
以下は、resnet50d 設定を作成して保存する方法の例です：


```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

これにより、`custom-resnet` フォルダ内に `config.json` という名前のファイルが保存されます。その後、`from_pretrained` メソッドを使用して構成を再ロードできます。


```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

また、[`PretrainedConfig`] クラスの他のメソッドを使用することもできます。たとえば、[`~PretrainedConfig.push_to_hub`] を使用して、設定を直接 Hub にアップロードできます。

## Writing a custom model

ResNet の設定ができたので、モデルを書き始めることができます。実際には2つのモデルを書きます。1つはバッチの画像から隠れた特徴を抽出するモデル（[`BertModel`] のようなもの）で、もう1つは画像分類に適したモデル（[`BertForSequenceClassification`] のようなもの）です。

前述したように、この例をシンプルに保つために、モデルの緩いラッパーのみを書きます。このクラスを書く前に行う必要がある唯一のことは、ブロックタイプと実際のブロッククラスの間のマップです。その後、すべてを `ResNet` クラスに渡して設定からモデルを定義します：

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

画像を分類するモデルの場合、forwardメソッドを変更するだけです：

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

両方の場合、`PreTrainedModel`から継承し、`config`を使用してスーパークラスの初期化を呼び出します（通常の`torch.nn.Module`を書くときのような感じです）。
`config_class`を設定する行は必須ではありませんが、（最後のセクションを参照）、モデルを自動クラスに登録したい場合に使用できます。

<Tip>

モデルがライブラリ内のモデルと非常に似ている場合、このモデルと同じ構成を再利用できます。

</Tip>

モデルが返す内容は何でも構いませんが、ラベルが渡されるときに損失を含む辞書を返す（`ResnetModelForImageClassification`のように行ったもの）と、
モデルを[`Trainer`]クラス内で直接使用できるようになります。独自のトレーニングループまたは他のライブラリを使用する予定である限り、
別の出力形式を使用することも問題ありません。

さて、モデルクラスができたので、1つ作成しましょう：

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

再度、[`PreTrainedModel`]のいずれかのメソッド、例えば[`~PreTrainedModel.save_pretrained`]や
[`~PreTrainedModel.push_to_hub`]などを使用できます。次のセクションでは、モデルの重みをコードと一緒に
Hugging Face Hub にプッシュする方法を見てみます。
しかし、まずはモデル内に事前学習済みの重みをロードしましょう。

独自のユースケースでは、おそらく独自のデータでカスタムモデルをトレーニングすることになるでしょう。
このチュートリアルではスピードアップのために、resnet50dの事前学習済みバージョンを使用します。
私たちのモデルはそれをラップするだけなので、これらの重みを転送するのは簡単です：


```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

さて、[`~PreTrainedModel.save_pretrained`]または[`~PreTrainedModel.push_to_hub`]を実行したときに、
モデルのコードが保存されるようにする方法を見てみましょう。

## Sending the code to the Hub

<Tip warning={true}>

このAPIは実験的であり、次のリリースでわずかな変更があるかもしれません。

</Tip>

まず、モデルが`.py`ファイルに完全に定義されていることを確認してください。
ファイルは相対インポートを他のファイルに依存できますが、すべてのファイルが同じディレクトリにある限り（まだこの機能ではサブモジュールはサポートしていません）、問題ありません。
この例では、現在の作業ディレクトリ内に名前が「resnet_model」のフォルダを作成し、その中に`modeling_resnet.py`ファイルと`configuration_resnet.py`ファイルを定義します。
構成ファイルには`ResnetConfig`のコードが含まれ、モデリングファイルには`ResnetModel`と`ResnetModelForImageClassification`のコードが含まれています。


```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

`__init__.py`は空であっても問題ありません。Pythonが`resnet_model`をモジュールとして検出できるようにするために存在します。

<Tip warning={true}>

ライブラリからモデリングファイルをコピーする場合、ファイルの先頭にあるすべての相対インポートを`transformers`パッケージからインポートに置き換える必要があります。

</Tip>

既存の設定やモデルを再利用（またはサブクラス化）できることに注意してください。

コミュニティとモデルを共有するために、次の手順に従ってください：まず、新しく作成したファイルからResNetモデルと設定をインポートします：


```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

次に、`save_pretrained`メソッドを使用してこれらのオブジェクトのコードファイルをコピーし、特定のAutoクラス（特にモデルの場合）に正しく登録するようライブラリに指示する必要があります。次のように実行します：

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

注意: 設定については自動クラスを指定する必要はありません（設定用の自動クラスは1つしかなく、[`AutoConfig`]です）が、
モデルについては異なります。カスタムモデルは多くの異なるタスクに適している可能性があるため、
モデルが正確な自動クラスのうちどれに適しているかを指定する必要があります。

次に、前述のように設定とモデルを作成しましょう：

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

モデルをHubに送信するには、ログインしていることを確認してください。ターミナルで次のコマンドを実行します：

```bash
huggingface-cli login
```

またはノートブックから：

```py
from huggingface_hub import notebook_login

notebook_login()
```

次に、次のようにして、独自の名前空間にプッシュできます（または、メンバーである組織にプッシュできます）：

```py
resnet50d.push_to_hub("custom-resnet50d")
```

モデリングの重みとJSON形式の構成に加えて、このフォルダー「custom-resnet50d」内のモデリングおよび構成「.py」ファイルもコピーされ、結果はHubにアップロードされました。結果はこの[model repo](https://huggingface.co/sgugger/custom-resnet50d)で確認できます。

詳細については、[Hubへのプッシュ方法](model_sharing)を参照してください。

## Using a model with custom code

自動クラスと `from_pretrained` メソッドを使用して、リポジトリ内のカスタムコードファイルと共に任意の構成、モデル、またはトークナイザを使用できます。 Hubにアップロードされるすべてのファイルとコードはマルウェアのスキャンが実施されます（詳細は[Hubセキュリティ](https://huggingface.co/docs/hub/security#malware-scanning)ドキュメンテーションを参照してください）、しかし、依然として悪意のあるコードを実行しないために、モデルコードと作者を確認する必要があります。
`trust_remote_code=True` を設定してカスタムコードを持つモデルを使用できます：


```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

コミットハッシュを「revision」として渡すことも強く推奨されています。これにより、モデルの作者がコードを悪意のある新しい行で更新しなかったことを確認できます（モデルの作者を完全に信頼している場合を除きます）。


```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

モデルリポジトリのコミット履歴をブラウジングする際には、任意のコミットのコミットハッシュを簡単にコピーできるボタンがあります。

## Registering a model with custom code to the auto classes

🤗 Transformersを拡張するライブラリを作成している場合、独自のモデルを含めるために自動クラスを拡張したい場合があります。
これはコードをHubにプッシュすることとは異なり、ユーザーはカスタムモデルを取得するためにあなたのライブラリをインポートする必要があります
（Hubからモデルコードを自動的にダウンロードするのとは対照的です）。

構成に既存のモデルタイプと異なる `model_type` 属性がある限り、またあなたのモデルクラスが適切な `config_class` 属性を持っている限り、
次のようにそれらを自動クラスに追加できます：

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

注意: `AutoConfig` にカスタム設定を登録する際の最初の引数は、カスタム設定の `model_type` と一致する必要があります。
また、任意の自動モデルクラスにカスタムモデルを登録する際の最初の引数は、それらのモデルの `config_class` と一致する必要があります。
