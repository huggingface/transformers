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

# Philosophy

🤗 Transformersは、次のような目的で構築された意見を持つライブラリです：

- 大規模なTransformersモデルを使用、研究、または拡張したい機械学習研究者および教育者。
- これらのモデルを微調整したり、本番環境で提供したり、またはその両方を行いたい実務家。
- 与えられた機械学習タスクを解決するために、事前トレーニングされたモデルをダウンロードして使用したいエンジニア。

このライブラリは、2つの強力な目標を持って設計されました：

1. できるだけ簡単かつ高速に使用できるようにすること：

   - ユーザー向けの抽象化を限りなく少なくし、実際、ほとんどの場合、抽象化はありません。
     各モデルを使用するために必要な3つの標準クラスだけが存在します：[構成](main_classes/configuration)、
     [モデル](main_classes/model)、および前処理クラス（NLP用の[トークナイザ](main_classes/tokenizer)、ビジョン用の[イメージプロセッサ](main_classes/image_processor)、
     オーディオ用の[特徴抽出器](main_classes/feature_extractor)、およびマルチモーダル入力用の[プロセッサ](main_classes/processors)）。
   - これらのクラスは、共通の`from_pretrained()`メソッドを使用して、事前トレーニング済みのインスタンスから簡単かつ統一された方法で初期化できます。このメソッドは、事前トレーニング済みのチェックポイントから関連するクラスのインスタンスと関連データ（構成のハイパーパラメータ、トークナイザの語彙、モデルの重み）をダウンロード（必要な場合はキャッシュ）して読み込みます。これらの基本クラスの上に、ライブラリは2つのAPIを提供しています：[パイプライン]は、特定のタスクでモデルをすばやく推論に使用するためのものであり、[`Trainer`]はPyTorchモデルを迅速にトレーニングまたは微調整するためのものです（すべてのTensorFlowモデルは`Keras.fit`と互換性があります）。
   - その結果、このライブラリはニューラルネットワークのモジュラーツールボックスではありません。ライブラリを拡張または構築したい場合は、通常のPython、PyTorch、TensorFlow、Kerasモジュールを使用し、ライブラリの基本クラスから継承してモデルの読み込みと保存などの機能を再利用するだけです。モデルのコーディング哲学について詳しく知りたい場合は、[Repeat Yourself](https://huggingface.co/blog/transformers-design-philosophy)ブログ投稿をチェックしてみてください。

2. オリジナルのモデルにできるだけ近い性能を持つ最新のモデルを提供すること：

   - 各アーキテクチャに対して、公式な著者から提供された結果を再現する少なくとも1つの例を提供します。
   - コードは通常、可能な限り元のコードベースに近いものであり、これはPyTorchコードがTensorFlowコードに変換されることから生じ、逆もまた然りです。

その他のいくつかの目標：

- モデルの内部をできるだけ一貫して公開すること：

   - フルな隠れ状態と注意の重みにアクセスできる単一のAPIを提供します。
   - 前処理クラスと基本モデルのAPIは標準化され、簡単にモデル間を切り替えることができます。

- これらのモデルの微調整と調査のための有望なツールを主観的に選定すること：

   - 語彙と埋め込みに新しいトークンを追加するための簡単で一貫した方法。
   - Transformerヘッドをマスクおよびプルーンするための簡単な方法。

- PyTorch、TensorFlow 2.0、およびFlaxの間を簡単に切り替えて、1つのフレームワークでトレーニングし、別のフレームワークで推論を行うことを可能にすること。

## Main concepts

このライブラリは、各モデルについて次の3つのタイプのクラスを中心に構築されています：

- **モデルクラス**は、ライブラリで提供される事前トレーニング済みの重みと互換性のあるPyTorchモデル（[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)）、Kerasモデル（[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)）またはJAX/Flaxモデル（[flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)）を使用できます。
- **構成クラス**は、モデルを構築するために必要なハイパーパラメータを格納します（層の数や隠れ層のサイズなど）。これらを自分でインスタンス化する必要はありません。特に、変更を加えずに事前トレーニング済みモデルを使用している場合、モデルを作成すると自動的に構成がインスタンス化されるようになります（これはモデルの一部です）。
- **前処理クラス**は、生データをモデルが受け入れる形式に変換します。[トークナイザ](main_classes/tokenizer)は各モデルの語彙を保存し、文字列をトークン埋め込みのインデックスのリストにエンコードおよびデコードするためのメソッドを提供します。[イメージプロセッサ](main_classes/image_processor)はビジョン入力を前処理し、[特徴抽出器](main_classes/feature_extractor)はオーディオ入力を前処理し、[プロセッサ](main_classes/processors)はマルチモーダル入力を処理します。

これらのすべてのクラスは、事前トレーニング済みのインスタンスからインスタンス化し、ローカルに保存し、Hubで共有することができる3つのメソッドを使用しています：

- `from_pretrained()`は、ライブラリ自体によって提供される（[モデルハブ](https://huggingface.co/models)でサポートされているモデルがあります）か、ユーザーによってローカルに保存された（またはサーバーに保存された）事前トレーニング済みバージョンからモデル、構成、前処理クラスをインスタンス化するためのメソッドです。
- `save_pretrained()`は、モデル、構成、前処理クラスをローカルに保存し、`from_pretrained()`を使用して再読み込みできるようにします。
- `push_to_hub()`は、モデル、構成、前処理クラスをHubに共有し、誰でも簡単にアクセスできるようにします。
