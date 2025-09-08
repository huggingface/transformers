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

# Models

ベースクラスである [`PreTrainedModel`]、[`TFPreTrainedModel`]、[`FlaxPreTrainedModel`] は、モデルの読み込みと保存に関する共通のメソッドを実装しており、これはローカルのファイルやディレクトリから、またはライブラリが提供する事前学習モデル構成（HuggingFaceのAWS S3リポジトリからダウンロード）からモデルを読み込むために使用できます。

[`PreTrainedModel`] と [`TFPreTrainedModel`] は、次の共通のメソッドも実装しています：

- 語彙に新しいトークンが追加された場合に、入力トークン埋め込みのリサイズを行う
- モデルのアテンションヘッドを刈り込む

各モデルに共通するその他のメソッドは、[`~modeling_utils.ModuleUtilsMixin`]（PyTorchモデル用）および[`~modeling_tf_utils.TFModuleUtilsMixin`]（TensorFlowモデル用）で定義されており、テキスト生成の場合、[`~generation.GenerationMixin`]（PyTorchモデル用）、[`~generation.TFGenerationMixin`]（TensorFlowモデル用）、および[`~generation.FlaxGenerationMixin`]（Flax/JAXモデル用）もあります。


## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'></a>


### 大規模モデルの読み込み

Transformers 4.20.0では、[`~PreTrainedModel.from_pretrained`] メソッドが再設計され、[Accelerate](https://huggingface.co/docs/accelerate/big_modeling) を使用して大規模モデルを扱うことが可能になりました。これには Accelerate >= 0.9.0 と PyTorch >= 1.9.0 が必要です。以前の方法でフルモデルを作成し、その後事前学習の重みを読み込む代わりに（これにはメモリ内のモデルサイズが2倍必要で、ランダムに初期化されたモデル用と重み用の2つが必要でした）、モデルを空の外殻として作成し、事前学習の重みが読み込まれるときにパラメーターを実体化するオプションが追加されました。

さらに、モデルが完全にRAMに収まらない場合（現時点では推論のみ有効）、異なるデバイスにモデルを直接配置できます。`device_map="auto"` を使用すると、Accelerateは各レイヤーをどのデバイスに配置するかを決定し、最速のデバイス（GPU）を最大限に活用し、残りの部分をCPU、あるいはGPU RAMが不足している場合はハードドライブにオフロードします。モデルが複数のデバイスに分割されていても、通常どおり実行されます。


```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

モデルがデバイス間でどのように分割されたかは、その `hf_device_map` 属性を見ることで確認できます:

```py
t0pp.hf_device_map
```

```python out
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

同じフォーマットに従って、独自のデバイスマップを作成することもできます（レイヤー名からデバイスへの辞書です）。モデルのすべてのパラメータを指定されたデバイスにマップする必要がありますが、1つのレイヤーが完全に同じデバイスにある場合、そのレイヤーのサブモジュールのすべてがどこに行くかの詳細を示す必要はありません。例えば、次のデバイスマップはT0ppに適しています（GPUメモリがある場合）:

```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

モデルのメモリへの影響を最小限に抑えるもう 1 つの方法は、低精度の dtype (`torch.float16` など) でモデルをインスタンス化するか、以下で説明する直接量子化手法を使用することです。

### Model Instantiation dtype

Pytorch では、モデルは通常 `torch.float32` 形式でインスタンス化されます。これは、しようとすると問題になる可能性があります
重みが fp16 にあるモデルをロードすると、2 倍のメモリが必要になるためです。この制限を克服するには、次のことができます。
`dtype` 引数を使用して、目的の `dtype` を明示的に渡します。

```python
model = T5ForConditionalGeneration.from_pretrained("t5", dtype=torch.float16)
```
または、モデルを常に最適なメモリ パターンでロードしたい場合は、特別な値 `"auto"` を使用できます。
そして、`dtype` はモデルの重みから自動的に導出されます。

```python
model = T5ForConditionalGeneration.from_pretrained("t5", dtype="auto")
```

スクラッチからインスタンス化されたモデルには、どの `dtype` を使用するかを指示することもできます。

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

Pytorch の設計により、この機能は浮動小数点 dtype でのみ使用できます。

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
