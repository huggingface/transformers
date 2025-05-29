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

# Export to TorchScript

<Tip>

これはTorchScriptを使用した実験の最初であり、可変入力サイズのモデルに対するその能力をまだ探求中です。これは私たちの関心の焦点であり、今後のリリースでは、より柔軟な実装や、PythonベースのコードとコンパイルされたTorchScriptを比較するベンチマークを含む、より多くのコード例で詳細な分析を行います。

</Tip>

[TorchScriptのドキュメント](https://pytorch.org/docs/stable/jit.html)によれば：

> TorchScriptは、PyTorchコードから直列化および最適化可能なモデルを作成する方法です。

TorchScriptを使用すると、効率志向のC++プログラムなど、他のプログラムでモデルを再利用できるようになります。PyTorchベースのPythonプログラム以外の環境で🤗 Transformersモデルをエクスポートして使用するためのインターフェースを提供しています。ここでは、TorchScriptを使用してモデルをエクスポートし、使用する方法を説明します。

モデルをエクスポートするには、次の2つの要件があります：

- `torchscript`フラグを使用したモデルのインスタンス化
- ダミーの入力を使用したフォワードパス

これらの必要条件は、以下で詳細に説明されているように、開発者が注意する必要があるいくつかのことを意味します。

## TorchScript flag and tied weights

`torchscript`フラグは、ほとんどの🤗 Transformers言語モデルにおいて、`Embedding`レイヤーと`Decoding`レイヤー間で重みが連結されているため必要です。
TorchScriptでは、重みが連結されているモデルをエクスポートすることはできませんので、事前に重みを切り離して複製する必要があります。

`torchscript`フラグを使用してインスタンス化されたモデルは、`Embedding`レイヤーと`Decoding`レイヤーが分離されており、そのため後でトレーニングしてはいけません。
トレーニングは、これらの2つのレイヤーを非同期にする可能性があり、予期しない結果をもたらす可能性があります。

言語モデルヘッドを持たないモデルには言及しませんが、これらのモデルには連結された重みが存在しないため、`torchscript`フラグなしで安全にエクスポートできます。

## Dummy inputs and standard lengths

ダミー入力はモデルのフォワードパスに使用されます。入力の値はレイヤーを通じて伝播される間、PyTorchは各テンソルに実行された異なる操作を追跡します。これらの記録された操作は、モデルの*トレース*を作成するために使用されます。

トレースは入力の寸法に対して作成されます。そのため、ダミー入力の寸法に制約され、他のシーケンス長やバッチサイズでは動作しません。異なるサイズで試すと、以下のエラーが発生します：

```
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```

お勧めしますのは、モデルの推論中に供給される最大の入力と同じ大きさのダミー入力サイズでモデルをトレースすることです。パディングを使用して不足値を補完することもできます。ただし、モデルがより大きな入力サイズでトレースされるため、行列の寸法も大きくなり、より多くの計算が発生します。

異なるシーケンス長のモデルをエクスポートする際に、各入力に対して実行される演算の総数に注意して、パフォーマンスを密接にフォローすることをお勧めします。

## Using TorchScript in Python

このセクションでは、モデルの保存と読み込み、および推論にトレースを使用する方法を示します。

### Saving a model

TorchScriptで`BertModel`をエクスポートするには、`BertConfig`クラスから`BertModel`をインスタンス化し、それをファイル名`traced_bert.pt`でディスクに保存します：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### Loading a model

以前に保存した `BertModel`、`traced_bert.pt` をディスクから読み込んで、以前に初期化した `dummy_input` で使用できます。

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```


### Using a traced model for inference

トレースモデルを使用して推論を行うには、その `__call__` ダンダーメソッドを使用します。

```python
traced_model(tokens_tensor, segments_tensors)
```


## Deploy Hugging Face TorchScript models to AWS with the Neuron SDK

AWSはクラウドでの低コストで高性能な機械学習推論向けに [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) インスタンスファミリーを導入しました。Inf1インスタンスはAWS Inferentiaチップによって駆動され、ディープラーニング推論ワークロードに特化したカスタムビルドのハードウェアアクセラレータです。[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) はInferentia用のSDKで、トランスフォーマーモデルをトレースして最適化し、Inf1に展開するためのサポートを提供します。

Neuron SDK が提供するもの:

1. クラウドでの推論のためにTorchScriptモデルをトレースして最適化するための、1行のコード変更で使用できる簡単なAPI。
2. [改善されたコストパフォーマンス](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/) のためのボックス外のパフォーマンス最適化。
3. [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) または [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html) で構築されたHugging Faceトランスフォーマーモデルへのサポート。

### Implications

BERT（Bidirectional Encoder Representations from Transformers）アーキテクチャやその変種（[distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) や [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta) など）に基づくトランスフォーマーモデルは、非生成タスク（抽出型質問応答、シーケンス分類、トークン分類など）において、Inf1上で最適に動作します。ただし、テキスト生成タスクも [AWS Neuron MarianMT チュートリアル](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html) に従ってInf1上で実行できます。Inferentiaでボックス外で変換できるモデルに関する詳細情報は、Neuronドキュメンテーションの [Model Architecture Fit](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) セクションにあります。

### Dependencies

モデルをAWS Neuronに変換するには、[Neuron SDK 環境](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide) が必要で、[AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html) に事前に構成されています。

### Converting a model for AWS Neuron

モデルをAWS NEURON用に変換するには、[PythonでTorchScriptを使用する](torchscript#using-torchscript-in-python) と同じコードを使用して `BertModel` をトレースします。Python APIを介してNeuron SDKのコンポーネントにアクセスするために、`torch.neuron` フレームワーク拡張をインポートします。

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

次の行を変更するだけで済みます。

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

これにより、Neuron SDKはモデルをトレースし、Inf1インスタンス向けに最適化します。

AWS Neuron SDKの機能、ツール、サンプルチュートリアル、最新のアップデートについて詳しく知りたい場合は、[AWS NeuronSDK ドキュメンテーション](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html) をご覧ください。



