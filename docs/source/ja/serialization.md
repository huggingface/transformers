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

# Export to ONNX

🤗 Transformersモデルを本番環境に展開する際には、モデルを特殊なランタイムおよびハードウェアで読み込み、実行できるように、モデルをシリアライズされた形式にエクスポートすることが必要であるか、その恩恵を受けることができることがあります。

🤗 Optimumは、Transformersの拡張機能であり、PyTorchまたはTensorFlowからモデルをONNXやTFLiteなどのシリアライズされた形式にエクスポートすることを可能にする「exporters」モジュールを提供しています。また、🤗 Optimumは、最大の効率でターゲットハードウェアでモデルをトレーニングおよび実行するためのパフォーマンス最適化ツールも提供しています。

このガイドでは、🤗 Transformersモデルを🤗 Optimumを使用してONNXにエクスポートする方法を示しており、モデルをTFLiteにエクスポートする方法については[Export to TFLiteページ](tflite)を参照してください。

## Export to ONNX 

[ONNX（Open Neural Network eXchange）](http://onnx.ai)は、PyTorchおよびTensorFlowを含むさまざまなフレームワークで深層学習モデルを表現するための共通の一連の演算子とファイル形式を定義するオープンスタンダードです。モデルがONNX形式にエクスポートされると、これらの演算子はニューラルネットワークを介するデータの流れを表す計算グラフ（一般的には「中間表現」と呼ばれる）を構築するために使用されます。

標準化された演算子とデータ型を備えたグラフを公開することで、ONNXはフレームワーク間の切り替えを容易にします。たとえば、PyTorchでトレーニングされたモデルはONNX形式にエクスポートし、それをTensorFlowでインポートすることができます（逆も同様です）。

ONNX形式にエクスポートされたモデルは、以下のように使用できます：
- [グラフ最適化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization)や[量子化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)などのテクニックを使用して推論のために最適化。
- [`ORTModelForXXX`クラス](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort)を介してONNX Runtimeで実行し、🤗 Transformersでおなじみの`AutoModel` APIに従います。
- [最適化された推論パイプライン](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines)を介して実行し、🤗 Transformersの[`pipeline`]関数と同じAPIを持っています。

🤗 Optimumは、設定オブジェクトを活用してONNXエクスポートをサポートしており、これらの設定オブジェクトは多くのモデルアーキテクチャ用に事前に作成されており、他のアーキテクチャにも簡単に拡張できるように設計されています。

事前に作成された設定のリストについては、[🤗 Optimumドキュメント](https://huggingface.co/docs/optimum/exporters/onnx/overview)を参照してください。

🤗 TransformersモデルをONNXにエクスポートする方法は2つあります。以下では両方の方法を示します：

- export with 🤗 Optimum via CLI.
- export with 🤗 Optimum with `optimum.onnxruntime`.

### Exporting a 🤗 Transformers model to ONNX with CLI

🤗 TransformersモデルをONNXにエクスポートするには、まず追加の依存関係をインストールしてください：

```bash
pip install optimum-onnx
```

すべての利用可能な引数を確認するには、[🤗 Optimumドキュメント](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)を参照してください。または、コマンドラインでヘルプを表示することもできます：


```bash
optimum-cli export onnx --help
```

🤗 Hubからモデルのチェックポイントをエクスポートするには、例えば `distilbert/distilbert-base-uncased-distilled-squad` を使いたい場合、以下のコマンドを実行してください：

```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

進行状況を示し、結果の `model.onnx` が保存される場所を表示するログは、以下のように表示されるはずです：


```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

上記の例は🤗 Hubからのチェックポイントのエクスポートを示しています。ローカルモデルをエクスポートする場合、まずモデルの重みとトークナイザのファイルを同じディレクトリ（`local_path`）に保存してください。CLIを使用する場合、🤗 Hubのチェックポイント名の代わりに`model`引数に`local_path`を渡し、`--task`引数を指定してください。[🤗 Optimumドキュメント](https://huggingface.co/docs/optimum/exporters/task_manager)でサポートされているタスクのリストを確認できます。`task`引数が指定されていない場合、タスク固有のヘッドを持たないモデルアーキテクチャがデフォルトで選択されます。


```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

エクスポートされた `model.onnx` ファイルは、ONNX標準をサポートする[多くのアクセラレータ](https://onnx.ai/supported-tools.html#deployModel)の1つで実行できます。たとえば、[ONNX Runtime](https://onnxruntime.ai/)を使用してモデルを読み込み、実行する方法は以下の通りです：


```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

🤗 HubからTensorFlowのチェックポイントをエクスポートするプロセスは、同様です。例えば、[Keras organization](https://huggingface.co/keras-io)から純粋なTensorFlowのチェックポイントをエクスポートする方法は以下の通りです：


```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### Exporting a 🤗 Transformers model to ONNX with `optimum.onnxruntime`

CLIの代わりに、🤗 TransformersモデルをONNXにプログラム的にエクスポートすることもできます。以下のように行います：

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # Load a model from transformers and export it to ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # Save the onnx model and tokenizer
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### Exporting a model for an unsupported architecture

現在エクスポートできないモデルをサポートするために貢献したい場合、まず[`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview)でサポートされているかどうかを確認し、サポートされていない場合は[🤗 Optimumに貢献](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)してください。
