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

# Export to TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide)は、モバイルフォン、組み込みシステム、およびモノのインターネット（IoT）デバイスなど、リソースに制約のあるデバイスに機械学習モデルを展開するための軽量なフレームワークです。TFLiteは、計算能力、メモリ、および電力消費が限られているこれらのデバイス上でモデルを効率的に最適化して実行するために設計されています。
TensorFlow Liteモデルは、`.tflite`ファイル拡張子で識別される特別な効率的なポータブル形式で表されます。

🤗 Optimumは、🤗 TransformersモデルをTFLiteにエクスポートするための機能を`exporters.tflite`モジュールを介して提供しています。サポートされているモデルアーキテクチャのリストについては、[🤗 Optimumのドキュメント](https://huggingface.co/docs/optimum/exporters/tflite/overview)をご参照ください。

モデルをTFLiteにエクスポートするには、必要な依存関係をインストールしてください：


```bash
pip install optimum[exporters-tf]
```

すべての利用可能な引数を確認するには、[🤗 Optimumドキュメント](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model)を参照するか、コマンドラインでヘルプを表示してください：

```bash
optimum-cli export tflite --help
```

🤗 Hubからモデルのチェックポイントをエクスポートするには、例えば `bert-base-uncased` を使用する場合、次のコマンドを実行します：

```bash
optimum-cli export tflite --model bert-base-uncased --sequence_length 128 bert_tflite/
```

進行状況を示すログが表示され、生成された `model.tflite` が保存された場所も表示されるはずです：

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
 ```

上記の例は🤗 Hubからチェックポイントをエクスポートする方法を示しています。ローカルモデルをエクスポートする場合、まずモデルの重みファイルとトークナイザファイルを同じディレクトリ（`local_path`）に保存したことを確認してください。CLIを使用する場合、🤗 Hubのチェックポイント名の代わりに`model`引数に`local_path`を渡します。


