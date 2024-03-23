<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Processors

Transformers ライブラリでは、プロセッサは 2 つの異なる意味を持ちます。
- [Wav2Vec2](../model_doc/wav2vec2) などのマルチモーダル モデルの入力を前処理するオブジェクト (音声とテキスト)
  または [CLIP](../model_doc/clip) (テキストとビジョン)
- 古いバージョンのライブラリで GLUE または SQUAD のデータを前処理するために使用されていたオブジェクトは非推奨になりました。

## Multi-modal processors

マルチモーダル モデルでは、オブジェクトが複数のモダリティ (テキスト、
視覚と音声）。これは、2 つ以上の処理オブジェクトをグループ化するプロセッサーと呼ばれるオブジェクトによって処理されます。
トークナイザー (テキスト モダリティ用)、画像プロセッサー (視覚用)、特徴抽出器 (オーディオ用) など。

これらのプロセッサは、保存およびロード機能を実装する次の基本クラスを継承します。

[[autodoc]] ProcessorMixin

## Deprecated processors

すべてのプロセッサは、同じアーキテクチャに従っています。
[`~data.processors.utils.DataProcessor`]。プロセッサは次のリストを返します。
[`~data.processors.utils.InputExample`]。これら
[`~data.processors.utils.InputExample`] は次のように変換できます。
[`~data.processors.utils.Input features`] をモデルにフィードします。

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[一般言語理解評価 (GLUE)](https://gluebenchmark.com/) は、
既存の NLU タスクの多様なセットにわたるモデルのパフォーマンス。紙と同時発売された [GLUE: A
自然言語理解のためのマルチタスクベンチマークおよび分析プラットフォーム](https://openreview.net/pdf?id=rJ4km2R5t7)

このライブラリは、MRPC、MNLI、MNLI (不一致)、CoLA、SST2、STSB、
QQP、QNLI、RTE、WNLI。

それらのプロセッサは次のとおりです。

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]


さらに、次のメソッドを使用して、データ ファイルから値をロードし、それらをリストに変換することができます。
[`~data.processors.utils.InputExample`]。

[[autodoc]] data.processors.glue.glue_convert_examples_to_features

## XNLI

[クロスリンガル NLI コーパス (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) は、
言語を超えたテキスト表現の品質。 XNLI は、[*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/) に基づくクラウドソースのデータセットです。テキストのペアには、15 個のテキスト含意アノテーションがラベル付けされています。
さまざまな言語 (英語などの高リソース言語とスワヒリ語などの低リソース言語の両方を含む)。

論文 [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053) と同時にリリースされました。

このライブラリは、XNLI データをロードするプロセッサをホストします。

- [`~data.processors.utils.XnliProcessor`]

テストセットにはゴールドラベルが付いているため、評価はテストセットで行われますのでご了承ください。

これらのプロセッサを使用する例は、[run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) スクリプトに示されています。

## SQuAD

[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) は、次のベンチマークです。
質問応答に関するモデルのパフォーマンスを評価します。 v1.1 と v2.0 の 2 つのバージョンが利用可能です。最初のバージョン
(v1.1) は、論文 [SQuAD: 100,000+ question for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) とともにリリースされました。 2 番目のバージョン (v2.0) は、論文 [Know What You Don't と同時にリリースされました。
知っておくべき: SQuAD の答えられない質問](https://arxiv.org/abs/1806.03822)。

このライブラリは、次の 2 つのバージョンのそれぞれのプロセッサをホストします。

### Processors

それらのプロセッサは次のとおりです。

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

どちらも抽象クラス [`~data.processors.utils.SquadProcessor`] を継承しています。

[[autodoc]] data.processors.squad.SquadProcessor
    - all

さらに、次のメソッドを使用して、SQuAD の例を次の形式に変換できます。
モデルの入力として使用できる [`~data.processors.utils.SquadFeatures`]。

[[autodoc]] data.processors.squad.squad_convert_examples_to_features

これらのプロセッサと前述の方法は、データを含むファイルだけでなく、
*tensorflow_datasets* パッケージ。以下に例を示します。

### Example usage

以下にプロセッサを使用した例と、データ ファイルを使用した変換方法を示します。

```python
# Loading a V2 processor
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# Loading a V1 processor
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

*tensorflow_datasets* の使用は、データ ファイルを使用するのと同じくらい簡単です。

```python
# tensorflow_datasets only handle Squad V1.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

これらのプロセッサを使用する別の例は、[run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) スクリプトに示されています。
