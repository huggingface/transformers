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


# Checks on a Pull Request

🤗 Transformersリポジトリでプルリクエストを開くと、追加しているパッチが既存のものを壊していないことを確認するために、かなりの数のチェックが実行されます。これらのチェックには、次の4つのタイプがあります：
- 通常のテスト
- ドキュメンテーションのビルド
- コードとドキュメンテーションのスタイル
- リポジトリ全体の一貫性

このドキュメントでは、これらのさまざまなチェックとその背後にある理由、そしてそれらのいずれかがあなたのプルリクエストで失敗した場合のローカルでのデバッグ方法について説明します。

なお、理想的には、開発者用のインストールが必要です：


```bash
pip install transformers[dev]
```

または編集可能なインストールの場合：


```bash
pip install -e .[dev]
```

トランスフォーマーズのリポジトリ内で作業しています。トランスフォーマーズのオプションの依存関係の数が増えたため、すべてを取得できない可能性があります。開発用インストールが失敗した場合、作業しているディープラーニングフレームワーク（PyTorch、TensorFlow、および/またはFlax）をインストールし、次の手順を実行してください。


```bash
pip install transformers[quality]
```

または編集可能なインストールの場合：

```bash
pip install -e .[quality]
```

## Tests

`ci/circleci: run_tests_` で始まるすべてのジョブは、Transformersのテストスイートの一部を実行します。これらのジョブは、特定の環境でライブラリの一部に焦点を当てて実行されます。たとえば、`ci/circleci: run_tests_pipelines_tf` は、TensorFlowのみがインストールされた環境でパイプラインのテストを実行します。

テストスイートの一部のみが実行されるように注意してください。テストスイートは、変更前と変更後のPRのライブラリの違いを決定し、その違いに影響を受けるテストを選択するためのユーティリティが実行されます。このユーティリティは、ローカルで以下のコマンドを実行して実行できます：


```bash
python utils/tests_fetcher.py
```

1. リポジトリのルートからスクリプトを実行します。これは次のステップを実行します：

   1. 差分内の各ファイルをチェックし、変更がコード内にあるか、コメントやドキュメンテーション文字列のみにあるかを確認します。実際のコード変更があるファイルのみを保持します。

   2. 内部のマップを構築します。このマップは、ライブラリのソースコードの各ファイルが再帰的に影響を与えるすべてのファイルを提供します。モジュールAがモジュールBに影響を与えるとは、モジュールBがモジュールAをインポートする場合を指します。再帰的な影響を得るには、モジュールAからモジュールBへのモジュールのチェーンが必要で、各モジュールは前のモジュールをインポートする必要があります。

   3. このマップをステップ1で収集したファイルに適用します。これにより、PRに影響を受けるモデルファイルのリストが得られます。

   4. これらのファイルをそれに対応するテストファイルにマップし、実行するテストのリストを取得します。

2. スクリプトをローカルで実行する場合、ステップ1、3、および4の結果が表示され、実行するテストがわかります。スクリプトはまた、`test_list.txt` という名前のファイルを作成します。このファイルには実行するテストのリストが含まれており、次のコマンドでローカルで実行できます：

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

## Documentation build

`build_pr_documentation` ジョブは、ドキュメンテーションのビルドを行い、あなたのPRがマージされた後にすべてが正常に表示されることを確認します。ボットがプレビューのドキュメンテーションへのリンクをPRに追加します。PRに対する変更は、プレビューに自動的に反映されます。ドキュメンテーションのビルドに失敗した場合、失敗したジョブの隣にある「詳細」をクリックして、何が問題になっているかを確認できます。多くの場合、問題は`toctree`内のファイルが不足しているなど、単純なものです。

ドキュメンテーションをローカルでビルドまたはプレビューしたい場合は、[docsフォルダ内の`README.md`](https://github.com/huggingface/transformers/tree/main/docs)をご覧ください。

## Code and documentation style

すべてのソースファイル、例、テストには、`black`と`ruff`を使用してコードのフォーマットが適用されます。また、ドックストリングと`rst`ファイルのフォーマット、Transformersの`__init__.py`ファイルで実行される遅延インポートの順序についてもカスタムツールが存在します（`utils/style_doc.py`と`utils/custom_init_isort.py`）。これらすべては、以下を実行することで起動できます。


```bash
make style
```

CIは、`ci/circleci: check_code_quality` チェック内でこれらのチェックが適用されていることを確認します。また、`ruff` を実行し、未定義の変数や使用されていない変数がある場合にエラーを報告します。このチェックをローカルで実行するには、以下のコマンドを使用してください。


```bash
make quality
```

時間がかかることがあります。したがって、現在のブランチで変更したファイルのみで同じことを実行するには、次のコマンドを実行します。


```bash
make fixup
```

この最後のコマンドは、リポジトリの整合性のためのすべての追加のチェックも実行します。それらを詳しく見てみましょう。

## Repository consistency

これには、あなたのPRがリポジトリを適切な状態に保ったままであることを確認するためのすべてのテストが含まれており、ci/`circleci: check_repository_consistency` チェックによって実行されます。ローカルでこのチェックを実行するには、以下を実行します。

```bash
make repo-consistency
```

これを確認します：

- `utils/check_repo.py` によって実行される、init に追加されたすべてのオブジェクトが文書化されています。
- `utils/check_inits.py` によって実行される、すべての `__init__.py` ファイルがその2つのセクションで同じ内容を持っています。
- `utils/check_copies.py` によって実行される、他のモジュールからのコピーとして識別されたすべてのコードが元のコードと一致しています。
- `utils/check_config_docstrings.py` によって実行される、すべての設定クラスには少なくとも1つの有効なチェックポイントがドキュメント文字列に記載されています。
- `utils/check_config_attributes.py` によって実行される、すべての設定クラスには、対応するモデリングファイルで使用されている属性のみが含まれています。
- `utils/check_copies.py` によって実行される、README とドキュメントのインデックスの翻訳が、メインのREADME と同じモデルリストを持っています。
- `utils/check_table.py` によって実行される、ドキュメンテーションの自動生成テーブルが最新であることを確認します。
- `utils/check_dummies.py` によって実行される、すべてのオブジェクトが利用可能であり、オプションの依存関係がすべてインストールされていなくても問題ありません。

このチェックが失敗する場合、最初の2つの項目は手動で修正する必要があり、最後の4つはコマンドを実行して自動的に修正できます。


```bash
make fix-copies
```

追加のチェックポイントは、新しいモデルを追加するPull Request（PR）に関連しています。主に次の点を確認します：

- すべての追加されたモデルは、Auto-mapping（`utils/check_repo.py`で実行）に含まれています。
<!-- TODO Sylvain、共通のテストが実装されていることを確認するチェックを追加してください。-->
- すべてのモデルが適切にテストされています（`utils/check_repo.py`で実行）。

<!-- TODO Sylvain、以下を追加してください
- すべてのモデルがメインのREADMEおよびメインのドキュメント内に追加されています。
- 使用されているすべてのチェックポイントが実際にHubに存在しています
-->


### Check copies

Transformersライブラリは、モデルコードに関して非常に意見があるため、各モデルは他のモデルに依存せずに完全に1つのファイルに実装する必要があります。したがって、特定のモデルのコードのコピーが元のコードと一貫しているかどうかを確認する仕組みを追加しました。これにより、バグ修正がある場合、他の影響を受けるモデルをすべて確認し、変更を伝達するかコピーを破棄するかを選択できます。

<Tip>

ファイルが別のファイルの完全なコピーである場合、それを`utils/check_copies.py`の`FULL_COPIES`定数に登録する必要があります。

</Tip>

この仕組みは、`# Copied from xxx`という形式のコメントに依存しています。`xxx`は、コピーされているクラスまたは関数の完全なパスを含む必要があります。例えば、`RobertaSelfOutput`は`BertSelfOutput`クラスの直接のコピーですので、[こちら](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L289)にコメントがあります。


```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
```

注意点として、これをクラス全体に適用する代わりに、コピー元の関連メソッドに適用できます。たとえば、[こちら](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L598)では、`RobertaPreTrainedModel._init_weights` が `BertPreTrainedModel` からコピーされており、以下のコメントがあります：


```py
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
```

注：矢印の周りにはスペースが含まれていてはいけません（もちろん、そのスペースが置換パターンの一部である場合を除きます）。

カンマで区切られた複数のパターンを追加できます。例えば、ここでは `CamemberForMaskedLM` は `RobertaForMaskedLM` の直接のコピーで、2つの置換があります： `Roberta` から `Camembert` へ、そして `ROBERTA` から `CAMEMBERT` へと置換されます。[こちら](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/camembert/modeling_camembert.py#L929)で、この作業はコメント付きで行われています。


```py
# Copied from transformers.models.roberta.modeling_roberta.RobertaForMaskedLM with Roberta->Camembert, ROBERTA->CAMEMBERT
```


もし順序が重要な場合（以前の置換と競合する可能性があるため）、置換は左から右に実行されます。

<Tip>

もし置換がフォーマットを変更する場合（たとえば、短い名前を非常に長い名前に置き換える場合など）、自動フォーマッタを適用した後にコピーが確認されます。

</Tip>

パターンが同じ置換の異なるケース（大文字と小文字のバリアントがある）の場合、オプションとして `all-casing` を追加するだけの別の方法もあります。[こちら](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/mobilebert/modeling_mobilebert.py#L1237)は、`MobileBertForSequenceClassification` 内の例で、コメントがついています。


```py
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

この場合、コードは「BertForSequenceClassification」からコピーされ、次のように置換されます：
- `Bert` を `MobileBert` に置き換える（例：`init`で `MobileBertModel` を使用する場合）
- `bert` を `mobilebert` に置き換える（例：`self.mobilebert` を定義する場合）
- `BERT` を `MOBILEBERT` に置き換える（定数 `MOBILEBERT_INPUTS_DOCSTRING` 内で）

