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

# Testing

🤗 Transformersモデルがどのようにテストされ、新しいテストを書いて既存のテストを改善できるかを見てみましょう。

このリポジトリには2つのテストスイートがあります：

1. `tests` -- 一般的なAPI用のテスト
2. `examples` -- APIの一部ではないさまざまなアプリケーション用のテスト

## How transformers are tested

1. PRが提出されると、9つのCircleCiジョブでテストされます。PRへの新しいコミットごとに再テストされます。これらのジョブは、[この設定ファイル](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml)で定義されており、必要な場合は同じ環境を自分のマシンで再現できます。

   これらのCIジョブは `@slow` テストを実行しません。

2. [GitHub Actions](https://github.com/huggingface/transformers/actions)によって実行される3つのジョブがあります：

   - [torch hub integration](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml): torch hubの統合が動作するかどうかを確認します。

   - [self-hosted (push)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): `main` にコミットが行われた場合に、GPUで高速テストを実行します。このジョブは、`main` でのコミットが以下のフォルダーのコードを更新した場合にのみ実行されます：`src`、`tests`、`.github`（追加されたモデルカード、ノートブックなどの実行を防ぐため）。

   - [self-hosted runner](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): GPUで `tests` と `examples` の通常のテストと遅いテストを実行します。

```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```
    結果は[here](https://github.com/huggingface/transformers/actions)で観察できます。

## Running tests



### Choosing which tests to run

このドキュメントは、テストを実行する方法の多くの詳細について説明しています。すべてを読んだ後でも、さらに詳細が必要な場合は、[こちら](https://docs.pytest.org/en/latest/usage.html)で見つけることができます。

以下は、テストを実行するためのいくつかの最も便利な方法です。

すべて実行します:
```console
pytest
```

または：
```bash
make test
```


後者は次のように定義されることに注意してください。

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

以下は、pytestに渡す設定情報です。

- テストプロセスをCPUコアの数と同じだけ実行するように指示します。ただし、RAMが十分でない場合は注意が必要です。
- 同じファイルからのすべてのテストは、同じテストプロセスで実行されるようにします。
- 出力のキャプチャを行いません。
- 冗長モードで実行します。


### Getting the list of all tests

テストスイートのすべてのテスト：

```bash
pytest --collect-only -q
```

指定されたテスト ファイルのすべてのテスト:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### Run a specific test module

個別のテスト モジュールを実行するには:

```bash
pytest tests/utils/test_logging.py
```

### Run specific tests

ほとんどのテストでunittestが使用されているため、特定のサブテストを実行するには、それらのテストを含むunittestクラスの名前を知っている必要があります。例えば、それは次のようになるかもしれません：


```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

テストの実行方法:

テストファイル: `tests/test_optimization.py`
クラス名: `OptimizationTest`
テスト関数の名前: `test_adam_w`

ファイルに複数のクラスが含まれている場合は、特定のクラスのテストのみを実行することを選択できます。例えば：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

テストクラス内のすべてのテストを実行します。

前述の通り、`OptimizationTest` クラスに含まれるテストを実行するには、次のコマンドを実行できます：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

キーワード式を使用してテストを実行できます。

名前に `adam` が含まれるテストのみを実行するには：

```bash
pytest -k adam tests/test_optimization.py
```

`and`および`or`は、すべてのキーワードが一致するか、いずれかを示すために使用できます。`not`は否定するために使用できます。

`adam`という名前を含むテストを除いてすべてのテストを実行するには：

```bash
pytest -k "not adam" tests/test_optimization.py
```

# Japanese Translation

以下は、提供されたテキストの日本語訳です。

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

たとえば、`test_adafactor`と`test_adam_w`の両方を実行するには、以下のコマンドを使用できます:

```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

注意: ここでは、`or` を使用しています。キーワードのいずれか一つが一致すれば、両方を含めるためです。

両方のパターンを含むテストのみを含めたい場合は、`and` を使用してください。

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### Run `accelerate` tests

時々、モデルに対して `accelerate` テストを実行する必要があります。たとえば、`OPT` 実行に対してこれらのテストを実行したい場合、コマンドに `-m accelerate_tests` を追加するだけで済みます：

```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py 
```

### Run documentation tests 

ドキュメンテーションの例が正しいかどうかをテストするには、`doctests` が合格しているかを確認する必要があります。
例として、[`WhisperModel.forward` のドックストリング](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035)を使用しましょう。


```python 
r"""
Returns:

Example:
    ```python
    >>> import torch
    >>> from transformers import WhisperModel, WhisperFeatureExtractor
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```"""

```

指定したファイル内のすべてのドックストリング例を自動的にテストするために、以下の行を実行してください：

```bash 
pytest --doctest-modules <path_to_file_or_dir>
```

ファイルにマークダウン拡張子がある場合は、`--doctest-glob="*.md"`引数を追加する必要があります。
