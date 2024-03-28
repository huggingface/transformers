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


### Run only modified tests

[pytest-picked](https://github.com/anapaulagomes/pytest-picked)を使用すると、未ステージングのファイルまたは現在のブランチ（Gitに従って）に関連するテストを実行できます。これは、変更内容に関連するテストのみ実行されるため、変更が何も壊れていないことを迅速に確認する素晴らしい方法です。変更されていないファイルに関連するテストは実行されません。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

すべてのテストは、変更されたがまだコミットされていないファイルとフォルダから実行されます。

### Automatically rerun failed tests on source modification

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)は、非常に便利な機能を提供しており、すべての失敗したテストを検出し、ファイルを修正する間にそれらの失敗したテストを連続して再実行することができます。そのため、修正を行った後にpytestを再起動する必要がありません。すべてのテストが合格するまで繰り返され、その後再度フルランが実行されます。


```bash
pip install pytest-xdist
```

モードに入るには： `pytest -f`または`pytest --looponfail`

ファイルの変更は、`looponfailroots`ルートディレクトリとその内容全体（再帰的に）を見て検出されます。この値のデフォルトが機能しない場合、`setup.cfg`で設定オプションを変更してプロジェクト内で変更できます。


```ini
[tool:pytest]
looponfailroots = transformers tests
```

または `pytest.ini`/`tox.ini` ファイル:

```ini
[pytest]
looponfailroots = transformers tests
```

ファイルの変更を探すことは、iniファイルのディレクトリを基準にして指定されたディレクトリ内でのみ行われます。

[pytest-watch](https://github.com/joeyespo/pytest-watch) は、この機能の代替実装です。

### Skip a test module

特定のテストモジュールを除外してすべてのテストモジュールを実行したい場合、実行するテストの明示的なリストを指定することができます。例えば、`test_modeling_*.py` テストを除外してすべてを実行するには次のようにします：

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### Clearing state

CIビルドおよび速度に対する隔離が重要な場合（キャッシュに対して）、キャッシュをクリアする必要があります：

```bash
pytest --cache-clear tests
```

### Running tests in parallel

前述のように、`make test` は `pytest-xdist` プラグインを介してテストを並列実行します（`-n X` 引数、例: `-n 2` で2つの並列ジョブを実行）。

`pytest-xdist` の `--dist=` オプションを使用すると、テストがどのようにグループ化されるかを制御できます。`--dist=loadfile` は同じファイルにあるテストを同じプロセスに配置します。

テストの実行順序が異なり予測不可能であるため、`pytest-xdist` を使用してテストスイートを実行すると失敗が発生する場合（つまり、いくつかの未検出の連動テストがある場合）、[pytest-replay](https://github.com/ESSS/pytest-replay) を使用してテストを同じ順序で再生し、その後、失敗するシーケンスを最小限にするのに役立ちます。

### Test order and repetition

潜在的な相互依存性や状態に関連するバグ（ティアダウン）を検出するために、テストを複数回、連続して、ランダムに、またはセットで繰り返すことは有用です。そして、単純な複数回の繰り返しは、DLのランダム性によって明らかになるいくつかの問題を検出するのに役立ちます。

#### Repeat tests

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

そして、すべてのテストを複数回実行します (デフォルトでは 50 回)。

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

このプラグインは、`pytest-xdist` の `-n` フラグでは動作しません。

</Tip>

<Tip>


別のプラグイン `pytest-repeat` もありますが、これは `unittest` では動作しません。

</Tip>

#### Run tests in a random order

```bash
pip install pytest-random-order
```

重要: `pytest-random-order` が存在すると、テストは自動的にランダム化されます。設定の変更や変更は必要ありません。
コマンドラインオプションは必須です。

前に説明したように、これにより、結合されたテスト (1 つのテストの状態が別のテストの状態に影響を与える) の検出が可能になります。いつ
`pytest-random-order` がインストールされていると、そのセッションに使用されたランダム シードが出力されます。例:


```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

そのため、指定された特定のシーケンスが失敗した場合、その正確なシードを追加することでそれを再現できます。例:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

特定のテストのリストを使用しない場合、またはまったくリストを使用しない場合、同じテストの正確な順序を再現します。テストのリストを手動で絞り込み始めると、シードに依存せず、テストが失敗した正確な順序で手動でリストを指定する必要があります。これには、`--random-order-bucket=none` を使用してランダム化を無効にするようpytestに指示する必要があります。例えば、次のようにします：


```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

すべてのテストのシャッフルを無効にするには:

```bash
pytest --random-order-bucket=none
```

デフォルトでは、`--random-order-bucket=module` が暗黙的に適用され、モジュールレベルでファイルをシャッフルします。また、`class`、`package`、`global`、および`none` レベルでシャッフルすることもできます。詳細については、その[ドキュメンテーション](https://github.com/jbasko/pytest-random-order)を参照してください。

別のランダム化の代替手段は、[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly) です。このモジュールは非常に似た機能/インターフェースを持っていますが、`pytest-random-order` で利用可能なバケットモードを持っていません。インストール後に自動的に有効になるという同じ問題があります。

### Look and feel variations

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) は、外観と操作性を向上させ、プログレスバーを追加し、即座に失敗したテストとアサーションを表示するプラグインです。インストール後に自動的にアクティブ化されます。

```bash
pip install pytest-sugar
```


これを使用せずにテストを実行するには、次を実行します。

```bash
pytest -p no:sugar
```

またはアンインストールします。

#### Report each sub-test name and its progress

`pytest` による単一またはグループのテストの場合 (`pip install pytest-pspec` の後):


```bash
pytest --pspec tests/test_optimization.py
```


#### Instantly shows failed tests

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) では、失敗とエラーが即座に表示されます。
テストセッションが終了するまで待機します。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### To GPU or not to GPU

GPU が有効な設定で、CPU のみモードでテストするには、`CUDA_VISIBLE_DEVICES=""`を追加します。

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```


または、複数の GPU がある場合は、`pytest` でどれを使用するかを指定できます。たとえば、
2 番目の GPU GPU `0` と `1` がある場合は、次を実行できます。

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

これは、異なるGPUで異なるタスクを実行したい場合に便利です。

一部のテストはCPUのみで実行する必要があり、他のテストはCPU、GPU、またはTPUで実行する必要があり、また別のテストは複数のGPUで実行する必要があります。次のスキップデコレーターは、テストのCPU/GPU/TPUに関する要件を設定するために使用されます：

- `require_torch` - このテストはtorchの下でのみ実行されます。
- `require_torch_gpu` - `require_torch` に加えて、少なくとも1つのGPUが必要です。
- `require_torch_multi_gpu` - `require_torch` に加えて、少なくとも2つのGPUが必要です。
- `require_torch_non_multi_gpu` - `require_torch` に加えて、0または1つのGPUが必要です。
- `require_torch_up_to_2_gpus` - `require_torch` に加えて、0、1、または2つのGPUが必要です。
- `require_torch_xla` - `require_torch` に加えて、少なくとも1つのTPUが必要です。

以下の表にGPUの要件を示します：

| n gpus | decorator                      |
|--------+--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


たとえば、使用可能な GPU が 2 つ以上あり、pytorch がインストールされている場合にのみ実行する必要があるテストを次に示します。


```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

テストに `tensorflow` が必要な場合は、`require_tf` デコレータを使用します。例えば：

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

これらのデコレータは積み重ねることができます。たとえば、テストが遅く、pytorch で少なくとも 1 つの GPU が必要な場合は、次のようになります。
設定方法:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

`@parametrized` のような一部のデコレータはテスト名を書き換えるため、`@require_*` スキップ デコレータをリストする必要があります。
最後にそれらが正しく動作するようにします。正しい使用例は次のとおりです

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

この順序の問題は `@pytest.mark.parametrize` には存在しません。最初または最後に配置しても、それでも問題は解決されます。
仕事。ただし、それは非単体テストでのみ機能します。

内部テスト:

- 利用可能な GPU の数:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  # works with torch and tf
```

### Testing with a specific PyTorch backend or device

特定のtorchデバイスでテストスイートを実行するには、`TRANSFORMERS_TEST_DEVICE="$device"` を追加します。ここで `$device` は対象のバックエンドです。例えば、CPUでテストするには以下のようにします：

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

この変数は、`mps`などのカスタムまたはあまり一般的ではない PyTorch バックエンドをテストするのに役立ちます。また、特定の GPU をターゲットにしたり、CPU 専用モードでテストしたりすることで、`CUDA_VISIBLE_DEVICES`と同じ効果を達成するために使用することもできます。

特定のデバイスでは、初めて「torch」をインポートした後、追加のインポートが必要になります。これは、環境変数 `TRANSFORMERS_TEST_BACKEND` を使用して指定できます。

```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```

### Distributed training

`pytest` は直接的に分散トレーニングを処理することはできません。試みると、サブプロセスは正しい処理を行わず、自分自身が `pytest` であると思い込んでテストスイートをループで実行し続けます。ただし、通常のプロセスを生成し、それから複数のワーカーを生成し、IOパイプを管理するプロセスを生成すれば機能します。

これを使用するいくつかのテストがあります：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

実行ポイントにすぐに移動するには、これらのテスト内で `execute_subprocess_async` 呼び出しを検索してください。

これらのテストを実行するには、少なくとも2つのGPUが必要です：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### Output capture

テストの実行中に、`stdout` および `stderr` に送信された出力はキャプチャされます。テストまたはセットアップメソッドが失敗した場合、通常、それに対応するキャプチャされた出力が失敗のトレースバックと共に表示されます。

出力のキャプチャを無効にし、`stdout` と `stderr` を通常通りに取得するには、`-s` または `--capture=no` を使用してください：

これらのテストを実行するには少なくとも2つのGPUが必要です：

```bash
pytest -s tests/utils/test_logging.py
```

テスト結果を JUnit 形式の出力に送信するには:

```bash
py.test tests --junitxml=result.xml
```

### Color control

色を持たないようにする（例：黄色のテキストを白い背景に表示すると読みにくいです）：


```bash
pytest --color=no tests/utils/test_logging.py
```

### Sending test report to online pastebin service

テスト失敗ごとに URL を作成します。


```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

これにより、テスト実行情報がリモートのPasteサービスに送信され、各エラーに対してURLが提供されます。通常通りテストを選択するか、たとえば特定のエラーのみを送信したい場合は `-x` を追加で指定できます。

テストセッション全体のログに対するURLを作成する方法：


```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## Writing tests

🤗 transformersのテストは `unittest` を基にしていますが、 `pytest` で実行されるため、ほとんどの場合、両方のシステムの機能を使用できます。

[こちら](https://docs.pytest.org/en/stable/unittest.html)でサポートされている機能を読むことができますが、重要なことは、ほとんどの `pytest` のフィクスチャが動作しないことです。パラメータ化も同様ですが、似たような方法で動作する `parameterized` モジュールを使用しています。

### Parametrization

同じテストを異なる引数で複数回実行する必要があることがよくあります。これはテスト内部から行うこともできますが、その場合、そのテストを単一の引数セットで実行する方法はありません。


```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

デフォルトでは、このテストは3回実行され、それぞれの実行で `test_floor` の最後の3つの引数がパラメータリストの対応する引数に割り当てられます。

そして、`negative` と `integer` パラメータのセットのみを実行することもできます:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

または、`Negative`のサブテストを除くすべての場合、次のようになります。

```bash
pytest -k "not negative" tests/test_mytest.py
```

`-k` フィルターを使用することに加えて、各サブテストの正確な名前を調べ、その正確な名前を使用して任意のサブテストまたはすべてのサブテストを実行することができます。


```bash
pytest test_this1.py --collect-only -q
```

すると次のものがリストされます:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```


したがって、2 つの特定のサブテストのみを実行できるようになりました。

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

`transformers`の開発者依存関係にすでに含まれているモジュール[parameterized](https://pypi.org/project/parameterized/) は、`unittests` と `pytest` テストの両方で機能します。

ただし、テストが `unittest` でない場合、`pytest.mark.parametrize` を使用することができます（または既存のテストのいくつかで、主に `examples` の下で使用されているのを見ることができます）。

次に、同じ例を示しますが、今度は `pytest` の `parametrize` マーカーを使用しています：


```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

`parameterized` と同様に、`pytest.mark.parametrize` を使用すると、`-k` フィルタが役立たない場合でも、サブテストの実行を細かく制御できます。ただし、このパラメータ化関数はサブテストの名前をわずかに異なるものにします。以下にその例を示します：


```bash
pytest test_this2.py --collect-only -q
```

すると次のものがリストされます:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

これで、特定のテストのみを実行できるようになりました。

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

前の例と同様に。

### Files and directories

テストの中で、現在のテストファイルからの相対位置を知る必要があることがよくあります。しかし、これは簡単なことではありません。なぜなら、テストは複数のディレクトリから呼び出されるか、異なる深さのサブディレクトリに存在することがあるからです。`transformers.test_utils.TestCasePlus` というヘルパークラスは、すべての基本パスを整理し、簡単にアクセスできるようにすることで、この問題を解決します。

- `pathlib` オブジェクト（すべて完全に解決されたもの）：

  - `test_file_path` - 現在のテストファイルのパス、つまり `__file__`
  - `test_file_dir` - 現在のテストファイルを含むディレクトリ
  - `tests_dir` - `tests` テストスイートのディレクトリ
  - `examples_dir` - `examples` テストスイートのディレクトリ
  - `repo_root_dir` - リポジトリのディレクトリ
  - `src_dir` - `transformers` サブディレクトリが存在する場所

- パスの文字列表現――上記と同じですが、これらは `pathlib` オブジェクトではなく文字列としてパスを返します：

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

これらを使用し始めるには、テストが `transformers.test_utils.TestCasePlus` のサブクラスに存在することを確認するだけです。例：

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

もし、`pathlib` を介してパスを操作する必要がない場合、または単に文字列としてパスが必要な場合は、`pathlib` オブジェクトに `str()` を呼び出すか、`_str` で終わるアクセサを使用できます。例：

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### Temporary files and directories

一意の一時ファイルとディレクトリの使用は、並列テストの実行には欠かせません。これにより、テストがお互いのデータを上書きしないようにします。また、これらを作成した各テストの終了時に一時ファイルとディレクトリが削除されることを望みます。そのため、これらのニーズを満たすパッケージである `tempfile` のようなパッケージの使用は重要です。

しかし、テストのデバッグ時には、一時ファイルやディレクトリに何が格納されているかを確認できる必要があり、テストを再実行するたびにランダムに変更されないその正確なパスを知りたいと思います。

`transformers.test_utils.TestCasePlus` というヘルパークラスは、このような目的に最適です。これは `unittest.TestCase` のサブクラスであるため、テストモジュールで簡単に継承することができます。

以下はその使用例です：


```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

このコードはユニークな一時ディレクトリを作成し、`tmp_dir` をその場所に設定します。

- ユニークな一時ディレクトリを作成します：

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` には、作成された一時ディレクトリへのパスが含まれます。期間終了後は自動的に削除されます
テスト。

- 任意の一時ディレクトリを作成し、テストの開始前にそれが空であることを確認し、テスト後には空にしないでください。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

これは、特定のディレクトリを監視し、前のテストがそこにデータを残さないことを確認したい場合に、デバッグに役立ちます。

- `before` と `after` 引数を直接オーバーライドすることで、デフォルトの動作をオーバーライドできます。以下のいずれかの動作に導きます：

  - `before=True`：テストの開始時に常に一時ディレクトリがクリアされます。
  - `before=False`：一時ディレクトリが既に存在する場合、既存のファイルはそのままになります。
  - `after=True`：テストの終了時に常に一時ディレクトリが削除されます。
  - `after=False`：テストの終了時に常に一時ディレクトリはそのままになります。

<Tip>

`rm -r`の相当を安全に実行するために、明示的な `tmp_dir` が使用される場合、プロジェクトリポジトリのチェックアウトのサブディレクトリのみが許可されます。誤って `/tmp` などのファイルシステムの重要な部分が削除されないように、常に `./` から始まるパスを渡してください。

</Tip>

<Tip>

各テストは複数の一時ディレクトリを登録でき、要求がない限りすべて自動で削除されます。

</Tip>

### Temporary sys.path override

別のテストからインポートするために一時的に `sys.path` をオーバーライドする必要がある場合、`ExtendSysPath` コンテキストマネージャを使用できます。例：


```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### Skipping tests

これは、バグが見つかり、新しいテストが作成された場合であっても、バグがまだ修正されていない場合に役立ちます。メインリポジトリにコミットできるようにするには、`make test` の実行中にそれをスキップする必要があります。

メソッド：

- **skip** は、テストが特定の条件が満たされた場合にのみパスすることを期待しており、それ以外の場合は pytest がテストの実行をスキップします。一般的な例は、Windows専用のテストを非Windowsプラットフォームでスキップする場合、または現在利用できない外部リソースに依存するテストをスキップする場合です（例: データベースが利用できない場合）。

- **xfail** は、何らかの理由でテストが失敗することを期待しています。一般的な例は、まだ実装されていない機能のテストや、まだ修正されていないバグのテストです。テストが予想される失敗にもかかわらずパスした場合（pytest.mark.xfailでマークされたテスト）、それはxpassとしてテストサマリーに報告されます。

これらの2つの間の重要な違いの1つは、`skip` はテストを実行しない点であり、`xfail` は実行します。したがって、バグのあるコードが他のテストに影響を与える場合は、`xfail` を使用しないでください。

#### Implementation

- テスト全体を無条件にスキップする方法は次のとおりです：


```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

または pytest 経由:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

または `xfail` の方法:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```


- テスト内の内部チェックに基づいてテストをスキップする方法は次のとおりです。

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

またはモジュール全体:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

または `xfail` の方法:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 一部のインポートが欠落している場合にモジュール内のすべてのテストをスキップする方法は次のとおりです。

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 条件に基づいてテストをスキップします。

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

または：

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```


またはモジュール全体をスキップします。

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

詳細、例、および方法についての詳細は[こちら](https://docs.pytest.org/en/latest/skipping.html)を参照してください。

### Slow tests

テストライブラリは着実に成長しており、テストの一部は数分かかります。そのため、CIでテストスイートの完了を待つのは1時間待つ余裕がないことがあります。したがって、いくつかの例外を除いて、遅いテストは以下の例のようにマークすべきです：


```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```


テストが`@slow`としてマークされたら、そのようなテストを実行するには、環境変数 `RUN_SLOW=1`を設定します。例:

```bash
RUN_SLOW=1 pytest tests
```

`@parameterized` のようなデコレータはテスト名を書き換えるため、`@slow` および他のスキップデコレータ `@require_*` は正しく動作するためには、最後にリストアップする必要があります。以下は正しい使用例の一例です：


```python no-style
@parameterized.expand(...)
@slow
def test_integration_foo():
```

このドキュメントの冒頭で説明したように、遅いテストは定期的なスケジュールに従って実行され、PRのCIチェックでは実行されません。そのため、一部の問題がPRの提出時に見落とされ、マージされる可能性があります。そのような問題は次回のスケジュールされたCIジョブで検出されます。しかし、それはまた、PRを提出する前に自分のマシンで遅いテストを実行する重要性を意味しています。

どのテストを遅いテストとしてマークすべきかを選択するための、おおまかな意思決定メカニズムが次に示されています：

- テストがライブラリの内部コンポーネントの1つに焦点を当てている場合（例: モデリングファイル、トークン化ファイル、パイプライン）、そのテストは遅いテストスイートで実行する必要があります。それがライブラリの他の側面、たとえばドキュメンテーションや例に焦点を当てている場合、それらのテストは遅いテストスイートで実行する必要があります。そして、このアプローチを洗練させるために例外を設ける必要があります。

- 重いウェイトセットや約50MB以上のデータセットをダウンロードする必要があるすべてのテスト（例: モデル統合テスト、トークナイザ統合テスト、パイプライン統合テスト）は遅いテストとして設定する必要があります。新しいモデルを追加する場合、統合テスト用にランダムなウェイトを持つ小さなバージョンを作成し、ハブにアップロードする必要があります。これについては以下の段落で詳しく説明します。

- 特に高速化されていないトレーニングを行う必要があるすべてのテストは遅いテストとして設定する必要があります。

- 一部の「遅い」であるべきでないテストが非常に遅い場合、およびそれらを `@slow` として設定する必要がある場合には例外を導入できます。大容量のファイルをディスクに保存および読み込みする自動モデリングテストは、`@slow` としてマークされたテストの良い例です。

- CIで1秒未満でテストが完了する場合（ダウンロードを含む）、それは通常のテストであるべきです。

すべての非遅いテストは、さまざまな内部要素を完全にカバーする必要がありますが、高速である必要があります。たとえば、特別に作成された小さなモデル（レイヤー数が最小限で、語彙サイズが小さいなど）を使用して、かなりのカバレッジを実現できます。その後、`@slow` テストでは大規模な遅いモデルを使用して質的なテストを実行できます。これらを使用するには、以下のように *tiny* モデルを探してください：


```bash
grep tiny tests examples
```

[スクリプトの例](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py)があり、これにより tiny-wmt19-en-de のような小さなモデルが作成されます。特定のモデルのアーキテクチャに簡単に調整できます。

実行時間を誤って測定することが簡単です。たとえば、巨大なモデルのダウンロードに関するオーバーヘッドがある場合、ローカルでテストするとダウンロードされたファイルがキャッシュされ、ダウンロード時間が計測されなくなります。したがって、CIログの実行速度レポート（`pytest --durations=0 tests` の出力）を確認してください。

このレポートは、遅いテストとしてマークされていない遅い外れ値や、高速に書き直す必要があるテストを見つけるのにも役立ちます。テストスイートがCIで遅くなり始めた場合、このレポートのトップリストには最も遅いテストが表示されます。

### Testing the stdout/stderr output

`stdout` および/または `stderr` に書き込む関数をテストするために、テストは `pytest` の [capsys システム](https://docs.pytest.org/en/latest/capture.html) を使用してこれらのストリームにアクセスできます。以下はその方法です：


```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```


そしてもちろん、ほとんどの場合、`stderr`は例外の一部として提供されるため、そのような場合には try/excel を使用する必要があります。
ケース：

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```


stdout をキャプチャするもう 1 つのアプローチは、`contextlib.redirect_stdout`を使用することです。

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

stdout をキャプチャする際の重要な潜在的な問題は、通常の `print` でこれまでに出力された内容をリセットする可能性がある `\r` 文字が含まれている可能性があることです。`pytest` 自体には問題はありませんが、`pytest -s` ではこれらの文字がバッファに含まれるため、`-s` ありとなしでテストを実行できるようにするには、`re.sub(r'~.*\r', '', buf, 0, re.M)` を使用してキャプチャされた出力に対して追加のクリーンアップを行う必要があります。

しかし、その後、`\r` が含まれているかどうかにかかわらず、すべての操作を自動的に処理するヘルパーコンテキストマネージャラッパーがあります。したがって、次のように簡単に行えます：


```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

完全なテスト例は次のとおりです。

```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

`stderr` をキャプチャしたい場合は、代わりに `CaptureStderr` クラスを使用してください。

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

両方のストリームを一度にキャプチャする必要がある場合は、親の `CaptureStd` クラスを使用します。

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```


また、テストの問題のデバッグを支援するために、デフォルトで、これらのコンテキスト マネージャーは終了時にキャプチャされたストリームを自動的に再生します。
文脈から。

### Capturing logger stream

ロガーの出力を検証する必要がある場合は、`CaptureLogger`を使用できます。

```python
from transformers import logging
from transformers.testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### Testing with environment variables

特定のテストで環境変数の影響をテストしたい場合は、ヘルパー デコレータを使用できます。
`transformers.testing_utils.mockenv`

```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

場合によっては、外部プログラムを呼び出す必要があるため、`os.environ` に`PYTHONPATH`を設定してインクルードする必要があります。
複数のローカル パス。ヘルパー クラス `transformers.test_utils.TestCasePlus` が役に立ちます。

```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

テストファイルが `tests` テストスイートまたは `examples` のどちらにあるかに応じて
`env[PYTHONPATH]` を使用して、これら 2 つのディレクトリのいずれかを含めます。また、テストが確実に行われるようにするための `src` ディレクトリも含めます。
現在のリポジトリに対して実行され、最後に、テストが実行される前にすでに設定されていた `env[PYTHONPATH]` を使用して実行されます。
何かあれば呼ばれます。

このヘルパー メソッドは `os.environ` オブジェクトのコピーを作成するため、元のオブジェクトはそのまま残ります。


### Getting reproducible results

状況によっては、テストのランダム性を削除したい場合があります。同一の再現可能な結果セットを取得するには、
シードを修正する必要があります:

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
tf.random.set_seed(seed)
```


### Debugging tests

警告が発生した時点でデバッガーを開始するには、次の手順を実行します。

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## Working with github actions workflows

セルフプッシュのワークフローCIジョブをトリガーするには、以下の手順を実行する必要があります：

1. `transformers` のリモートリポジトリで新しいブランチを作成します（フォークではなく、元のリポジトリで行います）。
2. ブランチの名前は `ci_` または `ci-` で始まる必要があります（`main` もトリガーしますが、`main` ではPRを作成できません）。また、特定のパスでのみトリガーされます - このドキュメントが書かれた後に変更された場合に備えて、最新の定義は[こちら](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml)の *push:* にあります。
3. このブランチからPRを作成します。
4. その後、このジョブが[ここ](https://github.com/huggingface/transformers/actions/workflows/self-push.yml)に表示されます。ジョブはバックログがある場合、すぐに実行されないことがあります。

## Testing Experimental CI Features

CI機能のテストは通常のCIの正常な動作に干渉する可能性があるため、新しいCI機能を追加する場合、以下の手順に従う必要があります。

1. テストが必要なものをテストするための新しい専用のジョブを作成します。
2. 新しいジョブは常に成功する必要があるため、常にグリーン ✓（詳細は以下参照）を表示する必要があります。
3. さまざまな種類のPR（ユーザーフォークブランチ、非フォークブランチ、github.com UIから直接ファイルを編集するブランチ、さまざまな強制プッシュなど）が実行されるまでいくつかの日間実行し、実験的なジョブのログを監視します（意図的に常にグリーンになるようになっている全体のジョブの緑ではなく）。
4. すべてが安定していることが明確になったら、新しい変更を既存のジョブに統合します。

このように、CI機能自体の実験が通常のワークフローに干渉しないようにできます。

では、新しいCI機能が開発中である間、ジョブを常に成功させるにはどうすればいいでしょうか？

TravisCIのような一部のCIは `ignore-step-failure` をサポートし、全体のジョブを成功として報告しますが、この文書が作成された時点ではCircleCIとGithub Actionsはそれをサポートしていません。

したがって、以下のワークアラウンドを使用できます：

1. bashスクリプト内で潜在的な失敗を抑制するために実行コマンドの冒頭に `set +euo pipefail` を記述します。
2. 最後のコマンドは成功する必要があります。たとえば `echo "done"` または単に `true` を使用できます。

以下は例です：



```yaml
- run:
    name: run CI experiment
    command: |
        set +euo pipefail
        echo "setting run-all-despite-any-errors-mode"
        this_command_will_fail
        echo "but bash continues to run"
        # emulate another failure
        false
        # but the last command must be a success
        echo "during experiment do not remove: reporting success to CI, even if there were failures"
```


単純なコマンドの場合は、次のようにすることもできます。

```bash
cmd_that_may_fail || true
```

もちろん、結果に満足したら、実験的なステップやジョブを通常のジョブと統合し、`set +euo pipefail` などの追加した要素を削除して、実験的なジョブが通常のCIの動作に干渉しないようにします。

このプロセス全体は、実験的なステップに対して `allow-failure` のようなものを設定し、PRの全体のステータスに影響を与えずに失敗させることができれば、はるかに簡単になったでしょう。しかし、前述の通り、現在はCircleCIとGithub Actionsはこの機能をサポートしていません。

この機能に関しての投票や、CIに特有のスレッドでその進捗状況を確認できます：

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)

