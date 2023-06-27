<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置
获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0 除非适用法律要求或书面同意，否则根据许可证分发的软件是根据“按原样”基础分发的，不附带任何形式的明示或暗示保证。请参阅许可证以获取
特定语言下的权限和限制。请注意，此文件是 Markdown 格式的，但包含了特定的语法，用于我们的 doc-builder（类似于 MDX），这可能在您的 Markdown 查看器中无法正确呈现。-->

# 测试

让我们来看看🤗 Transformers 模型是如何进行测试的，以及您如何编写新的测试并改进现有的测试。
存储库中有两个测试套件：

1. `tests` -- 用于通用 API 的测试 2. `examples` -- 主要用于各种不属于 API 的应用程序的测试


## Transformers 如何进行测试

1. 提交 PR 后，它会通过 9 个 CircleCi 作业进行测试。对 PR 进行的每次新提交都会重新测试。这些作业   在此 [配置文件](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml) 中定义，因此如果需要，您可以在您的机器上重现相同的   环境。

   这些 CI 作业不会运行 `@slow` 测试。
2. 由 [github actions](https://github.com/huggingface/transformers/actions) 运行了 3 个作业：
   - [torch hub integration](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml)：检查 torch hub     集成是否正常工作。
   - [self-hosted (push)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml)：仅在提交到     `main` 的情况下，仅在 GPU 上运行快速测试。仅在以下文件夹中的代码更新时才运行：`src`，     `tests`，`.github`（以防止在添加模型卡片、笔记本等时运行）。
   - [self-hosted runner](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml)：在 GPU 上运行正常和慢速测试     在 `tests` 和 `examples` 中：
```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

   结果可以在 [此处](https://github.com/huggingface/transformers/actions) 查看。


## 运行测试




### 选择要运行的测试

本文档详细介绍了如何运行测试的许多细节。如果阅读后还需要更多细节您可以在 [这里](https://docs.pytest.org/en/latest/usage.html) 找到。

下面是一些运行测试的最有用的方法。

运行全部：
```console
pytest
```

或者：
```bash
make test
```

请注意，后者定义为：
```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

告诉 pytest 执行以下操作：
- 运行与 CPU 内核数相同数量的测试进程（如果没有大量 RAM，则可能太多）- 确保来自同一文件的所有测试将由同一个测试进程运行- 不捕获输出- 以详细模式运行


### 获取所有测试的列表

测试套件的所有测试：
```bash
pytest --collect-only -q
```

给定测试文件的所有测试：
```bash
pytest tests/test_optimization.py --collect-only -q
```

### 运行特定的测试模块

要运行单个测试模块：
```bash
pytest tests/test_logging.py
```

### 运行特定的测试

由于大多数测试内部使用 unittest，要运行特定的子测试，您需要知道包含这些测试的 unittest 类的名称。例如，它可以是：
```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里：
- `tests/test_optimization.py` - 包含测试的文件- `OptimizationTest` - 类的名称- `test_adam_w` - 特定测试函数的名称
如果文件包含多个类，则可以选择仅运行给定类的测试。例如：
```bash
pytest tests/test_optimization.py::OptimizationTest
```

将运行该类中的所有测试。
如前所述，您可以通过运行以下命令来查看 `OptimizationTest` 类中包含哪些测试。
```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

您可以按关键字表达式运行测试。
仅运行名称包含 `adam` 的测试：
```bash
pytest -k adam tests/test_optimization.py
```

可以使用逻辑 `and` 和 `or` 来指示是否所有关键字都应匹配或任意关键字都应匹配。可以使用 `not` 进行否定。
仅运行名称不包含 `adam` 的所有测试：
```bash
pytest -k "not adam" tests/test_optimization.py
```

并且您可以将两个模式组合成一个：
```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要同时运行 `test_adafactor` 和 `test_adam_w`，您可以使用：
```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

请注意，这里我们使用 `or`，因为我们希望关键字中的任何一个都匹配以包括两个测试。
如果您只想包括同时包含两个模式的测试，则应使用 `and`：
```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 运行 `accelerate` 测试
有时，您需要在模型上运行 `accelerate` 测试。为此，您只需在命令中添加 `-m accelerate_tests`，例如，如果要在 `OPT` 上运行这些测试，请运行：
```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py 
```


### 运行文档测试

为了测试文档示例是否正确，您应该检查 `doctests` 是否通过。例如，让我们使用 [`WhisperModel.forward` 的 docstring](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035)：
```python 
r"""
Returns:

Example:
``` python    >>> import torch
    >>> from transformers import WhisperModel, WhisperFeatureExtractor
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split = "validation")
    >>> inputs = feature_extractor(ds [0]["audio"] ["array"], return_tensors = "pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids = decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
```"""
```

只需运行以下行以自动测试所需文件中的每个 docstring 示例：
```bash 
pytest --doctest-modules <path_to_file_or_dir>
```

如果文件具有 markdown 扩展名，则应添加 `--doctest-glob="*.md"` 参数。

### 仅运行已修改的测试

您可以通过使用 [pytest-picked](https://github.com/anapaulagomes/pytest-picked) 运行与未暂存文件或当前分支（根据 Git）相关的测试来测试您的更改是否破坏了任何内容，因为它不会运行与您未更改的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

将从已修改但尚未提交的文件和文件夹中运行所有测试。

### 在源代码修改时自动重新运行失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) 提供了一项非常有用的功能，可以检测所有失败的测试，然后等待您修改文件并不断重新运行这些失败的测试，直到它们在您修复它们时通过为止。因此，您无需在进行修复后重新启动 pytest。在执行完全运行之前，这将重复进行。which again a full run is performed.

```bash
pip install pytest-xdist
```

进入此模式：`pytest -f` 或 `pytest --looponfail`
通过检查 `looponfailroots` 根目录及其所有内容（递归）来检测文件更改。如果默认值对您不起作用，您可以在项目中通过在 `setup.cfg` 中设置配置选项来更改它：
```ini
[tool:pytest]
looponfailroots = transformers tests
```

或者 `pytest.ini`/``tox.ini`` 文件：
```ini
[pytest]
looponfailroots = transformers tests
```

这将导致仅查找相对于 ini 文件目录的相应目录中的文件更改。[pytest-watch](https://github.com/joeyespo/pytest-watch) 是此功能的另一种实现方式。
[pytest-watch](https://github.com/joeyespo/pytest-watch) is an alternative implementation of this functionality.


### 跳过测试模块

如果您想运行所有测试模块，除了一些测试模块，您可以通过提供一个明确的测试列表来排除它们。例如，要运行除了 `test_modeling_*.py` 测试之外的所有测试：

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### 清除状态

在 CI 构建和需要隔离（针对速度）的情况下，应清除缓存：
```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述，`make test` 通过 `pytest-xdist` 插件以并行方式运行测试（使用 `-n X` 参数，例如 `-n 2` 以运行 2 个并行作业）。

`pytest-xdist` 的 `--dist=` 选项允许控制测试的分组方式。`--dist=loadfile` 将位于同一文件中的测试放置在同一进程中。因为执行测试的顺序是不同且不可预测的，如果使用 `pytest-xdist` 运行测试套件时出现失败（表示存在一些未检测到的耦合测试），可以使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 以相同顺序重放测试，这应该有助于将失败序列减少到最小。

### 测试顺序和重复多次

重复运行测试是很好的做法，可以按顺序、随机或成组多次运行，以便检测任何潜在的相互依赖和与状态相关的错误（拆除）。此外，直接多次重复运行也有助于发现由于深度学习的随机性而暴露出的一些问题。#### 重复运行测试
- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder)：

然后将每个测试运行多次（默认为 50 次）：


#### Repeat tests

<Tip>
```bash
pip install pytest-flakefinder
```

And then run every test multiple times (50 by default):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

此插件不适用于 `pytest-xdist` 的 `-n` 标志。
</Tip>
<Tip>
还有另一个插件 `pytest-repeat`，但它不适用于 `unittest`。
</Tip>

#### 随机顺序运行测试
```bash
pip install pytest-random-order
```

重要提示：只要安装了 `pytest-random-order`，就会自动随机排列测试，无需进行任何配置更改或命令行选项。command line options is required.

如前所述，这样可以检测到耦合测试，其中一个测试的状态会影响另一个测试的状态。

当安装了 `pytest-random-order` 时，它将打印用于该会话的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

因此，如果给定的特定序列失败，可以通过添加该精确的种子来复现它，例如：
```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

只有在使用相同的测试列表（或不使用任何列表）时，它才会复现完全相同的顺序。一旦开始手动缩小列表，就不再依赖种子，而是必须按照它们失败的确切顺序手动列出它们，并告诉 pytest 不要随机排列它们，而是使用 `--random-order-bucket=none`，例如：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要禁用所有测试的洗牌：
```bash
pytest --random-order-bucket=none
```

默认情况下，`--random-order-bucket=module` 被隐式地应用，它将在模块级别对文件进行洗牌。它还可以在 `class`、`package`、`global` 和 `none` 级别进行洗牌。有关完整详情，请参阅其[文档](https://github.com/jbasko/pytest-random-order)。

另一种随机化的选择是：[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)。这个模块具有非常类似的功能/接口，但它不具备 `pytest-random-order` 中的 bucket 模式。它在安装后也会自动生效。

### 外观变化
#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) 是一个改善外观和添加进度条的插件，并立即显示失败和断言的插件。安装后会自动激活。

```bash
pip install pytest-sugar
```

要在没有 pytest-sugar 的情况下运行测试，请运行：
```bash
pytest -p no:sugar
```

或卸载它。


#### 报告每个子测试名称及其进度

对于通过 `pytest` 运行的单个或一组测试（在 `pip install pytest-pspec` 后）：
```bash
pytest --pspec tests/test_optimization.py
```

#### 立即显示失败的测试
[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) 立即显示失败和错误，而不是等到测试会话结束。waiting until the end of test session.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### 使用 GPU 还是不使用 GPU
在启用 GPU 的设置上，要以仅 CPU 模式进行测试，请添加 `CUDA_VISIBLE_DEVICES=""`：
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/test_logging.py
```

如果您有多个 GPU，可以指定 `pytest` 要使用的 GPU。例如，如果您有 GPU `0` 和 `1`，可以运行：second gpu if you have gpus `0` and `1`, you can run:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/test_logging.py
```

这在您想要在不同 GPU 上运行不同任务时非常方便。
某些测试必须在仅使用 CPU 上运行，另一些测试必须在 CPU 或 GPU 或 TPU 上运行，还有一些测试必须在多个 GPU 上运行。以下是用于设置测试的 CPU/GPU/TPU 要求的跳过装饰器：

- `require_torch` - 仅在 torch 下运行此测试- `require_torch_gpu` - 与 `require_torch` 相同，还需要至少 1 个 GPU- `require_torch_multi_gpu` - 与 `require_torch` 相同，还需要至少 2 个 GPU- `require_torch_non_multi_gpu` - 与 `require_torch` 相同，还需要 0 或 1 个 GPU- `require_torch_up_to_2_gpus` - 与 `require_torch` 相同，还需要 0 或 1 或 2 个 GPU- `require_torch_tpu` - 与 `require_torch` 相同，还需要至少 1 个 TPU
让我们在下表中描述 GPU 要求：

| n 个 GPU | 装饰器                           ||---------+--------------------------------|| `>= 0`  | `@require_torch`               || `>= 1`  | `@require_torch_gpu`           || `>= 2`  | `@require_torch_multi_gpu`     || `< 2`   | `@require_torch_non_multi_gpu` || `< 3`   | `@require_torch_up_to_2_gpus`  |

例如，下面是一个只有在有 2 个或更多个可用 GPU 并且已安装 pytorch 的情况下才能运行的测试：
```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

如果测试需要 `tensorflow`，请使用 `require_tf` 装饰器。例如：
```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

这些装饰器可以叠加使用。例如，如果一个测试很慢并且在 pytorch 下至少需要一个 GPU，则可以进行如下设置：how to set it up:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

对于 `@parametrized` 等重写测试名称的装饰器，必须将 `@require_*` 跳过装饰器列在最后以确保其正确工作。以下是正确使用的示例：last for them to work correctly. Here is an example of the correct usage:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

这个顺序问题在 `@pytest.mark.parametrize` 中不存在，您可以将它放在第一位或最后一位，它仍然可以正常工作。但它只适用于非单元测试。work. But it only works with non-unittests.

在测试内部：
- 可用的 GPU 数量：
```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  # works with torch and tf
```

### 分布式训练

`pytest` 无法直接处理分布式训练。如果尝试这样做，子进程会执行错误的操作，导致它们认为它们是 `pytest` 并开始循环运行测试套件。但是，如果一个普通进程生成一个子进程并管理 IO 管道，则可以正常工作。


以下是使用它的一些测试：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)
要直接跳转到执行点，请在这些测试中搜索 `execute_subprocess_async` 调用。
您至少需要 2 个 GPU 才能看到这些测试的效果：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### 输出捕获

在测试执行期间，发送到 `stdout` 和 `stderr` 的任何输出都将被捕获。如果测试或设置方法失败，其相应的捕获输出通常将与失败的回溯一起显示。

要禁用输出捕获并正常获取 `stdout` 和 `stderr`，请使用 `-s` 或 `--capture=no`：

```bash
pytest -s tests/test_logging.py
```

将测试结果发送到 JUnit 格式的输出：
```bash
py.test tests --junitxml=result.xml
```

### 颜色控制

要取消颜色（例如，白色背景上的黄色不可读）：
```bash
pytest --color=no tests/test_logging.py
```

### 将测试报告发送到在线粘贴服务

为每个测试失败创建一个 URL：
```bash
pytest --pastebin=failed tests/test_logging.py
```

这将向远程 Paste 服务提交测试运行信息，并为每个失败提供一个 URL。您可以像平常一样选择测试，或者如果您只想发送一个特定的失败，则可以添加例如-x。创建整个测试会话日志的 URL:

## 编写测试
```bash
pytest --pastebin=all tests/test_logging.py
```

🤗 transformers 测试基于 `unittest`，但由 `pytest` 运行，因此大部分时间都可以使用这两个系统的功能。

可用。您可以在 [这里](https://docs.pytest.org/en/stable/unittest.html) 阅读支持的功能，但重要的是要记住，大多数 `pytest` fixtures 不起作用。既不支持参数化，但我们使用的是模块`parameterized` 以类似的方式工作。### 参数化 `parameterized` that works in a similar way.


### Parametrization

通常，需要多次运行相同的测试，但使用不同的参数。可以从测试内部完成，但是那样就无法仅运行该测试的一组参数。现在，默认情况下，此测试将运行 3 次，每次都将最后 3 个 `test_floor` 的参数分配给参数列表中的相应参数。
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

Now, by default this test will be run 3 times, each time with the last 3 arguments of `test_floor` being assigned the
相应参数分配给参数列表中的相应参数。
您可以使用以下方式仅运行 `negative` 和 `integer` 参数集：
```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或运行除了 `negative` 子测试以外的所有子测试：
```bash
pytest -k "not negative" tests/test_mytest.py
```

除了使用刚才提到的 `-k` 过滤器之外，还可以找出每个子测试的确切名称，并使用它们的确切名称运行任何一个或全部。它们的确切名称。
```bash
pytest test_this1.py --collect-only -q
```

and it will list:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

因此，现在您可以仅运行 2 个特定的子测试：
```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

模块 [parameterized](https://pypi.org/project/parameterized/) 已经在开发者依赖项中，可以同时用于 `unittest` 和 `pytest` 测试。

如果测试不是 `unittest`，则可以使用 `pytest.mark.parametrize`（或者您可以在某些现有测试中看到它的使用，主要在 `examples` 下）。

这是同样的例子，这次使用 `pytest` 的 `parametrize` 标记：some existing tests, mostly under `examples`).

Here is the same example, this time using `pytest`'s `parametrize` marker:

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



如果 `-k` 过滤器无法完成任务，则使用 `pytest.mark.parametrize` 可以对子测试进行精细控制。除此之外，此参数化函数为子测试创建了稍微不同的一组名称。它们如下所示：
```bash
pytest test_this2.py --collect-only -q
```

and it will list:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

因此，现在您可以仅运行特定的测试：
```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

与前一个示例一样。


### 文件和目录

在测试中，我们经常需要知道相对于当前测试文件的位置，这并不容易，因为测试可能从多个目录调用，或者可能位于不同深度的子目录中。

辅助类 `transformers.test_utils.TestCasePlus` 通过解决所有基本路径问题并提供简单的访问器来解决此问题：

- `pathlib` 对象（全部已解析）：
  - `test_file_path` - the current test file path, i.e. `__file__`
  - `test_file_dir` - the directory containing the current test file
  - `tests_dir` - the directory of the `tests` test suite
  - `examples_dir` - the directory of the `examples` test suite
  - `repo_root_dir` - the directory of the repository
  - `src_dir` - the directory of `src` (i.e. where the `transformers` sub-dir resides)
- 字符串化的路径---与上述路径相同，但将路径作为字符串返回，而不是 `pathlib` 对象：
  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`
要开始使用这些，您只需要确保测试位于 `transformers.test_utils.TestCasePlus` 的子类中。例如：

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果您不需要通过 `pathlib` 操纵路径，或者您只需要路径作为字符串，您可以始终在 `pathlib` 对象上调用 `str()`，或者使用以 `_str` 结尾的访问器。例如：
```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### 临时文件和目录
使用唯一的临时文件和目录对于并行测试运行至关重要，以防测试互相覆盖彼此的数据。此外，我们希望在创建每个创建的测试结束时删除临时文件和目录。因此，使用解决这些需求的 `tempfile` 等软件包是必不可少的。them. Therefore, using packages like `tempfile`, which address these needs is essential.

但是，当调试测试时，您需要能够查看临时文件或目录中的内容，并且希望知道其确切路径，而不是在每次测试重新运行时随机化。
辅助类 `transformers.test_utils.TestCasePlus` 最适合此类目的。它是 `unittest.TestCase` 的子类，因此我们可以轻松继承它在测试模块中。
以下是其使用示例：
```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

此代码创建了一个唯一的临时目录，并将 `tmp_dir` 设置为其位置。
- 创建唯一的临时目录：
```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` 将包含所创建临时目录的路径。它将在测试结束时自动删除。test.

- 创建我选择的临时目录，在测试开始之前确保它为空，并在测试之后不清空它。
```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

当您希望监视特定目录并确保之前的测试没有在其中留下任何数据时，这非常有用。leave any data in there.

- 您可以通过直接覆盖 `before` 和 `after` 参数来覆盖默认行为，从而实现以下行为之一：  - `before=True`：始终在测试开始时清除临时目录。
  - `before=True`: the temporary dir will always be cleared at the beginning of the test.
  - `before=False`：如果临时目录已经存在，则任何现有文件将保留在那里。  - `after=True`：始终在测试结束时删除临时目录。  - `after=False`：始终在测试结束时保留临时目录。
<Tip>

<Tip> 为了安全地运行等效于 `rm -r`，如果使用了显式的 `tmp_dir`，则只允许项目存储库检出的子目录，以免错误地清除 `/tmp` 或类似的重要文件系统的一部分。即始终传递以 `./` 开头的路径。
</Tip>
<Tip>

每个测试可以注册多个临时目录，除非另有要求，否则它们都将被自动删除。otherwise.

</Tip>

### 临时 sys.path 覆盖
如果您需要临时覆盖 `sys.path` 以从另一个测试中导入，例如，可以使用 `ExtendSysPath` 上下文管理器。示例：`ExtendSysPath` context manager. Example:


```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试

当发现错误并编写新测试时，但错误尚未修复时，跳过测试非常有用。为了能够将其提交到主存储库，我们需要确保在 `make test` 期间跳过它。
方法：

- **skip** 表示仅当满足某些条件时，您的测试才能通过，否则 pytest 将跳过运行整个测试。常见示例是在非 Windows 平台上跳过仅  适用于 Windows 的测试，或者跳过依赖于当前不可用的外部资源的测试（例如数据库）。  

- **xfail** 表示出于某种原因，您期望测试失败。常见示例是尚未实现的功能的测试，或尚未修复的错误。当测试通过  时，尽管预期失败（标记为  pytest.mark.xfail），这是一个 xpass，将在测试摘要中报告。
两者之间的一个重要区别是 `skip` 不会运行测试，而 `xfail` 会。因此，如果有错误的代码会导致一些影响其他测试的错误状态，请不要使用 `xfail`。#### 实施
- 这里是如何无条件跳过整个测试的方法：
- Here is how to skip whole test unconditionally:

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

或通过 pytest：
```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

或者使用 `xfail` 方式：
```python no-style
@pytest.mark.xfail
def test_feature_x():
```

- 这里是如何根据测试内部的某些内部检查跳过一个测试的方法：
```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

或整个模块：
```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

或者使用 `xfail` 方式：
```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 这里是如何在模块中跳过所有测试，如果某个导入丢失：
```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 根据条件跳过测试：
```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

或者：
```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

或者跳过整个模块：
```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

更多细节、示例和方法可以在 [此处](https://docs.pytest.org/en/latest/skipping.html) 找到。

### 慢速测试

测试库正在不断增长，一些测试需要几分钟才能运行，因此我们不能等待测试套件在持续集成上完成一小时。因此，除了关键测试外，慢速测试应该被标记为：如下例所示：

一旦测试被标记为 `@slow`，要运行这些测试，请设置 `RUN_SLOW=1` 环境变量，例如：

```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

某些修饰符（如 `@parameterized`）会重写测试名称，因此 `@slow` 和其他跳过修饰符必须以正确的顺序列出才能正常工作。以下是正确使用的示例：
```bash
RUN_SLOW=1 pytest tests
```

Some decorators like `@parameterized` rewrite test names, therefore `@slow` and the rest of the skip decorators
`@require_*` have to be listed last for them to work correctly. Here is an example of the correct usage:

```python no-style
@parameteriz ed.expand(...)
@slow
def test_integration_foo():
```

如本文档开头所解释的，在计划的基础上运行慢速测试，而不是在 PR 的 CI 检查中运行。因此，在 PR 提交和合并之前，在本地运行慢速测试非常重要。这意味着可能会在 PR 提交时错过一些问题并进行合并。此类问题将在下一个计划的 CI 作业中被发现。但这也意味着在提交 PR 之前在本地运行慢速测试非常重要。这里有一个大致的决策机制，用于选择哪些测试应标记为慢速：

如果测试侧重于库的内部组件之一（例如，建模文件、分词文件、流程等），则应在非慢速测试套件中运行该测试。如果测试侧重于库的其他方面，例如文档或示例，则应在慢速测试套件中运行这些测试。然后，为了细化这一方法，我们应该有一些例外情况:

- 所有需要下载大量权重集或大于~50MB 的数据集的测试（例如，模型或分词器 (Tokenizer)集成测试，流程集成测试）应设置为慢速。如果要添加新模型，应创建并上传到 Hub 的一个微型版本（带有随机权重）进行集成测试。这在以下段落中讨论。  
- 所有需要进行非特定优化的训练的测试应设置为慢速。- 如果某些应该不是慢速测试的测试非常慢，可以设置它们为 `@slow`。自动建模测试会将大型文件保存到磁盘并加载，这是标记为 `@slow` 的测试的一个很好的示例。  `@slow`. 
- 如果测试在 CI 上运行时间不到 1 秒（包括下载，如果有的话），那么它应该是一个普通测试。

总的来说，所有非慢速测试需要完全覆盖不同的内部功能，同时保持快速。例如，可以通过使用具有随机权重的特殊创建的微型模型进行测试来实现较高的覆盖率。这些模型具有最小数量的层（例如，2），词汇量（例如，1000）等。然后，`@slow` 测试可以使用大型慢速模型进行定性测试。要查看这些的使用，请搜索带有“tiny”的模型：

```bash
grep tiny tests examples
```

这里有一个 [脚本示例](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py) 创建了微型模型 [stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de)。您可以根据自己模型的架构轻松进行调整。model's architecture.

如果例如存在下载巨大模型的开销，很容易错误地测量运行时间，但如果在本地测试，下载的文件将被缓存，因此不会测量下载时间。因此，请检查 CI 日志中的执行速度报告（`pytest --durations=0 tests` 的输出）。


该报告还有助于找到未标记为慢速的慢速异常值，或者需要重新编写以提高速度的异常值。如果您注意到测试套件在 CI 上开始变慢，该报告的前几行将显示最慢的测试。



### 测试 stdout/stderr 输出

为了测试向 `stdout` 和/或 `stderr` 写入的函数，可以使用 `pytest` 的 [capsys 系统](https://docs.pytest.org/en/latest/capture.html)。以下是如何实现的：`pytest`'s [capsys system](https://docs.pytest.org/en/latest/capture.html). 


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

当然，大多数情况下，`stderr` 将作为异常的一部分，因此在这种情况下必须使用 try/except：a case:

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

通过 `contextlib.redirect_stdout` 也可以捕获 stdout：
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

捕获 stdout 的一个重要潜在问题是它可能包含 `\r` 字符，这在正常的 `print` 中会重置到目前为止打印的所有内容。`pytest` 没有问题，但是在 `pytest -s` 中，这些字符会包含在缓冲区中，因此为了能够在有和没有 `-s` 的情况下运行测试，必须对捕获的输出进行额外的清理，使用 `re.sub(r'~.*\r', '', buf, 0, re.M)`。


但是，我们还有一个辅助的上下文管理器包装器，可以自动处理所有这一切，无论它是否包含一些 `\r`，都是一个简单的：

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

这里有一个完整的测试示例：
```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

如果您想捕获 `stderr`，请改用 `CaptureStderr` 类：
```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

如果您需要同时捕获两个流，请使用父类 `CaptureStd`：
```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

此外，为了帮助调试测试问题，默认情况下，这些上下文管理器会在退出上下文时自动回放捕获的流。


### 捕获日志记录器流

如果需要验证日志记录器的输出，可以使用 `CaptureLogger`：
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

### 使用环境变量进行测试

如果要测试特定测试的环境变量的影响，可以使用辅助装饰器 `transformers.testing_utils.mockenv` 


```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

有时需要调用外部程序，该程序需要在 `os.environ` 中设置 `PYTHONPATH` 以包含多个本地路径。一个辅助类 `transformers.test_utils.TestCasePlus` 来帮助：


```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

根据测试文件是否位于 `tests` 测试套件或 `examples` 下，它将正确设置 `env[PYTHONPATH]` 以包括这两个目录之一，还会设置 `src` 目录以确保测试针对的是当前的 repo，最后是在调用测试之前的任何情况下设置的 `env[PYTHONPATH]`。


此辅助方法创建了 `os.environ` 对象的副本，因此原始副本保持不变。

### 获取可重现的结果

在某些情况下，您可能希望删除测试的随机性。为了获得相同的可重现结果，您需要固定种子：

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

### 调试测试

要在警告点启动调试器，请执行以下操作：
```bash
pytest tests/test_logging.py -W error::UserWarning --pdb
```

## 使用 github actions 工作流

要触发自助推送工作流 CI 作业，您必须：
1. 在 `transformers` 源上创建一个新的分支（非派生！）。
2. 分支名称必须以 `ci_` 或 `ci-` 开头（`main` 也会触发它，但我们不能在 `main` 上进行 PR），它仅在特定路径下触发-您可以在此处找到最新的定义，以防自从编写这份文档以来它发生了更改 [这里](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml) 的 *push:* 下
3. 从此分支创建一个 PR。
4. 然后您可以在 [这里](https://github.com/huggingface/transformers/actions/workflows/self-push.yml) 看到作业出现。如果有积压，它可能不会立即运行。
## 测试实验性 

CI 功能测试 CI 功能可能会产生问题，因为它可能会干扰正常的 CI 功能。因此，如果要添加新的 CI 功能，应按照以下方式进行。
1. 创建一个测试所需内容的新的专用作业。
2. 新作业必须始终成功，以便给出绿色✓（详细信息如下）。
3. 让其运行几天，以便各种不同类型的 PR 在其上运行（用户派生分支，非派生分支，源自 github.com UI 直接文件编辑的分支，各种强制推送等-有   很多）同时监视实验作业的日志（而不是整体作业绿色，因为它始终是绿色的）
4. 当一切都稳定后，将新更改合并到现有作业中。这样一来，对 CI 功能本身的实验就不会干扰正常的工作流程。现在，在开发新的 CI 功能时，如何使作业始终成功？某些 CI，如 TravisCI 支持 ignore-step-failure，并将整体作业报告为成功，但是 CircleCI 和 Github Actions 在编写本文时不支持此功能。

因此，可以使用以下解决方法：
1. 在运行命令的开头使用 `set +euo pipefail` 来抑制 bash 脚本中的大多数潜在故障。
2. 最后一个命令必须成功：`echo "done"` 或只是 `true` 即可。
下面是一个示例：对于简单的命令，您也可以执行以下操作：当然，一旦对结果满意，将实验步骤或作业与其他正常作业集成在一起，同时删除 `set +euo pipefail` 或您可能添加的任何其他内容，以确保实验作业不会干扰正常的 CI 功能。
如果能够像设置 `allow-failure` 一样设置实验步骤，让它失败而不影响 PR 的整体状态，那么整个过程将更加简单。但正如前面提到的，CircleCI 和 Github Actions 目前不支持此功能。
您可以为此功能投票，并查看这些 CI 特定的主题所在位置：- [Github Actions:](https://github.com/actions/toolkit/issues/399)- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344) 





这是案例：

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

对于简单的命令，你也可以这样做：

```bash
cmd_that_may_fail || true
```

当然，一旦对结果满意，将实验性的步骤或作业与其他正常的作业集成在一起，同时删除 `set +euo pipefail` 或任何其他你可能添加的内容，以确保实验性作业不会影响正常 CI 的运行。

如果我们能够为实验性步骤设置类似于 `allow-failure` 的选项，让它在不影响 PR 总体状态的情况下失败，整个过程将更加容易。但正如前面提到的，CircleCI 和 Github Actions 目前不支持此功能。

你可以在这些 CI 相关的讨论中投票支持该功能，并了解其进展情况：

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)
