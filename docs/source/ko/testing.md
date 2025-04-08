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

# 테스트[[testing]]


먼저 🤗 Transformers 모델이 어떻게 테스트되는지 살펴보고, 새로운 테스트를 작성 및 기존 테스트를 개선하는 방법을 알아봅시다.

이 저장소에는 2개의 테스트 스위트가 있습니다:

1. `tests` - 일반 API에 대한 테스트
2. `examples` - API의 일부가 아닌 다양한 응용 프로그램에 대한 테스트

## Transformers 테스트 방법[[how-transformers-are-tested]]

1. PR이 제출되면 9개의 CircleCi 작업으로 테스트가 진행됩니다. 해당 PR에 대해 새로운 커밋이 생성될 때마다 테스트는 다시 진행됩니다. 이 작업들은
   이 [config 파일](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml)에 정의되어 있으므로 필요하다면
   사용자의 로컬 환경에서 동일하게 재현해 볼 수 있습니다.

   이 CI 작업은 `@slow` 테스트를 실행하지 않습니다.

2. [github actions](https://github.com/huggingface/transformers/actions)에 의해 실행되는 작업은 3개입니다:

   - [torch hub integration](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml):
    torch hub integration이 작동하는지 확인합니다.

   - [self-hosted (push)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): `main` 브랜치에서 커밋이 업데이트된 경우에만 GPU를 이용한 빠른 테스트를 실행합니다.
    이는 `src`, `tests`, `.github` 폴더 중 하나에 코드가 업데이트된 경우에만 실행됩니다.
    (model card, notebook, 기타 등등을 추가한 경우 실행되지 않도록 하기 위해서입니다)

   - [self-hosted runner](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): `tests` 및 `examples`에서
   GPU를 이용한 일반 테스트, 느린 테스트를 실행합니다.


```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

   결과는 [여기](https://github.com/huggingface/transformers/actions)에서 확인할 수 있습니다.


## 테스트 실행[[running-tests]]





### 실행할 테스트 선택[[choosing-which-tests-to-run]]

이 문서는 테스트를 실행하는 다양한 방법에 대해 자세히 설명합니다.
모든 내용을 읽은 후에도, 더 자세한 내용이 필요하다면 [여기](https://docs.pytest.org/en/latest/usage.html)에서 확인할 수 있습니다.

다음은 가장 유용한 테스트 실행 방법 몇 가지입니다.

모두 실행:

```console
pytest
```

또는:

```bash
make test
```

후자는 다음과 같이 정의됩니다:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

위의 명령어는 pytest에게 아래의 내용을 전달합니다:

- 사용 가능한 CPU 코어 수만큼 테스트 프로세스를 실행합니다. (RAM이 충분하지 않다면, 테스트 프로세스 수가 너무 많을 수 있습니다!)
- 동일한 파일의 모든 테스트는 동일한 테스트 프로세스에서 실행되어야 합니다.
- 출력을 캡처하지 않습니다.
- 자세한 모드로 실행합니다.



### 모든 테스트 목록 가져오기[[getting-the-list-of-all-tests]]

테스트 스위트의 모든 테스트:

```bash
pytest --collect-only -q
```

지정된 테스트 파일의 모든 테스트:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### 특정 테스트 모듈 실행[[run-a-specific-test-module]]

개별 테스트 모듈 실행하기:

```bash
pytest tests/utils/test_logging.py
```

### 특정 테스트 실행[[run-specific-tests]]

대부분의 테스트 내부에서는 unittest가 사용됩니다. 따라서 특정 하위 테스트를 실행하려면 해당 테스트를 포함하는 unittest 클래스의 이름을 알아야 합니다.
예를 들어 다음과 같을 수 있습니다:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

위의 명령어의 의미는 다음과 같습니다:

- `tests/test_optimization.py` - 테스트가 있는 파일
- `OptimizationTest` - 클래스의 이름
- `test_adam_w` - 특정 테스트 함수의 이름

파일에 여러 클래스가 포함된 경우, 특정 클래스의 테스트만 실행할 수도 있습니다. 예를 들어 다음과 같습니다:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

이 명령어는 해당 클래스 내부의 모든 테스트를 실행합니다.

앞에서 언급한 것처럼 `OptimizationTest` 클래스에 포함된 테스트를 확인할 수 있습니다.

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

키워드 표현식을 사용하여 테스트를 실행할 수도 있습니다.

`adam`이라는 이름을 포함하는 테스트만 실행하려면 다음과 같습니다:

```bash
pytest -k adam tests/test_optimization.py
```

논리 연산자 `and`와 `or`를 사용하여 모든 키워드가 일치해야 하는지 또는 어느 하나가 일치해야 하는지를 나타낼 수 있습니다.
`not`은 부정할 때 사용할 수 있습니다.

`adam`이라는 이름을 포함하지 않는 모든 테스트를 실행하려면 다음과 같습니다:

```bash
pytest -k "not adam" tests/test_optimization.py
```

두 가지 패턴을 하나로 결합할 수도 있습니다:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

예를 들어 `test_adafactor`와 `test_adam_w`를 모두 실행하려면 다음을 사용할 수 있습니다:

```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

여기서 `or`를 사용하는 것에 유의하세요. 두 키워드 중 하나가 일치하도록 하기 위한 목적으로 사용하기 때문입니다.

두 패턴이 모두 포함되어야 하는 테스트만 실행하려면, `and`를 사용해야 합니다:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### `accelerate` 테스트 실행[[run-`accelerate`-tests]]

모델에서 `accelerate` 테스트를 실행해야 할 때가 있습니다. 이를 위해서는 명령어에 `-m accelerate_tests`를 추가하면 됩니다.
예를 들어, `OPT`에서 이러한 테스트를 실행하려면 다음과 같습니다:
```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py
```

### 문서 테스트 실행[[run-documentation-tests]]

예시 문서가 올바른지 테스트하려면 `doctests`가 통과하는지 확인해야 합니다.
예를 들어, [`WhisperModel.forward`'s docstring](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035)를 사용해 봅시다:

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

원하는 파일의 모든 docstring 예제를 자동으로 테스트하려면 다음 명령을 실행하면 됩니다:
```bash
pytest --doctest-modules <path_to_file_or_dir>
```
파일의 확장자가 markdown인 경우 `--doctest-glob="*.md"` 인수를 추가해야 합니다.

### 수정된 테스트만 실행[[run-only-modified-tests]]

수정된 파일 또는 현재 브랜치 (Git 기준)와 관련된 테스트를 실행하려면 [pytest-picked](https://github.com/anapaulagomes/pytest-picked)을 사용할 수 있습니다.
이는 변경한 내용이 테스트에 영향을 주지 않았는지 빠르게 확인할 수 있는 좋은 방법입니다.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

수정되었지만, 아직 커밋되지 않은 모든 파일 및 폴더에서 테스트가 실행됩니다.

### 소스 수정 시 실패한 테스트 자동 재실행[[automatically-rerun-failed-tests-on-source-modification]]

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)는 모든 실패한 테스트를 감지하고,
파일을 수정한 후에 파일을 계속 재실행하여 테스트가 성공할 때까지 기다리는 매우 유용한 기능을 제공합니다.
따라서 수정한 내용을 확인한 후 pytest를 다시 시작할 필요가 없습니다.
모든 테스트가 통과될 때까지 이 과정을 반복한 후 다시 전체 실행이 이루어집니다.

```bash
pip install pytest-xdist
```

재귀적 모드의 사용: `pytest -f` 또는 `pytest --looponfail`

파일의 변경 사항은 `looponfailroots` 루트 디렉터리와 해당 내용을 (재귀적으로) 확인하여 감지됩니다.
이 값의 기본값이 작동하지 않는 경우,
`setup.cfg`의 설정 옵션을 변경하여 프로젝트에서 변경할 수 있습니다:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

또는 `pytest.ini`/`tox.ini`` 파일:

```ini
[pytest]
looponfailroots = transformers tests
```

이렇게 하면 ini-file의 디렉터리를 기준으로 상대적으로 지정된 각 디렉터리에서 파일 변경 사항만 찾게 됩니다.


이 기능을 대체할 수 있는 구현 방법인 [pytest-watch](https://github.com/joeyespo/pytest-watch)도 있습니다.


### 특정 테스트 모듈 건너뛰기[[skip-a-test-module]]

모든 테스트 모듈을 실행하되 특정 모듈을 제외하려면, 실행할 테스트 목록을 명시적으로 지정할 수 있습니다.
예를 들어, `test_modeling_*.py` 테스트를 제외한 모든 테스트를 실행하려면 다음을 사용할 수 있습니다:

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### 상태 초기화[[clearing state]]

CI 빌드 및 (속도에 대한) 격리가 중요한 경우, 캐시를 지워야 합니다:

```bash
pytest --cache-clear tests
```

### 테스트를 병렬로 실행[[running-tests-in-parallel]]

이전에 언급한 것처럼 `make test`는 테스트를 병렬로 실행하기 위해
`pytest-xdist` 플러그인(`-n X` 인수, 예를 들어 `-n 2`를 사용하여 2개의 병렬 작업 실행)을 통해 실행됩니다.

`pytest-xdist`의 `--dist=` 옵션을 사용하여 테스트를 어떻게 그룹화할지 제어할 수 있습니다.
`--dist=loadfile`은 하나의 파일에 있는 테스트를 동일한 프로세스로 그룹화합니다.

실행된 테스트의 순서가 다르고 예측할 수 없기 때문에, `pytest-xdist`로 테스트 스위트를 실행하면 실패가 발생할 수 있습니다 (검출되지 않은 결합된 테스트가 있는 경우).
이 경우 [pytest-replay](https://github.com/ESSS/pytest-replay)를 사용하면 동일한 순서로 테스트를 다시 실행해서
실패하는 시퀀스를 최소화하는 데에 도움이 됩니다.

### 테스트 순서와 반복[[test-order-and-repetition]]

잠재적인 종속성 및 상태 관련 버그(tear down)를 감지하기 위해
테스트를 여러 번, 연속으로, 무작위로 또는 세트로 반복하는 것이 좋습니다.
그리고 직접적인 여러 번의 반복은 DL의 무작위성에 의해 발견되는 일부 문제를 감지하는 데에도 유용합니다.


#### 테스트를 반복[[repeat-tests]]

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

모든 테스트를 여러 번 실행합니다(기본값은 50번):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

이 플러그인은 `pytest-xdist`의 `-n` 플래그와 함께 작동하지 않습니다.

</Tip>

<Tip>

`pytest-repeat`라는 또 다른 플러그인도 있지만 `unittest`와 함께 작동하지 않습니다.

</Tip>

#### 테스트를 임의의 순서로 실행[[run-tests-in-a-random-order]]

```bash
pip install pytest-random-order
```

중요: `pytest-random-order`가 설치되면 테스트가 자동으로 임의의 순서로 섞입니다.
구성 변경이나 커맨드 라인 옵션이 필요하지 않습니다.

앞서 설명한 것처럼 이를 통해 한 테스트의 상태가 다른 테스트의 상태에 영향을 미치는 결합된 테스트를 감지할 수 있습니다.
`pytest-random-order`가 설치되면 해당 세션에서 사용된 랜덤 시드가 출력되며 예를 들어 다음과 같습니다:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

따라서 특정 시퀀스가 실패하는 경우에는 정확한 시드를 추가하여 재현할 수 있습니다. 예를 들어 다음과 같습니다:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

정확히 동일한 테스트 목록(또는 목록이 없음)을 사용하는 경우에만 정확한 순서를 재현합니다.
목록을 수동으로 좁히기 시작하면 더 이상 시드에 의존할 수 없고 실패했던 정확한 순서로 수동으로 목록을 나열해야합니다. 그리고 `--random-order-bucket=none`을 사용하여 pytest에게 순서를 임의로 설정하지 않도록 알려야 합니다.
예를 들어 다음과 같습니다:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

모든 테스트에 대해 섞기를 비활성화하려면 다음과 같습니다:

```bash
pytest --random-order-bucket=none
```

기본적으로 `--random-order-bucket=module`이 내재되어 있으므로, 모듈 수준에서 파일을 섞습니다.
또한 `class`, `package`, `global` 및 `none` 수준에서도 섞을 수 있습니다.
자세한 내용은 해당 [문서](https://github.com/jbasko/pytest-random-order)를 참조하세요.

또 다른 무작위화의 대안은 [`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)입니다.
이 모듈은 매우 유사한 기능/인터페이스를 가지고 있지만, `pytest-random-order`에 있는 버킷 모드를 사용할 수는 없습니다.
설치 후에는 자동으로 적용되는 문제도 동일하게 가집니다.

### 외관과 느낌을 변경[[look-and-feel-variations]

#### pytest-sugar 사용[[pytest-sugar]]

[pytest-sugar](https://github.com/Frozenball/pytest-sugar)는 테스트가 보여지는 형태를 개선하고,
진행 상황 바를 추가하며, 실패한 테스트와 검증을 즉시 표시하는 플러그인입니다. 설치하면 자동으로 활성화됩니다.

```bash
pip install pytest-sugar
```

pytest-sugar 없이 테스트를 실행하려면 다음과 같습니다:

```bash
pytest -p no:sugar
```

또는 제거하세요.



#### 각 하위 테스트 이름과 진행 상황 보고[[report-each-sub-test-name-and-its-progress]]

`pytest`를 통해 단일 또는 그룹의 테스트를 실행하는 경우(`pip install pytest-pspec` 이후):

```bash
pytest --pspec tests/test_optimization.py
```

#### 실패한 테스트 즉시 표시[[instantly-shows-failed-tests]]

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail)은 테스트 세션의 끝까지 기다리지 않고
실패 및 오류를 즉시 표시합니다.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### GPU 사용 여부[[to-GPU-or-not-to-GPU]]

GPU가 활성화된 환경에서, CPU 전용 모드로 테스트하려면 `CUDA_VISIBLE_DEVICES=""`를 추가합니다:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

또는 다중 GPU가 있는 경우 `pytest`에서 사용할 GPU를 지정할 수도 있습니다.
예를 들어, GPU `0` 및 `1`이 있는 경우 다음을 실행할 수 있습니다:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

이렇게 하면 다른 GPU에서 다른 작업을 실행하려는 경우 유용합니다.

일부 테스트는 반드시 CPU 전용으로 실행해야 하며, 일부는 CPU 또는 GPU 또는 TPU에서 실행해야 하고, 일부는 여러 GPU에서 실행해야 합니다.
다음 스킵 데코레이터는 테스트의 요구 사항을 CPU/GPU/TPU별로 설정하는 데 사용됩니다:

- `require_torch` - 이 테스트는 torch에서만 실행됩니다.
- `require_torch_gpu` - `require_torch`에 추가로 적어도 1개의 GPU가 필요합니다.
- `require_torch_multi_gpu` - `require_torch`에 추가로 적어도 2개의 GPU가 필요합니다.
- `require_torch_non_multi_gpu` - `require_torch`에 추가로 0개 또는 1개의 GPU가 필요합니다.
- `require_torch_up_to_2_gpus` - `require_torch`에 추가로 0개, 1개 또는 2개의 GPU가 필요합니다.
- `require_torch_xla` - `require_torch`에 추가로 적어도 1개의 TPU가 필요합니다.

GPU 요구 사항을 표로 정리하면 아래와 같습니디ㅏ:


| n gpus | decorator                      |
|--------+--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


예를 들어, 2개 이상의 GPU가 있고 pytorch가 설치되어 있을 때에만 실행되어야 하는 테스트는 다음과 같습니다:

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

`tensorflow`가 필요한 경우 `require_tf` 데코레이터를 사용합니다. 예를 들어 다음과 같습니다:

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

이러한 데코레이터는 중첩될 수 있습니다.
예를 들어, 느린 테스트로 진행되고 pytorch에서 적어도 하나의 GPU가 필요한 경우 다음과 같이 설정할 수 있습니다:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

`@parametrized`와 같은 일부 데코레이터는 테스트 이름을 다시 작성하기 때문에 `@require_*` 스킵 데코레이터는 올바르게 작동하려면 항상 맨 마지막에 나열되어야 합니다.
다음은 올바른 사용 예입니다:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

`@pytest.mark.parametrize`에는 이러한 순서 문제는 없으므로 처음 혹은 마지막에 위치시킬 수 있고 이러한 경우에도 잘 작동할 것입니다.
하지만 unittest가 아닌 경우에만 작동합니다.

테스트 내부에서 다음을 사용할 수 있습니다:

- 사용 가능한 GPU 수:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  #torch와 tf와 함께 작동
```

### 분산 훈련[[distributed-training]]

`pytest`는 분산 훈련을 직접적으로 다루지 못합니다.
이를 시도하면 하위 프로세스가 올바른 작업을 수행하지 않고 `pytest`라고 생각하기에 테스트 스위트를 반복해서 실행하게 됩니다.
그러나 일반 프로세스를 생성한 다음 여러 워커를 생성하고 IO 파이프를 관리하도록 하면 동작합니다.

다음은 사용 가능한 테스트입니다:

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

실행 지점으로 바로 이동하려면, 해당 테스트에서 `execute_subprocess_async` 호출을 검색하세요.

이러한 테스트를 실행하려면 적어도 2개의 GPU가 필요합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### 출력 캡처[[output-capture]]

테스트 실행 중 `stdout` 및 `stderr`로 전송된 모든 출력이 캡처됩니다.
테스트나 설정 메소드가 실패하면 캡처된 출력은 일반적으로 실패 추적 정보와 함께 표시됩니다.

출력 캡처를 비활성화하고 `stdout` 및 `stderr`를 정상적으로 받으려면 `-s` 또는 `--capture=no`를 사용하세요:

```bash
pytest -s tests/utils/test_logging.py
```

테스트 결과를 JUnit 형식의 출력으로 보내려면 다음을 사용하세요:

```bash
py.test tests --junitxml=result.xml
```

### 색상 조절[[color-control]]

색상이 없게 하려면 다음과 같이 설정하세요(예를 들어 흰색 배경에 노란색 글씨는 가독성이 좋지 않습니다):

```bash
pytest --color=no tests/utils/test_logging.py
```

### online pastebin service에 테스트 보고서 전송[[sending test report to online pastebin service]]

각 테스트 실패에 대한 URL을 만듭니다:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

이렇게 하면 각 실패에 대한 URL을 제공하는 remote Paste service에 테스트 실행 정보를 제출합니다.
일반적인 테스트를 선택할 수도 있고 혹은 특정 실패만 보내려면 `-x`와 같이 추가할 수도 있습니다.

전체 테스트 세션 로그에 대한 URL을 생성합니다:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## 테스트 작성[[writing-tests]]

🤗 transformers 테스트는 대부분 `unittest`를 기반으로 하지만,
`pytest`에서 실행되므로 대부분의 경우 두 시스템의 기능을 사용할 수 있습니다.

지원되는 기능에 대해 [여기](https://docs.pytest.org/en/stable/unittest.html)에서 확인할 수 있지만,
기억해야 할 중요한 점은 대부분의 `pytest` fixture가 작동하지 않는다는 것입니다.
파라미터화도 작동하지 않지만, 우리는 비슷한 방식으로 작동하는 `parameterized` 모듈을 사용합니다.


### 매개변수화[[parametrization]]

동일한 테스트를 다른 인수로 여러 번 실행해야 하는 경우가 종종 있습니다.
테스트 내에서 이 작업을 수행할 수 있지만, 그렇게 하면 하나의 인수 세트에 대해 테스트를 실행할 수 없습니다.

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

이제 기본적으로 이 테스트는 `test_floor`의 마지막 3개 인수가
매개변수 목록의 해당 인수에 할당되는 것으로 3번 실행될 것입니다.

그리고 `negative` 및 `integer` 매개변수 집합만 실행하려면 다음과 같이 실행할 수 있습니다:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

또는 `negative` 하위 테스트를 제외한 모든 서브 테스트를 다음과 같이 실행할 수 있습니다:

```bash
pytest -k "not negative" tests/test_mytest.py
```

앞에서 언급한 `-k` 필터를 사용하는 것 외에도,
각 서브 테스트의 정확한 이름을 확인한 후에 일부 혹은 전체 서브 테스트를 실행할 수 있습니다.

```bash
pytest test_this1.py --collect-only -q
```

그리고 다음의 내용을 확인할 수 있을 것입니다:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

2개의 특정한 서브 테스트만 실행할 수도 있습니다:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

`transformers`의 개발자 종속성에 이미 있는 [parameterized](https://pypi.org/project/parameterized/) 모듈은
`unittests`와 `pytest` 테스트 모두에서 작동합니다.

그러나 테스트가 `unittest`가 아닌 경우 `pytest.mark.parametrize`를 사용할 수 있습니다(이미 있는 일부 테스트에서 사용되는 경우도 있습니다.
주로 `examples` 하위에 있습니다).

다음은 `pytest`의 `parametrize` 마커를 사용한 동일한 예입니다:

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

`parameterized`와 마찬가지로 `pytest.mark.parametrize`를 사용하면
`-k` 필터가 작동하지 않는 경우에도 실행할 서브 테스트를 정확하게 지정할 수 있습니다.
단, 이 매개변수화 함수는 서브 테스트의 이름 집합을 약간 다르게 생성합니다. 다음과 같은 모습입니다:

```bash
pytest test_this2.py --collect-only -q
```

그리고 다음의 내용을 확인할 수 있을 것입니다:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

특정한 테스트에 대해서만 실행할 수도 있습니다:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

이전의 예시와 같이 실행할 수 있습니다.



### 파일 및 디렉터리[[files-and-directories]]

테스트에서 종종 현재 테스트 파일과 관련된 상대적인 위치를 알아야 하는 경우가 있습니다.
테스트가 여러 디렉터리에서 호출되거나 깊이가 다른 하위 디렉터리에 있을 수 있기 때문에 그 위치를 아는 것은 간단하지 않습니다.
`transformers.test_utils.TestCasePlus`라는 헬퍼 클래스는 모든 기본 경로를 처리하고 간단한 액세서를 제공하여 이 문제를 해결합니다:


- `pathlib` 객체(완전히 정해진 경로)

  - `test_file_path` - 현재 테스트 파일 경로 (예: `__file__`)
  - test_file_dir` - 현재 테스트 파일이 포함된 디렉터리
  - tests_dir` - `tests` 테스트 스위트의 디렉터리
  - examples_dir` - `examples` 테스트 스위트의 디렉터리
  - repo_root_dir` - 저장소 디렉터리
  - src_dir` - `src`의 디렉터리(예: `transformers` 하위 디렉터리가 있는 곳)

- 문자열로 변환된 경로---위와 동일하지만, `pathlib` 객체가 아닌 문자열로 경로를 반환합니다:

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

위의 내용을 사용하려면 테스트가 'transformers.test_utils.TestCasePlus'의 서브클래스에 있는지 확인해야 합니다.
예를 들어 다음과 같습니다:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

만약 `pathlib`를 통해 경로를 조작할 필요가 없거나 경로를 문자열로만 필요로 하는 경우에는 `pathlib` 객체에 `str()`을 호출하거나 `_str`로 끝나는 접근자를 사용할 수 있습니다.
예를 들어 다음과 같습니다:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### 임시 파일 및 디렉터리[[temporary-files-and-directories]]

고유한 임시 파일 및 디렉터리를 사용하는 것은 병렬 테스트 실행에 있어 필수적입니다.
이렇게 함으로써 테스트들이 서로의 데이터를 덮어쓰지 않게 할 수 있습니다. 또한 우리는 생성된 테스트의 종료 단계에서 이러한 임시 파일 및 디렉터리를 제거하고 싶습니다.
따라서 이러한 요구 사항을 충족시켜주는 `tempfile`과 같은 패키지를 사용하는 것이 중요합니다.

그러나 테스트를 디버깅할 때는 임시 파일이나 디렉터리에 들어가는 내용을 확인할 수 있어야 하며,
재실행되는 각 테스트마다 임시 파일이나 디렉터리의 경로에 대해 무작위 값이 아닌 정확한 값을 알고 싶을 것입니다.

`transformers.test_utils.TestCasePlus`라는 도우미 클래스는 이러한 목적에 가장 적합합니다.
이 클래스는 `unittest.TestCase`의 하위 클래스이므로, 우리는 이것을 테스트 모듈에서 쉽게 상속할 수 있습니다.

다음은 해당 클래스를 사용하는 예시입니다:

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

이 코드는 고유한 임시 디렉터리를 생성하고 `tmp_dir`을 해당 위치로 설정합니다.

- 고유한 임시 디렉터리를 생성합니다:

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir`에는 생성된 임시 디렉터리의 경로가 포함됩니다.
이는 테스트의 종료 단계에서 자동으로 제거됩니다.

- 선택한 경로로 임시 디렉터리 생성 후에 테스트 시작 전에 비어 있는 상태인지 확인하고, 테스트 후에는 비우지 마세요.

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

이것은 디버깅할 때 특정 디렉터리를 모니터링하고,
그 디렉터리에 이전에 실행된 테스트가 데이터를 남기지 않도록 하는 데에 유용합니다.

- `before` 및 `after` 인수를 직접 오버라이딩하여 기본 동작을 변경할 수 있으며
다음 중 하나의 동작으로 이어집니다:

  - `before=True`: 테스트 시작 시 임시 디렉터리가 항상 지워집니다.
  - `before=False`: 임시 디렉터리가 이미 존재하는 경우 기존 파일은 그대로 남습니다.
  - `after=True`: 테스트 종료 시 임시 디렉터리가 항상 삭제됩니다.
  - `after=False`: 테스트 종료 시 임시 디렉터리가 항상 그대로 유지됩니다.

<Tip>

`rm -r`에 해당하는 명령을 안전하게 실행하기 위해,
명시적인 `tmp_dir`을 사용하는 경우 프로젝트 저장소 체크 아웃의 하위 디렉터리만 허용됩니다.
따라서 실수로 `/tmp`가 아닌 중요한 파일 시스템의 일부가 삭제되지 않도록 항상 `./`로 시작하는 경로를 전달해야 합니다.

</Tip>

<Tip>

각 테스트는 여러 개의 임시 디렉터리를 등록할 수 있으며,
별도로 요청하지 않는 한 모두 자동으로 제거됩니다.

</Tip>

### 임시 sys.path 오버라이드[[temporary-sys.path-override]]

`sys.path`를 다른 테스트로 임시로 오버라이드하기 위해 예를 들어 `ExtendSysPath` 컨텍스트 관리자를 사용할 수 있습니다.
예를 들어 다음과 같습니다:


```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 테스트 건너뛰기[[skipping-tests]]

이것은 버그가 발견되어 새로운 테스트가 작성되었지만 아직 그 버그가 수정되지 않은 경우에 유용합니다.
이 테스트를 주 저장소에 커밋하려면 `make test` 중에 건너뛰도록 해야 합니다.

방법:

- **skip**은 테스트가 일부 조건이 충족될 경우에만 통과될 것으로 예상되고, 그렇지 않으면 pytest가 전체 테스트를 건너뛰어야 함을 의미합니다.
일반적인 예로는 Windows가 아닌 플랫폼에서 Windows 전용 테스트를 건너뛰거나
외부 리소스(예를 들어 데이터베이스)에 의존하는 테스트를 건너뛰는 것이 있습니다.

- **xfail**은 테스트가 특정한 이유로 인해 실패할 것으로 예상하는 것을 의미합니다.
일반적인 예로는 아직 구현되지 않은 기능이나 아직 수정되지 않은 버그의 테스트가 있습니다.
`xfail`로 표시된 테스트가 예상대로 실패하지 않고 통과된 경우, 이것은 xpass이며 테스트 결과 요약에 기록됩니다.

두 가지 중요한 차이점 중 하나는 `skip`은 테스트를 실행하지 않지만 `xfail`은 실행한다는 것입니다.
따라서 오류가 있는 코드가 일부 테스트에 영향을 미칠 수 있는 경우 `xfail`을 사용하지 마세요.

#### 구현[[implementation]]

- 전체 테스트를 무조건 건너뛰려면 다음과 같이 할 수 있습니다:

```python no-style
@unittest.skip(reason="this bug needs to be fixed")
def test_feature_x():
```

또는 pytest를 통해:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

또는 `xfail` 방식으로:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

- 테스트 내부에서 내부 확인에 따라 테스트를 건너뛰는 방법은 다음과 같습니다:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

또는 모듈 전체:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

또는 `xfail` 방식으로:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- import가 missing된 모듈이 있을 때 그 모듈의 모든 테스트를 건너뛰는 방법:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 조건에 따라 테스트를 건너뛰는 방법:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

또는:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

또는 모듈 전체를 건너뛰는 방법:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

보다 자세한 예제 및 방법은 [여기](https://docs.pytest.org/en/latest/skipping.html)에서 확인할 수 있습니다.

### 느린 테스트[[slow-tests]]

테스트 라이브러리는 지속적으로 확장되고 있으며, 일부 테스트는 실행하는 데 몇 분이 걸립니다.
그리고 우리에게는 테스트 스위트가 CI를 통해 완료되기까지 한 시간을 기다릴 여유가 없습니다.
따라서 필수 테스트를 위한 일부 예외를 제외하고 느린 테스트는 다음과 같이 표시해야 합니다.

```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

`@slow`로 표시된 테스트를 실행하려면 `RUN_SLOW=1` 환경 변수를 설정하세요. 예를 들어 다음과 같습니다:

```bash
RUN_SLOW=1 pytest tests
```

`@parameterized`와 같은 몇 가지 데코레이터는 테스트 이름을 다시 작성합니다.
그러므로 `@slow`와 나머지 건너뛰기 데코레이터 `@require_*`가 올바르게 작동되려면 마지막에 나열되어야 합니다. 다음은 올바른 사용 예입니다.

```python no-style
@parameterized.expand(...)
@slow
def test_integration_foo():
```

이 문서의 초반부에 설명된 것처럼 느린 테스트는 PR의 CI 확인이 아닌 예약된 일정 기반으로 실행됩니다.
따라서 PR 제출 중에 일부 문제를 놓친 채로 병합될 수 있습니다.
이러한 문제들은 다음번의 예정된 CI 작업 중에 감지됩니다.
하지만 PR을 제출하기 전에 자신의 컴퓨터에서 느린 테스트를 실행하는 것 또한 중요합니다.

느린 테스트로 표시해야 하는지 여부를 결정하는 대략적인 결정 기준은 다음과 같습니다.

만약 테스트가 라이브러리의 내부 구성 요소 중 하나에 집중되어 있다면(예: 모델링 파일, 토큰화 파일, 파이프라인),
해당 테스트를 느린 테스트 스위트에서 실행해야 합니다.
만약 라이브러리의 다른 측면(예: 문서 또는 예제)에 집중되어 있다면,
해당 테스트를 느린 테스트 스위트에서 실행해야 합니다. 그리고 이 접근 방식을 보완하기 위해 예외를 만들어야 합니다.

- 무거운 가중치 세트나 50MB보다 큰 데이터셋을 다운로드해야 하는 모든 테스트(예: 모델 통합 테스트, 토크나이저 통합 테스트, 파이프라인 통합 테스트)를
  느린 테스트로 설정해야 합니다.
  새로운 모델을 추가하는 경우 통합 테스트용으로 무작위 가중치로 작은 버전을 만들어 허브에 업로드해야 합니다.
  이 내용은 아래 단락에서 설명됩니다.
- 특별히 빠르게 실행되도록 최적화되지 않은 학습을 수행해야 하는 테스트는 느린 테스트로 설정해야 합니다.
- 느리지 않아야 할 테스트 중 일부가 극도로 느린 경우
  예외를 도입하고 이를 `@slow`로 설정할 수 있습니다.
  대용량 파일을 디스크에 저장하고 불러오는 자동 모델링 테스트는 `@slow`으로 표시된 테스트의 좋은 예입니다.
- CI에서 1초 이내에 테스트가 완료되는 경우(다운로드 포함)에는 느린 테스트가 아니어야 합니다.

느린 테스트가 아닌 경우에는 다양한 내부를 완전히 커버하면서 빠르게 유지되어야 합니다.
예를 들어, 무작위 가중치를 사용하여 특별히 생성된 작은 모델로 테스트하면 상당한 커버리지를 얻을 수 있습니다.
이러한 모델은 최소한의 레이어 수(예: 2), 어휘 크기(예: 1000) 등의 요소만 가집니다. 그런 다음 `@slow` 테스트는 대형 느린 모델을 사용하여 정성적인 테스트를 수행할 수 있습니다.
이러한 작은 모델을 사용하는 방법을 확인하려면 다음과 같이 *tiny* 모델을 찾아보세요.

```bash
grep tiny tests examples
```

다음은 작은 모델[stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de)을 만든
[script](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py) 예시입니다.
특정 모델의 아키텍처에 맞게 쉽게 조정할 수 있습니다.

예를 들어 대용량 모델을 다운로드하는 경우 런타임을 잘못 측정하기 쉽지만,
로컬에서 테스트하면 다운로드한 파일이 캐시되어 다운로드 시간이 측정되지 않습니다.
대신 CI 로그의 실행 속도 보고서를 확인하세요(`pytest --durations=0 tests`의 출력).

이 보고서는 느린 이상값으로 표시되지 않거나 빠르게 다시 작성해야 하는 느린 이상값을 찾는 데도 유용합니다.
CI에서 테스트 스위트가 느려지기 시작하면 이 보고서의 맨 위 목록에 가장 느린 테스트가 표시됩니다.



### stdout/stderr 출력 테스트[[testing-the-stdout/stderr-output]]

`stdout` 및/또는 `stderr`로 쓰는 함수를 테스트하려면 `pytest`의 [capsys 시스템](https://docs.pytest.org/en/latest/capture.html)을 사용하여 해당 스트림에 액세스할 수 있습니다.
다음과 같이 수행할 수 있습니다.

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
    out, err = capsys.readouterr()  # 캡처된 출력 스트림 사용
    # 선택 사항: 캡처된 스트림 재생성
    sys.stdout.write(out)
    sys.stderr.write(err)
    # 테스트:
    assert msg in out
    assert msg in err
```

그리고, 물론 대부분의 경우에는 `stderr`는 예외의 일부로 제공됩니다.
그러므로 해당 경우에는 try/except를 사용해야 합니다.

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

`stdout`를 캡처하는 또 다른 방법은 `contextlib.redirect_stdout`를 사용하는 것입니다.

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
    # 선택 사항: 캡처된 스트림 재생성
    sys.stdout.write(out)
    # 테스트:
    assert msg in out
```

`stdout` 캡처에 관련된 중요한 문제 중 하나는 보통 `print`에서 이전에 인쇄된 내용을 재설정하는 `\r` 문자가 포함될 수 있다는 것입니다.
`pytest`에서는 문제가 없지만 `pytest -s`에서는 이러한 문자가 버퍼에 포함되므로
`-s`가 있거나 없는 상태에서 태스트를 수행할 수 있으려면 캡처된 출력에 대해 추가적인 정리가 필요합니다.
이 경우에는 `re.sub(r'~.*\r', '', buf, 0, re.M)`을 사용할 수 있습니다.

하지만 도우미 컨텍스트 관리자 래퍼를 사용하면
출력에 `\r`이 포함되어 있는지의 여부에 관계없이 모든 것을 자동으로 처리하므로 편리합니다.

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

다음은 전체 테스트 예제입니다.

```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

`stderr`를 캡처하고 싶다면, 대신 `CaptureStderr` 클래스를 사용하세요.

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

두 스트림을 동시에 캡처해야 한다면, 부모 `CaptureStd` 클래스를 사용하세요.

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

또한, 테스트의 디버깅을 지원하기 위해
이러한 컨텍스트 관리자는 기본적으로 컨텍스트에서 종료할 때 캡처된 스트림을 자동으로 다시 실행합니다.


### 로거 스트림 캡처[[capturing-logger-stream]]

로거 출력을 검증해야 하는 경우 `CaptureLogger`를 사용할 수 있습니다.

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

### 환경 변수를 이용하여 테스트[[testing-with-environment-variables]]

특정 테스트의 환경 변수 영향을 검증하려면
`transformers.testing_utils.mockenv`라는 도우미 데코레이터를 사용할 수 있습니다.

```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

일부 경우에는 외부 프로그램을 호출해야할 수도 있는데, 이 때에는 여러 개의 로컬 경로를 포함하는 `os.environ`에서 `PYTHONPATH`의 설정이 필요합니다.
헬퍼 클래스 `transformers.test_utils.TestCasePlus`가 도움이 됩니다:

```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # 이제 `env`를 사용하여 외부 프로그램 호출
```

테스트 파일이 `tests` 테스트 스위트 또는 `examples`에 있는지에 따라
`env[PYTHONPATH]`가 두 디렉터리 중 하나를 포함하도록 설정되며,
현재 저장소에 대해 테스트가 수행되도록 `src` 디렉터리도 포함됩니다.
테스트 호출 이전에 설정된 경우에는 `env[PYTHONPATH]`를 그대로 사용합니다.

이 헬퍼 메소드는 `os.environ` 객체의 사본을 생성하므로 원본은 그대로 유지됩니다.


### 재현 가능한 결과 얻기[[getting-reproducible-results]]

일부 상황에서 테스트에서 임의성을 제거하여 동일하게 재현 가능한 결과를 얻고 싶을 수 있습니다.
이를 위해서는 다음과 같이 시드를 고정해야 합니다.

```python
seed = 42

# 파이썬 RNG
import random

random.seed(seed)

# 파이토치 RNG
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 넘파이 RNG
import numpy as np

np.random.seed(seed)

# 텐서플로 RNG
tf.random.set_seed(seed)
```

### 테스트 디버깅[[debugging tests]]

경고가 있는 곳에서 디버거를 시작하려면 다음을 수행하세요.

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## Github Actions 워크플로우 작업 처리[[working-with-github-actions-workflows]]

셀프 푸시 워크플로우 CI 작업을 트리거하려면, 다음을 수행해야 합니다.

1. `transformers` 원본에서 새 브랜치를 만듭니다(포크가 아닙니다!).
2. 브랜치 이름은 `ci_` 또는 `ci-`로 시작해야 합니다(`main`도 트리거하지만 `main`에서는 PR을 할 수 없습니다).
   또한 특정 경로에 대해서만 트리거되므로 이 문서가 작성된 후에 변경된 내용은
   [여기](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml)의 *push:*에서 확인할 수 있습니다.
3. 이 브랜치에서 PR을 생성합니다
4. 그런 다음 [여기](https://github.com/huggingface/transformers/actions/workflows/self-push.yml)에서 작업이 나타나는지 확인할 수 있습니다.
   백로그가 있는 경우, 바로 실행되지 않을 수도 있습니다.




## 실험적인 CI 기능 테스트[[testing-Experimental-CI-Features]]

CI 기능을 테스트하는 것은 일반 CI 작동에 방해가 될 수 있기 때문에 잠재적으로 문제가 발생할 수 있습니다.
따라서 새로운 CI 기능을 추가하는 경우 다음과 같이 수행해야 합니다.

1. 테스트해야 할 내용을 테스트하는 새로운 전용 작업을 생성합니다.
2. 새로운 작업은 항상 성공해야만 녹색 ✓를 받을 수 있습니다(아래에 자세한 내용이 있습니다).
3. 다양한 PR 유형에 대한 확인을  위해
   (사용자 포크 브랜치, 포크되지 않은 브랜치, github.com UI 직접 파일 편집에서 생성된 브랜치, 강제 푸시 등 PR의 유형은 아주 다양합니다.)
   며칠 동안 실험 작업의 로그를 모니터링하면서 실행해봅니다.
   (의도적으로 항상 녹색을 표시하므로 작업 전체가 녹색은 아니라는 점에 유의합니다.)
4. 모든 것이 안정적인지 확인한 후, 새로운 변경 사항을 기존 작업에 병합합니다.

이렇게 하면 CI 기능 자체에 대한 실험이 일반 작업 흐름에 방해가 되지 않습니다.

그러나 새로운 CI 기능이 개발 중인 동안, 항상 성공하도록 할 수 있는 방법은 무엇일까요?

TravisCI와 같은 일부 CI는 `ignore-step-failure`를 지원하며 전체 작업을 성공한 것으로 보고하지만,
현재 우리가 사용하는 CircleCI와 Github Actions는 이를 지원하지 않습니다.

따라서 다음과 같은 해결책을 사용할 수 있습니다.

1. bash 스크립트에서 가능한 많은 오류를 억제하기 위해 실행 명령의 시작 부분에 `set +euo pipefail`을 추가합니다.
2. 마지막 명령은 반드시 성공해야 합니다. `echo "done"` 또는 `true`를 사용하면 됩니다.

예시는 다음과 같습니다.

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

간단한 명령의 경우 다음과 같이 수행할 수도 있습니다.

```bash
cmd_that_may_fail || true
```

결과에 만족한 후에는 물론, 실험적인 단계 또는 작업을 일반 작업의 나머지 부분과 통합하면서
`set +euo pipefail` 또는 기타 추가한 요소를 제거하여
실험 작업이 일반 CI 작동에 방해되지 않도록 해야 합니다.

이 전반적인 과정은 실험 단계가 PR의 전반적인 상태에 영향을 주지 않고 실패하도록
`allow-failure`와 같은 기능을 설정할 수 있다면 훨씬 더 쉬웠을 것입니다.
그러나 앞에서 언급한 바와 같이 CircleCI와 Github Actions는 현재 이러한 기능들 지원하지 않습니다.

이 기능의 지원을 위한 투표에 참여하고 CI 관련 스레드들에서 이러한 상황을 확인할 수도 있습니다.

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)
