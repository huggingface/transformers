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

# 로깅 [[logging]]

🤗 트랜스포머는 중앙 집중식 로깅 시스템을 제공하여 라이브러리의 출력 레벨을 쉽게 설정할 수 있습니다.

현재 라이브러리의 기본 출력 레벨은 `WARNING`으로 설정되어 있습니다.

출력 레벨을 변경하려면 직접적인 설정 메서드를 사용할 수 있습니다. 예를 들어, 출력 레벨을 INFO 수준으로 변경하는 방법은 다음과 같습니다.

```python
import transformers

transformers.logging.set_verbosity_info()
```

환경 변수 `TRANSFORMERS_VERBOSITY`를 사용하여 기본 출력 레벨을 재정의할 수도 있습니다. 이를 `debug`, `info`, `warning`, `error`, `critical`, `fatal` 중 하나로 설정할 수 있습니다. 예를 들어 다음과 같습니다.

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

또한, 일부 `warnings`는 환경 변수 `TRANSFORMERS_NO_ADVISORY_WARNINGS`를 1과 같은 true 값으로 설정하여 비활성화할 수 있습니다. 이렇게 하면 [`logger.warning_advice`]를 사용하여 기록된 경고가 비활성화됩니다. 예를 들어 다음과 같습니다.

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

다음은 라이브러리와 동일한 로거를 자신의 모듈이나 스크립트에서 사용하는 방법에 대한 예시입니다.

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```


이 로깅 모듈의 모든 메서드는 아래에 문서화되어 있으며, 주요 메서드는 현재 로거의 출력 수준을 가져오는 [`logging.get_verbosity`]와 원하는 출력 수준으로 설정하는 [`logging.set_verbosity`] 입니다. 출력 수준은 (가장 적은 출력에서 가장 많은 출력 순으로) 다음과 같으며, 해당 수준에 대응하는 정수 값은 괄호 안에 표시됩니다.

- `transformers.logging.CRITICAL` 또는 `transformers.logging.FATAL` (정숫값, 50): 가장 심각한 오류만 보고합니다.
- `transformers.logging.ERROR` (정숫값, 40): 오류만 보고합니다.
- `transformers.logging.WARNING` 또는 `transformers.logging.WARN` (정숫값, 30): 오류와 경고만 보고합니다. 이는 라이브러리에서 기본으로 사용되는 수준입니다.
- `transformers.logging.INFO` (정숫값, 20): 오류, 경고, 그리고 기본적인 정보를 보고합니다.
- `transformers.logging.DEBUG` (정숫값, 10): 모든 정보를 보고합니다.

기본적으로 모델 다운로드 중에는 `tqdm` 진행 표시줄이 표시됩니다. [`logging.disable_progress_bar`]와 [`logging.enable_progress_bar`]를 사용하여 이 동작을 숨기거나 다시 표시할 수 있습니다.

## `logging` vs `warnings`[[transformers.utils.logging.captureWarnings]]

Python에는 종종 함께 사용되는 두 가지 로깅 시스템이 있습니다. 위에서 설명한 `logging`과 `warnings`입니다. `warnings`는 특정 범주로 경고를 세분화할 수 있습니다. 예를 들어, 이미 더 이상 사용되지 않는 기능이나 경로에 대해 `FutureWarning`이 사용되고, 곧 사용 중단될 기능을 알리기 위해 `DeprecationWarning`이 사용됩니다.

트랜스포머 라이브러리에서는 두 시스템 모두를 사용합니다. `logging`의 `captureWarnings` 메서드를 활용하고 이를 조정하여 위에서 설명한 출력 수준 설정자들을 통해 이러한 경고 메시지들을 관리할 수 있도록 합니다.

라이브러리 개발자는 다음과 같은 지침을 따르는 것이 좋습니다.

- `warnings`는 라이브러리 개발자와 `transformers`에 의존하는 라이브러리 개발자들에게 유리합니다.
- `logging`은 일반적인 프로젝트 라이브러리 개발자보다는, 라이브러리를 사용하는 최종 사용자들에게 유리할 것입니다.

아래에서 `captureWarnings` 메소드에 대한 참고 사항을 확인할 수 있습니다.

[[autodoc]] logging.captureWarnings

## 기본 설정자 [[transformers.utils.logging.set_verbosity_error]]

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## 기타 함수 [[transformers.utils.logging.get_verbosity]]

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar
