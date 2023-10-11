<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

# Pull Request에 대한 검사 [[checks-on-a-pull-request]]

🤗 Transformers에서 Pull Request를 열 때, 기존에 있는 것을 망가뜨리지 않는지 확인하기 위해 상당한 수의 검사가 실행됩니다. 이러한 검사는 다음과 같은 네 가지 유형으로 구성됩니다:
- 일반적인 테스트
- 문서 빌드
- 코드 및 문서 스타일
- 일반 저장소 일관성

이 문서에서는 이러한 다양한 검사와 그 이유를 설명하고, PR에서 하나 이상의 검사가 실패한 경우 로컬에서 어떻게 디버그하는지 알아보겠습니다.

참고로, 이러한 검사를 사용하려면 개발 설치가 필요합니다:

```bash
pip install transformers[dev]
```

또는 Transformers 저장소 내에 편집 가능한 설치가 필요합니다:

```bash
pip install -e .[dev]
```

Transformers의 선택적 종속성 수가 많이 늘어났기 때문에 개발 설치를 실패할 수도 있습니다. 개발 설치가 실패하는 경우, 작업 중인 Deep Learning 프레임워크 (PyTorch, TensorFlow 및/또는 Flax)를 설치하고 다음 명령을 실행하세요.

```bash
pip install transformers[quality]
```

편집 가능한 설치의 경우는 다음 명령을 실행하세요.

```bash
pip install -e .[quality]
```


## 테스트 [[tests]]

`ci/circleci: run_tests_`로 시작하는 모든 작업은 Transformers 테스트 모음의 일부를 실행합니다. 이러한 작업은 특정 환경에서 일부 라이브러리에 중점을 둡니다. 예를 들어 `ci/circleci: run_tests_pipelines_tf`는 TensorFlow만 설치된 환경에서 파이프라인 테스트를 실행합니다.

테스트 모듈에서 실제로 변경 사항이 없을 때 테스트를 실행하지 않기 위해, 테스트 모음의 일부만 실행됩니다. 라이브러리의 변경 전후에 대한 차이를 확인하기 위해 유틸리티가 실행되고, 해당 차이에 영향을 받는 테스트가 선택됩니다. 이 유틸리티는 로컬에서 다음과 같이 실행할 수 있습니다:

```bash
python utils/tests_fetcher.py
```

Transformers 저장소의 최상단에서 실행합니다. 이 유틸리티는 다음과 같은 작업을 수행합니다:

1. 변경 사항이 있는 파일마다 변경 사항이 코드인지 주석 또는 문서 문자열인지 확인합니다. 실제 코드 변경이 있는 파일만 유지됩니다.
2. 소스 코드 파일의 각 파일에 대해 재귀적으로 영향을 주는 모든 파일을 제공하는 내부 맵을 작성합니다. 모듈 B가 모듈 A를 가져오면 모듈 A는 모듈 B에 영향을 줍니다. 재귀적인 영향에는 각 모듈이 이전 모듈을 가져오는 모듈 체인이 필요합니다.
3. 단계 1에서 수집한 파일에 이 맵을 적용하여 PR에 영향을 받는 모델 파일 목록을 얻습니다.
4. 각 파일을 해당하는 테스트 파일에 매핑하고 실행할 테스트 목록을 가져옵니다.

로컬에서 스크립트를 실행하면 단계 1, 3 및 4의 결과를 출력하여 실행되는 테스트를 알 수 있습니다. 스크립트는 또한 `test_list.txt`라는 파일을 생성하여 실행할 테스트 목록을 포함하며, 다음 명령으로 해당 테스트를 로컬에서 실행할 수 있습니다:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

잘못된 사항이 누락되었을 경우, 전체 테스트 모음도 매일 실행됩니다.

## 문서 빌드 [[documentation-build]]

`build_pr_documentation` 작업은 문서를 빌드하고 미리 보기를 생성하여 PR이 병합된 후 모든 것이 제대로 보이는지 확인합니다. 로봇은 PR에 문서 미리보기 링크를 추가합니다. PR에서 만든 변경 사항은 자동으로 미리보기에 업데이트됩니다. 문서 빌드에 실패한 경우 **세부 정보**를 클릭하여 어디에서 문제가 발생했는지 확인할 수 있습니다. 오류는 주로 `toctree`에 누락된 파일과 같이 간단한 오류입니다.

로컬에서 문서를 빌드하거나 미리 볼 경우, docs 폴더의 [`README.md`](https://github.com/huggingface/transformers/tree/main/docs)를 참조하세요.

## 코드 및 문서 스타일 [[code-and-documentation-style]]

`black`과 `ruff`를 사용하여 모든 소스 파일, 예제 및 테스트에 코드 형식을 적용합니다. 또한, `utils/style_doc.py`에서 문서 문자열과 `rst` 파일의 형식, 그리고 Transformers의 `__init__.py` 파일에서 실행되는 지연된 임포트의 순서에 대한 사용자 정의 도구가 있습니다. 이 모든 것은 다음을 실행함으로써 실행할 수 있습니다:

```bash
make style
```

CI는 이러한 사항이 `ci/circleci: check_code_quality` 검사 내에서 적용되었는지 확인합니다. 또한 `ruff`도 실행되며, 정의되지 않은 변수나 사용되지 않은 변수를 발견하면 경고합니다. 이 검사를 로컬에서 실행하려면 다음을 사용하세요:

```bash
make quality
```

이 작업은 많은 시간이 소요될 수 있으므로 현재 브랜치에서 수정한 파일에 대해서만 동일한 작업을 실행하려면 다음을 실행하세요.

```bash
make fixup
```

이 명령은 현재 브랜치에서 수정한 파일에 대한 모든 추가적인 검사도 실행합니다. 이제 이들을 살펴보겠습니다.

## 저장소 일관성 [[repository-consistency]]

이는 PR이 저장소를 정상적인 상태로 유지하는지 확인하는 모든 테스트를 모은 것이며, `ci/circleci: check_repository_consistency` 검사에서 수행됩니다. 다음을 실행함으로써 로컬에서 이 검사를 실행할 수 있습니다.

```bash
make repo-consistency
```

이 검사는 다음을 확인합니다.

- init에 추가된 모든 객체가 문서화되었는지 (`utils/check_repo.py`에서 수행)
- `__init__.py` 파일의 두 섹션에 동일한 내용이 있는지 (`utils/check_inits.py`에서 수행)
- 다른 모듈에서 복사된 코드가 원본과 일치하는지 (`utils/check_copies.py`에서 수행)
- 모든 구성 클래스에 docstring에 언급된 유효한 체크포인트가 적어도 하나 있는지 (`utils/check_config_docstrings.py`에서 수행)
- 모든 구성 클래스가 해당하는 모델링 파일에서 사용되는 속성만 포함하고 있는지 (`utils/check_config_attributes.py`에서 수행)
- README와 문서 인덱스의 번역이 메인 README와 동일한 모델 목록을 가지고 있는지 (`utils/check_copies.py`에서 수행)
- 문서의 자동 생성된 테이블이 최신 상태인지 (`utils/check_table.py`에서 수행)
- 라이브러리에는 선택적 종속성이 설치되지 않았더라도 모든 객체가 사용 가능한지 (`utils/check_dummies.py`에서 수행)

이러한 검사가 실패하는 경우, 처음 두 가지 항목은 수동으로 수정해야 하며, 나머지 네 가지 항목은 다음 명령을 실행하여 자동으로 수정할 수 있습니다.

```bash
make fix-copies
```

추가적인 검사는 새로운 모델을 추가하는 PR에 대한 것으로, 주로 다음과 같습니다:

- 추가된 모든 모델이 Auto-mapping에 있는지 (`utils/check_repo.py`에서 수행)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- 모든 모델이 올바르게 테스트되었는지 (`utils/check_repo.py`에서 수행)

<!-- TODO Sylvain, add the following
- 모든 모델이 메인 README, 주요 문서에 추가되었는지
- 사용된 모든 체크포인트가 실제로 Hub에 존재하는지

-->

### 복사본 확인 [[check-copies]]

Transformers 라이브러리는 모델 코드에 대해 매우 완고하며, 각 모델은 다른 모델에 의존하지 않고 완전히 단일 파일로 구현되어야 합니다. 이렇게 하기 위해 특정 모델의 코드 복사본이 원본과 일관된 상태로 유지되는지 확인하는 메커니즘을 추가했습니다. 따라서 버그 수정이 필요한 경우 다른 모델에 영향을 주는 모든 모델을 볼 수 있으며 수정을 적용할지 수정된 사본을 삭제할지 선택할 수 있습니다.

<Tip>

파일이 다른 파일의 완전한 사본인 경우 해당 파일을 `utils/check_copies.py`의 `FULL_COPIES` 상수에 등록해야 합니다.

</Tip>

이 메커니즘은 `# Copied from xxx` 형식의 주석을 기반으로 합니다. `xxx`에는 아래에 복사되는 클래스 또는 함수의 전체 경로가 포함되어야 합니다. 예를 들어 `RobertaSelfOutput`은 `BertSelfOutput` 클래스의 복사본입니다. 따라서 [여기](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L289)에서 주석이 있습니다:


```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
```

클래스 전체에 수정을 적용하는 대신에 복사본과 관련있는 메서드에 적용할 수도 있습니다. 예를 들어 [여기](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L598)에서 `RobertaPreTrainedModel._init_weights`가 `BertPreTrainedModel`의 동일한 메서드에서 복사된 것을 볼 수 있으며 해당 주석이 있습니다:

```py
# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
```

복사본이 이름만 다른 경우가 있습니다: 예를 들어 `RobertaAttention`에서 `BertSelfAttention` 대신 `RobertaSelfAttention`을 사용하지만 그 외에는 코드가 완전히 동일합니다: 이 때 `# Copied from`은 `Copied from xxx with foo->bar`와 같은 간단한 문자열 대체를 지원합니다. 이는 모든 `foo` 인스턴스를 `bar`로 바꿔서 코드를 복사합니다. [여기](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L304C1-L304C86)에서 어떻게 사용되는지 볼 수 있습니다:

```py
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
```

화살표 주변에는 공백이 없어야 합니다(공백이 대체 패턴의 일부인 경우는 예외입니다).

대체 패턴을 쉼표로 구분하여 여러 패턴을 추가할 수 있습니다. 예를 들어 `CamemberForMaskedLM`은 두 가지 대체 사항을 가진 `RobertaForMaskedLM`의 복사본입니다: `Roberta`를 `Camembert`로 대체하고 `ROBERTA`를 `CAMEMBERT`로 대체합니다. [여기](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/camembert/modeling_camembert.py#L929)에서 이것이 주석으로 어떻게 구현되었는지 확인할 수 있습니다:

```py
# Copied from transformers.models.roberta.modeling_roberta.RobertaForMaskedLM with Roberta->Camembert, ROBERTA->CAMEMBERT
```

순서가 중요한 경우(이전 수정과 충돌할 수 있는 경우) 수정은 왼쪽에서 오른쪽으로 실행됩니다.

<Tip>

새 변경이 서식을 변경하는 경우(짧은 이름을 매우 긴 이름으로 바꾸는 경우) 자동 서식 지정기를 적용한 후 복사본이 검사됩니다.

</Tip>

패턴의 대소문자가 다른 경우(대문자와 소문자가 혼용된 대체 양식) `all-casing` 옵션을 추가하는 방법도 있습니다. [여기](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/mobilebert/modeling_mobilebert.py#L1237)에서 `MobileBertForSequenceClassification`에서 사용된 예시를 볼 수 있습니다:

```py
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

이 경우, 코드는 다음과 같이 복사됩니다:
- `MobileBert`에서 `Bert`로(예: `MobileBertModel`을 init에서 사용할 때)
- `mobilebert`에서 `bert`로(예: `self.mobilebert`를 정의할 때)
- `MOBILEBERT`에서 `BERT`로(`MOBILEBERT_INPUTS_DOCSTRING` 상수에서)
