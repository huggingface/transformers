<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 어떻게 🤗 Transformers 모델을 TensorFlow로 변환하나요? [[how-to-convert-a-transformers-model-to-tensorflow]]

🤗 Transformers에서처럼 사용할 수 있는 여러 가지 프레임워크가 있다는 것은 애플리케이션을 설계할 때 그들의 강점을 유연하게 이용할 수 있다는 장점이 있지만, 모델 별로 호환성을 추가해야 한다는 단점 또한 존재한다는 것을 의미합니다. 좋은 소식은 기존 모델에 TensorFlow 호환성을 추가하는 것이 [처음부터 새로운 모델을 추가하는 것](add_new_model)보다도 간단하다는 것입니다! 

만약 대규모 TensorFlow 모델을 더 깊이 이해하려거나, 오픈 소스에 큰 기여를 하려거나, 선택한 모델에 Tensorflow를 활용하려한다면, 이 안내서는 여러분께 도움이 될 것입니다.

이 가이드는 Hugging Face 팀의 최소한의 감독 아래에서 🤗 Transformers에서 사용되는 TensorFlow 모델 가중치와/또는 아키텍처를 기여할 수 있는 커뮤니티 구성원인 여러분을 대상으로 합니다. 
새로운 모델을 작성하는 것은 쉬운 일이 아니지만, 이 가이드를 통해 조금 덜 힘들고 훨씬 쉬운 작업으로 만들 수 있습니다. 
모두의 경험을 모으는 것은 이 작업을 점차적으로 더 쉽게 만드는 데 굉장히 중요하기 때문에, 이 가이드를 개선시킬만한 제안이 떠오르면 공유하시는걸 적극적으로 권장합니다!

더 깊이 알아보기 전에, 🤗 Transformers를 처음 접하는 경우 다음 자료를 확인하는 것이 좋습니다:
- [🤗 Transformers의 일반 개요](add_new_model#general-overview-of-transformers)
- [Hugging Face의 TensorFlow 철학](https://huggingface.co/blog/tensorflow-philosophy)

이 가이드의 나머지 부분에서는 새로운 TensorFlow 모델 아키텍처를 추가하는 데 필요한 단계, Pytorch를 TensorFlow 모델 가중치로 변환하는 절차 및 ML 프레임워크 간의 불일치를 효율적으로 디버깅하는 방법을 알게 될 것입니다. 시작해봅시다!

<Tip>

사용하려는 모델이 이미 해당하는 TensorFlow 아키텍처가 있는지 확실하지 않나요?

선택한 모델([예](https://huggingface.co/bert-base-uncased/blob/main/config.json#L14))의 `config.json`의 `model_type` 필드를 확인해보세요. 🤗 Transformers의 해당 모델 폴더에는 "modeling_tf"로 시작하는 파일이 있는 경우, 해당 모델에는 해당 TensorFlow 아키텍처([예](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert))가 있다는 의미입니다.

</Tip>

## TensorFlow 모델 아키텍처 코드 추가하는 단계별 가이드 [[step-by-step-guide-to add-tensorFlow-model-architecture-code]]

대규모 아키텍처를 가진 모델을 설계하는 방법에는 여러가지가 있으며, 해당 설계를 구현하는 방법도 여러 가지입니다. 
그러나 우리는 [🤗 Transformers 일반 개요](add_new_model#general-overview-of-transformers)에서 언급한 대로 일관된 설계 선택에 따라야지만 🤗 Transformers를 사용하기 편할 것이라는 확고한 의견을 가지고 있습니다.
우리의 경험을 통해 TensorFlow 모델을 추가하는 데 관련된 중요한 몇 가지 사항을 알려 드릴 수 있습니다:

- 이미 있는걸 다시 개발하려 하지 마세요! 최소한 2개의 이미 구현된 모델을 대개 참조해야 합니다. 구현하려는 모델과 기능상 동일한 Pytorch 모델 하나와 같은 문제 유형을 풀고 있는 다른 TensorFlow 모델 하나를 살펴보세요.
- 우수한 모델 구현은 시간이 지나도 남아있습니다. 이것은 코드가 아름답다는 이유가 아니라 코드가 명확하고 디버깅 및 개선이 쉽기 때문입니다. TensorFlow 구현에서 다른 모델들과 패턴을 똑같이 하고 Pytorch 구현과의 불일치를 최소화하여 메인테이너의 업무를 쉽게 한다면, 기여한 코드가 오래도록 유지될 수 있습니다.
- 필요하다면 도움을 요청하세요! 🤗 Transformers 팀은 여러분을 돕기 위해 있으며, 여러분이 직면한 동일한 문제에 대한 해결책을 이미 찾은 경우도 있을 수 있습니다.

TensorFlow 모델 아키텍처를 추가하는 데 필요한 단계를 개략적으로 써보면:
1. 변환하려는 모델 선택
2. transformers 개발 환경 준비
3. (선택 사항) 이론적 측면 및 기존 구현 이해
4. 모델 아키텍처 구현
5. 모델 테스트 구현
6. PR (pull request) 제출
7. (선택 사항) 데모 빌드 및 공유

### 1.-3. 모델 기여 준비 [[1.-3.-prepare-your-model-contribution]]

**1. 변환하려는 모델 선택**

우선 기본 사항부터 시작해 보겠습니다. 먼저 변환하려는 아키텍처를 알아야 합니다. 
특정 아키텍처에 대한 관심 없는 경우, 🤗 Transformers 팀에게 제안을 요청하는 것은 여러분의 영향력을 극대화하는 좋은 방법입니다. 
우리는 TensorFlow에서 빠져 있는 가장 유명한 아키텍처로 이끌어 드리겠습니다. 
TensorFlow에서 사용할 모델이 이미 🤗 Transformers에 TensorFlow 아키텍처 구현이 있지만 가중치가 없는 경우, 
이 페이지의 [가중치 추가 섹션](#adding-tensorflow-weights-to-hub)으로 바로 이동하셔도 됩니다.

간단히 말해서, 이 안내서의 나머지 부분은 TensorFlow 버전의 *BrandNewBert*([가이드](add_new_model)와 동일한 예제)를 기여하려고 결정했다고 가정합니다.

<Tip>

TensorFlow 모델 아키텍처에 작업을 시작하기 전에 해당 작업이 진행 중인지 확인하세요. 
`BrandNewBert`를 검색하여
[pull request GitHub 페이지](https://github.com/huggingface/transformers/pulls?q=is%3Apr)에서 TensorFlow 관련 pull request가 없는지 확인할 수 있습니다.

</Tip>

**2. transformers 개발 환경 준비**


모델 아키텍처를 선택한 후, 관련 작업을 수행할 의도를 미리 알리기 위해 Draft PR을 여세요. 아래 지침대로 하시면 환경을 설정하고 Draft PR을 열 수 있습니다.

1. 'Fork' 버튼을 클릭하여 [리포지터리](https://github.com/huggingface/transformers)를 포크하세요. 이렇게 하면 GitHub 사용자 계정에 코드의 사본이 생성됩니다.


2. `transformers` 포크를 로컬 디스크에 클론하고 원본 리포지터리를 원격 리포지터리로 추가하세요.

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 개발 환경을 설정하세요. 예를 들어, 다음 명령을 실행하여 개발 환경을 설정할 수 있습니다.

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

운영 체제에 따라서 Transformers의 선택적 종속성이 증가하면서 위 명령이 실패할 수도 있습니다. 그런 경우 TensorFlow를 설치한 후 다음을 실행하세요.

```bash
pip install -e ".[quality]"
```

**참고:** CUDA를 설치할 필요는 없습니다. 새로운 모델이 CPU에서 작동하도록 만드는 것만으로 충분합니다.

4. 메인 브랜치에서 만드려는 기능이 잘 표현되는 이름으로 브랜치를 만듭니다.

```bash
git checkout -b add_tf_brand_new_bert
```

5. 메인 브랜치의 현재 상태를 페치(fetch)하고 리베이스하세요.

```bash
git fetch upstream
git rebase upstream/main
```

6. `transformers/src/models/brandnewbert/`에 `modeling_tf_brandnewbert.py`라는 빈 `.py` 파일을 추가하세요. 이 파일이 TensorFlow 모델 파일이 될 것입니다.

7. 변경 사항을 계정에 푸시하세요.

```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. 만족스러운 경우 GitHub에서 포크된 웹 페이지로 이동합니다. "Pull request"를 클릭합니다. Hugging Face 팀의 GitHub ID를 리뷰어로 추가해서, 앞으로의 변경 사항에 대해 Hugging Face 팀이 알림을 받을 수 있도록 합니다.


9. GitHub Pull Requests 페이지의 오른쪽에 있는 "Convert to draft"를 클릭하여 PR을 초안으로 변경하세요.

이제 🤗 Transformers에서 *BrandNewBert*를 TensorFlow로 변환할 개발 환경을 설정했습니다.


**3. (선택 사항) 이론적 측면 및 기존 구현 이해**


*BrandNewBert*처럼 자세한 글이 있다면 시간을 내어 논문을 읽는걸 추천드립니다. 이해하기 어려운 부분이 많을 수 있습니다. 그렇다고 해서 걱정하지 마세요! 목표는 논문의 심도있는 이론적 이해가 아니라 TensorFlow를 사용하여 🤗 Transformers에 모델을 효과적으로 다시 구현하는 데 필요한 필수 정보를 추출하는 것입니다. 많은 시간을 이론적 이해에 투자할 필요는 없지만 실용적인 측면에서 현재 존재하는 모델 문서 페이지(e.g. [model docs for BERT](model_doc/bert))에 집중하는 것이 좋습니다.


모델의 기본 사항을 이해한 후, 기존 구현을 이해하는 것이 중요합니다. 이는 작업 중인 모델에 대한 실제 구현이 여러분의 기대와 일치함을 확인하고, TensorFlow 측면에서의 기술적 문제를 예상할 수 있습니다.

막대한 양의 정보를 처음으로 학습할 때 압도당하는 것은 자연스러운 일입니다. 이 단계에서 모델의 모든 측면을 이해해야 하는 필요는 전혀 없습니다. 그러나 우리는 Hugging Face의 [포럼](https://discuss.huggingface.co/)을 통해 질문이 있는 경우 대답을 구할 것을 권장합니다.

### 4. 모델 구현 [[4-model-implementation]]


이제 드디어 코딩을 시작할 시간입니다. 우리의 제안된 시작점은 PyTorch 파일 자체입니다: `modeling_brand_new_bert.py`의 내용을 
`src/transformers/models/brand_new_bert/` 내부의
`modeling_tf_brand_new_bert.py`에 복사합니다. 이 섹션의 목표는 파일을 수정하고 🤗 Transformers의 import 구조를 업데이트하여 `TFBrandNewBert` 및 `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`가 성공적으로 작동하는 TensorFlow *BrandNewBert* 모델을 가져올 수 있도록 하는 것입니다.

유감스럽게도, PyTorch 모델을 TensorFlow로 변환하는 규칙은 없습니다. 그러나 프로세스를 가능한한 원활하게 만들기 위해 다음 팁을 따를 수 있습니다.

- 모든 클래스 이름 앞에 `TF`를 붙입니다(예: `BrandNewBert`는 `TFBrandNewBert`가 됩니다).
- 대부분의 PyTorch 작업에는 직접적인 TensorFlow 대체가 있습니다. 예를 들어, `torch.nn.Linear`는 `tf.keras.layers.Dense`에 해당하고, `torch.nn.Dropout`은 `tf.keras.layers.Dropout`에 해당합니다. 특정 작업에 대해 확신이 없는 경우 [TensorFlow 문서](https://www.tensorflow.org/api_docs/python/tf)나 [PyTorch 문서](https://pytorch.org/docs/stable/)를 참조할 수 있습니다.
- 🤗 Transformers 코드베이스에서 패턴을 찾으세요. 직접적인 대체가 없는 특정 작업을 만나면 다른 사람이 이미 동일한 문제를 해결한 경우가 많습니다.
- 기본적으로 PyTorch와 동일한 변수 이름과 구조를 유지하세요. 이렇게 하면 디버깅과 문제 추적, 그리고 문제 해결 추가가 더 쉬워집니다.
- 일부 레이어는 각 프레임워크마다 다른 기본값을 가지고 있습니다. 대표적인 예로 배치 정규화 레이어의 epsilon은 [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)에서 `1e-5`이고 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)에서 `1e-3`입니다. 문서를 모두 확인하세요!
- PyTorch의 `nn.Parameter` 변수는 일반적으로 TF 레이어의 `build()` 내에서 초기화해야 합니다. 다음 예를 참조하세요: [PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) /
   [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
- PyTorch 모델의 함수 상단에 `#copied from ...`가 있는 경우, TensorFlow 모델에 TensorFlow 아키텍처가 있다면 TensorFlow 모델이 해당 함수를 복사한 아키텍처에서 사용할 수 있습니다.
- TensorFlow 함수에서 `name` 속성을 올바르게 할당하는 것은 `from_pt=True` 가중치 교차 로딩을 수행하는 데 중요합니다. `name`은 대부분 PyTorch 코드의 해당 변수의 이름입니다. `name`이 제대로 설정되지 않으면 모델 가중치를 로드할 때 오류 메시지에서 확인할 수 있습니다.
- 기본 모델 클래스인 `BrandNewBertModel`의 로직은 실제로 Keras 레이어 서브클래스([예시](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719))인 `TFBrandNewBertMainLayer`에 있습니다. `TFBrandNewBertModel`은 이 레이어를 감싸기만 하는 래퍼 역할을 합니다.
- Keras 모델은 사전 훈련된 가중치를 로드하기 위해 빌드되어야 합니다. 따라서 `TFBrandNewBertPreTrainedModel`은 모델의 입력 예제인 `dummy_inputs`([예시](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)) 유지해야 합니다.
- 도움이 필요한 경우 도움을 요청하세요. 우리는 여기 있어서 도움을 드리기 위해 있는 것입니다! 🤗

모델 파일 자체 외에도 모델 클래스 및 관련 문서 페이지에 대한 포인터를 추가해야 합니다. 이 부분은 다른 PR([예시](https://github.com/huggingface/transformers/pull/18020/files))의 패턴을 따라 완전히 완료할 수 있습니다. 다음은 필요한 수동 변경 목록입니다.

- `src/transformers/__init__.py`에 *BrandNewBert*의 모든 공개 클래스를 포함합니다.
- `src/transformers/models/auto/modeling_tf_auto.py`에서 *BrandNewBert* 클래스를 해당 Auto 클래스에 추가합니다.
- `src/transformers/utils/dummy_tf_objects.py`에 *BrandNewBert*와 관련된 레이지 로딩 클래스를 추가합니다.
- `src/transformers/models/brand_new_bert/__init__.py`에서 공개 클래스에 대한 import 구조를 업데이트합니다.
- `docs/source/en/model_doc/brand_new_bert.md`에서 *BrandNewBert*의 공개 메서드에 대한 문서 포인터를 추가합니다.
- `docs/source/en/model_doc/brand_new_bert.md`의 *BrandNewBert* 기여자 목록에 자신을 추가합니다.
- 마지막으로 ✅ 녹색 체크박스를 TensorFlow 열 docs/source/en/index.md 안 BrandNewBert에 추가합니다.

구현이 만족하면 다음 체크리스트를 실행하여 모델 아키텍처가 준비되었는지 확인하세요.  

1. 훈련 시간에 다르게 동작하는 `training` 인수로 불리는 모든 레이어(예: Dropout)는 최상위 클래스에서 전파됩니다.
2. #copied from ...가능할 때마다 사용했습니다.
3. `TFBrandNewBertMainLayer`와 그것을 사용하는 모든 클래스는 `call`함수로 `@unpack_inputs`와 함께 데코레이터 됩니다.
4. `TFBrandNewBertMainLayer`는 `@keras_serializable`로 데코레이터 됩니다.
5. TensorFlow 모델은 `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`를 사용하여 PyTorch 가중치에서 로드할 수 있습니다.
6. 예상 입력 형식을 사용하여 TensorFlow 모델을 호출할 수 있습니다.

### 5. 모델 테스트 구현 [[5-add-model-tests]]

TensorFlow 모델 아키텍처를 구현하는 데 성공했습니다! 이제 TensorFlow 모델을 테스트하는 구현을 작성할 차례입니다. 이를 통해 모델이 예상대로 작동하는지 확인할 수 있습니다. 이전에 우리는 `test_modeling_brand_new_bert.py` 파일을 `tests/models/brand_new_bert/ into test_modeling_tf_brand_new_bert.py`에 복사한 뒤, TensorFlow로 교체하는 것이 좋습니다. 지금은, 모든 `.from_pretrained()`을 `from_pt=True`를 사용하여 존재하는 Pytorch 가중치를 가져오도록 해야합니다.  

완료하셨으면, 이제 진실의 순간이 찾아왔습니다: 테스트를 실행해 보세요! 😬

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

오류가 많이 나타날 것이지만 괜찮습니다! 기계 학습 모델을 디버깅하는 것은 악명높게 어려우며 성공의 핵심 요소는 인내심입니다 (`breakpoint()`도 필요합니다). 우리의 경험상으로는 ML 프레임워크 사이의 미묘한 불일치로 인해 가장 어려운 문제가 발생합니다. 이에 대한 몇 가지 지침이 이 가이드의 끝 부분에 있습니다. 다른 경우에는 일반 테스트가 직접 모델에 적용되지 않을 수 있으며, 이 경우 모델 테스트 클래스 레벨에서 재정의를 제안합니다. 문제가 무엇이든지 상관없이 문제가 있으면 당신이 고립되었다면 draft pull request에서 도움을 요청하는 것이 좋습니다.

모든 테스트가 통과되면 축하합니다. 이제 모델을 🤗 Transformers 라이브러리에 추가할 준비가 거의 완료된 것입니다! 🎉


테스트를 추가하는 방법에 대한 자세한 내용은 [🤗 Transformers의 테스트 가이드](https://huggingface.co/transformers/contributing.html#running-tests)를 참조하세요.

### 6.-7. 모든 사용자가 당신의 모델을 사용할 수 있게 하기 [[6.-7.-ensure-everyone -can-use-your-model]]

**6. 풀 요청 제출하기**

구현과 테스트가 완료되면 풀 요청을 제출할 시간입니다. 코드를 푸시하기 전에 코드 서식 맞추기 유틸리티인 `make fixup` 🪄 를 실행하세요. 이렇게 하면 자동으로 서식 오류를 수정하며 자동 검사가 실패하는 것을 방지할 수 있습니다.

이제 드래프트 풀 요청을 실제 풀 요청으로 변환하는 시간입니다. "리뷰 준비됨" 버튼을 클릭하고 Joao (`@gante`)와 Matt (`@Rocketknight1`)를 리뷰어로 추가하세요. 모델 풀 요청에는 적어도 3명의 리뷰어가 필요하지만, 그들이 당신의 모델에 적절한 추가 리뷰어를 찾을 것입니다.

모든 리뷰어들이 PR 상태에 만족하면 마지막으로 `.from_pretrained()` 호출에서 `from_pt=True` 플래그를 제거하는 것입니다. TensorFlow 가중치가 없기 때문에 이를 추가해야 합니다! 이를 수행하는 방법은 아래 섹션의 지침을 확인하세요.

마침내 TensorFlow 가중치가 병합되고, 적어도 3명의 리뷰어 승인을 받았으며 모든 CI 검사가 통과되었다면, 로컬로 테스트를 한 번 더 확인하세요.

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

그리고 우리는 당신의 PR을 병합할 것입니다! 마일스톤 달성을 축하드립니다! 🎉

**7. (선택 사항) 데모를 만들고 세상과 공유하기**

오픈 소스의 가장 어려운 부분 중 하나는 발견입니다. 다른 사용자들이 당신의 멋진 TensorFlow 기여를 어떻게 알 수 있을까요? 물론 적절한 커뮤니케이션으로 가능합니다! 📣

커뮤니티와 모델을 공유하는 두 가지 주요 방법이 있습니다:
- 데모 만들기. Gradio 데모, 노트북 및 모델을 자랑하는 다른 재미있는 방법을 포함합니다. [커뮤니티 기반 데모](https://huggingface.co/docs/transformers/community)에 노트북을 추가하는 것을 적극 권장합니다.
- Twitter와 LinkedIn과 같은 소셜 미디어에 이야기 공유하기. 당신의 작업에 자랑스러워하고 커뮤니티와 당신의 업적을 공유해야 합니다. 이제 당신의 모델은 전 세계의 수천 명의 엔지니어와 연구원들에 의해 사용될 수 있습니다 🌍! 우리는 당신의 게시물을 리트윗하고 커뮤니티와 함께 당신의 작업을 공유하는 데 도움이 될 것입니다.


## 🤗 허브에 TensorFlow 가중치 추가하기 [[adding-tensorFlow-weights-to-🤗-hub]]

TensorFlow 모델 아키텍처가 🤗 Transformers에서 사용 가능하다고 가정하고, PyTorch 가중치를 TensorFlow 가중치로 변환하는 것은 쉽습니다!

다음은 그 방법입니다:
1. 터미널에서 Hugging Face 계정으로 로그인되어 있는지 확인하십시오. `huggingface-cli login` 명령어를 사용하여 로그인할 수 있습니다. (액세스 토큰은 [여기](https://huggingface.co/settings/tokens)에서 찾을 수 있습니다.)
2. `transformers-cli pt-to-tf --model-name foo/bar`를 실행하십시오. 여기서 `foo/bar`는 변환하려는 PyTorch 가중치가 있는 모델 저장소의 이름입니다.
3. 방금 만든 🤗 허브 PR에서 `@joaogante`와 `@Rocketknight1`을 태그합니다.

그게 다입니다! 🎉


## ML 프레임워크 간 디버깅 🐛[[debugging-mismatches-across-ml-frameworks]]

새로운 아키텍처를 추가하거나 기존 아키텍처에 대한 TensorFlow 가중치를 생성할 때, PyTorch와 TensorFlow 간의 불일치로 인한 오류가 발생할 수 있습니다. 심지어 두 프레임워크의 모델 아키텍처 코드가 동일해 보일 수도 있습니다. 무슨 일이 벌어지고 있는 걸까요? 🤔

먼저, 이러한 불일치를 이해하는 이유에 대해 이야기해 보겠습니다. 많은 커뮤니티 멤버들은 🤗 Transformers 모델을 그대로 사용하고, 우리의 모델이 예상대로 작동할 것이라고 믿습니다. 두 프레임워크 간에 큰 불일치가 있으면 모델이 적어도 하나의 프레임워크에 대한 참조 구현을 따르지 않음을 의미합니다. 이는 모델이 의도한 대로 작동하지 않을 수 있음을 나타냅니다. 이는 아예 실행되지 않는 모델보다 나쁠 수 있습니다! 따라서 우리는 모든 모델의 프레임워크 불일치를 `1e-5`보다 작게 유지하는 것을 목표로 합니다.

기타 숫자 문제와 마찬가지로, 세세한 문제가 있습니다. 그리고 세세함에 집중하는 공정에서 필수 요소는 인내심입니다. 이러한 종류의 문제가 발생할 때 권장되는 작업 흐름은 다음과 같습니다:
1. 불일치의 원인을 찾아보십시오. 변환 중인 모델은 아마도 특정 지점까지 거의 동일한 내부 변수를 가지고 있을 것입니다. 두 프레임워크의 아키텍처에 `breakpoint()` 문을 넣고, 위에서 아래로 숫자 변수의 값을 비교하여 문제의 근원을 찾아냅니다.
2. 이제 문제의 근원을 찾았으므로 🤗 Transformers 팀에 연락하세요. 우리는 비슷한 문제를 이전에 겪었을 수 있으며 빠르게 해결책을 제공할 수 있습니다. 예외적인 경우에는 StackOverflow와 GitHub 이슈와 같은 인기있는 페이지를 확인하십시오.
3. 더 이상 해결책이 없는 경우, 더 깊이 들어가야 합니다. 좋은 소식은 문제의 원인을 찾았으므로 나머지 모델을 추상화하고 문제가 있는 명령어에 초점을 맞출 수 있습니다! 나쁜 소식은 해당 명령어의 소스 구현에 대해 알아봐야 한다는 것입니다. 일부 경우에는 참조 구현에 문제가 있을 수도 있으니 업스트림 저장소에서 이슈를 열기를 꺼리지 마십시오.

어떤 경우에는 🤗 Transformers 팀과의 토론을 통해 불일치를 수정할 수 없을 수도 있습니다. 모델의 출력 레이어에서 불일치가 매우 작지만 숨겨진 상태에서 크게 나타날 수 있기 때문입니다. 이 경우 모델을 배포하는 것을 우선시하기 위해 불일치를 무시하기로 결정할 수도 있습니다. 위에서 언급한 `pt-to-tf` CLI에는 가중치 변환 시 오류 메시지를 무시하는 `--max-error` 플래그가 있습니다.
