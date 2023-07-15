<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 🤗 Transformers 모델을 TensorFlow로 변환하는 방법은 무엇인가요? [[how-to-convert-a-transformers-model-to-tensorflow]]

🤗 Transformers를 사용할 수 있는 여러 가지 프레임워크를 가지고 있으면 응용 프로그램을 설계할 때 그 강점을 활용할 수 있는 유연성이 생깁니다.
그러나 이는 모델 별로 호환성을 추가해야한다는 것을 의미합니다. 좋은 소식은 기존 모델에 TensorFlow 호환성을 추가하는 것이 [새로운 모델을 처음부터 추가하는 것보다 간단하다는 것입니다](add_new_model)! 
큰 TensorFlow 모델을 더 깊이 이해하거나 주요 오픈 소스 기여를 수행하거나 선택한 모델에 TensorFlow를 사용하려는 경우 이 안내서는 여러분을 위한 것입니다.

이 가이드는 Hugging Face 팀의 최소한의 감독 아래에서 🤗 Transformers에서 사용되는 TensorFlow 모델 가중치와/또는 아키텍처를 기여할 수 있는 커뮤니티 구성원인 여러분을 대상으로 합니다. 
새로운 모델을 작성하는 것은 쉬운 일이 아니지만, 이 가이드를 통해 조금 덜 힘들고 훨씬 쉬운 작업으로 만들 수 있습니다. 
우리의 경험을 활용하는 것은 이 프로세스를 점차적으로 더 쉽게 만드는 데 굉장히 중요하며, 따라서 이 가이드에 대한 개선 제안을 적극적으로 권장합니다!

더 깊이 알아보기 전에, 🤗 Transformers에 처음 접하는 경우 다음 자료를 확인하는 것이 좋습니다:
- [🤗 Transformers의 일반 개요]
(add_new_model#general-overview-of-transformers)
- [Hugging Face의 TensorFlow 철학](https://huggingface.co/blog/tensorflow-philosophy)

이 가이드의 나머지 부분에서는 새로운 TensorFlow 모델 아키텍처를 추가하는 데 필요한 단계, PyTorch를 TensorFlow 모델 가중치로 변환하는 절차 및 ML 프레임워크 간의 불일치를 효율적으로 디버깅하는 방법을 알게 될 것입니다. 시작해봅시다!

<팁>

사용하려는 모델이 이미 해당하는 TensorFlow 아키텍처가 있는지 확실하지 않은 경우,

모델 선택지([example](https://huggingface.co/bert-base-uncased/blob/main/config.json#L14))의 `config.json`의 `model_type` 필드를 확인해보세요. 🤗 Transformers의 해당 모델 폴더에는 "modeling_tf"로 시작하는 파일이 있는 경우, 해당 모델에는 해당 TensorFlow 아키텍처([example](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert))가 있다는 의미입니다.

</팁>

## TensorFlow 모델 아키텍처 코드 추가하는 단계별 가이드 

큰 모델 아키텍처를 설계하는 여러 가지 방법이 있으며, 해당 디자인을 구현하는 여러 가지 방법도 있습니다. 
그러나 우리는 [🤗 Transformers의 일반 개요](add_new_model#general-overview-of-transformers)에서 언급한 대로 일관된 설계 선택에 따라 🤗 Transformers의 사용 편의성이 달려 있음을 상기시켜 드리겠습니다. 
우리의 경험에 따르면 TensorFlow 모델을 추가하는 데 관련된 중요한 몇 가지 사항을 알려 드릴 수 있습니다:

- 휠을 다시 발명하지 마세요! 대개 최소한 두 개의 참조 구현을 확인해야 합니다: 구현하려는 모델의 PyTorch 동등 버전 및 동일한 문제 유형에 대한 다른 TensorFlow 모델.
- 우수한 모델 구현은 시간에 따라 테스트를 통과합니다. 이것은 코드가 아름답다는 이유가 아니라 코드가 명확하고 디버그 및 개선이 쉬워야 하기 때문입니다. TensorFlow 구현에서 동일한 패턴을 복제하고 PyTorch 구현과의 불일치를 최소화하여 유지 관리자의 업무를 쉽게 할 경우, 기여한 사항이 오래 유지됨을 보장할 수 있습니다.
- 도움이 필요한 경우 도움을 요청하세요! 🤗 Transformers 팀은 여러분을 돕기 위해 여기에 있으며, 여러분이 직면한 동일한 문제에 대한 해결책을 이미 찾은 경우도 있을 수 있습니다.

TensorFlow 모델 아키텍처를 추가하는 데 필요한 단계의 개요를 제공합니다:
1. 변환하려는 모델 선택
2. transformers 개발 환경 준비
3. (선택 사항) 이론적 측면 및 기존 구현 이해
4. 모델 아키텍처 구현
5. 모델 테스트 구현
6. pull 요청 제출
7. (선택 사항) 데모 빌드 및 공유

### 1.-3. 모델 기여 준비 [[13-prepare-your-model-contribution]]

**1. 변환하려는 모델 선택**

우선 기본 사항부터 시작해 보겠습니다. 알고리즘을 변환하려는 아키텍처를 알아야 합니다. 
특정 아키텍처에 대한 목표가 없는 경우, 🤗 Transformers 팀에게 제안을 요청하는 것은 여러분의 영향력을 극대화하는 좋은 방법입니다. 
우리는 TensorFlow에서 빠져 있는 가장 유명한 아키텍처로 이끌어 드리겠습니다. 
TensorFlow에서 사용할 모델이 이미 🤗 Transformers에 TensorFlow 아키텍처 구현이 있지만 가중치가 없는 경우, 
이 페이지의 [가중치 추가 섹션](#adding-tensorflow-weights-to-hub)으로 바로 이동하셔도 됩니다.

간단히 말해서, 이 안내서의 나머지 부분은 TensorFlow 버전의 *BrandNewBert*([가이드](add_new_model)와 동일한 예제)를 기여하려고 결정했다고 가정합니다.

<팁>

TensorFlow 모델 아키텍처에 작업을 시작하기 전에 해당 작업이 진행 중인지 확인하세요. 
`BrandNewBert`를 검색하여
[풀 요청 GitHub 페이지](https://github.com/huggingface/transformers/pulls?q=is%3Apr)에서 TensorFlow 관련 풀 요청이 없는지 확인할 수 있습니다.

</팁>

**2. transformers 개발 환경 준비**


모델 아키텍처를 선택한 후, 관련 작업을 수행할 의도를 신호로 알리기 위해 draft PR을 엽니다. 환경을 설정하고 draft PR을 열려면 아래 지침을 따르세요.

1. 'Fork' 버튼을 클릭하여 [저장소](https://github.com/huggingface/transformers)를 포크합니다. 이렇게 하면 GitHub 사용자 계정에 코드의 사본이 생성됩니다.

2. `transformers` 포크를 로컬 디스크에 클론하고 기본 저장소를 원격 저장소로 추가합니다.

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 개발 환경을 설정합니다. 예를 들어, 다음 명령을 실행하여 개발 환경을 설정할 수 있습니다.

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```
운영 체제에 따라서 Transformers의 선택적 종속성이 증가하면서 이 명령으로 실패할 수도 있습니다. 그런 경우 TensorFlow를 설치한 후 다음을 수행하세요.
```bash
pip install -e ".[quality]"
```

**참고:** CUDA를 설치할 필요는 없습니다. 새로운 모델이 CPU에서 작동하도록 만드는 것만으로 충분합니다.

4. 메인 브랜치에서 설명적인 이름으로 브랜치를 만듭니다.

```bash
git checkout -b add_tf_brand_new_bert
```

5. 현재 메인 브랜치로 페치 및 리베이스합니다.

```bash
git fetch upstream
git rebase upstream/main
```

6. `transformers/src/models/brandnewbert/`에 `modeling_tf_brandnewbert.py`라는 빈 `.py` 파일을 추가합니다. 이 파일이 TensorFlow 모델 파일이 될 것입니다.

7. 변경 사항을 계정에 푸시합니다.

```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. 만족스러운 경우 GitHub에서 포크된 웹 페이지로 이동합니다. "Pull request"를 클릭합니다. Hugging Face 팀의 GitHub 핸들을 리뷰어에 추가하여 앞으로의 변경 사항에 대해 Hugging Face 팀이 통지를 받을 수 있도록 합니다.


9. GitHub 풀 요청 웹 페이지 오른쪽에 있는 "Convert to draft"를 클릭하여 PR을 드래프트로 변경합니다.

이제 🤗 Transformers에서 *BrandNewBert*를 TensorFlow로 변환할 개발 환경을 설정했습니다.


**3. (선택 사항) 이론적 측면 및 기존 구현 이해**


이론적 측면을 이해하는 데 시간을 할애해야 합니다. *BrandNewBert*의 논문을 읽어야 할 수도 있습니다. 이해하기 어려운 부분이 많을 수 있습니다. 그렇다고 해서 걱정하지 마세요! 목표는 논문의 심도있는 이론적 이해가 아니라 TensorFlow를 사용하여 🤗 Transformers에 모델을 효과적으로 다시 구현하는 데 필요한 정보를 추출하는 것입니다. 그럼에도 불구하고 이 단계에서 모델(e.g. [model docs for BERT](model_doc/bert))의 모든 면을 깊이 이해해야 하는 것은 필요하지 않습니다.

모델의 기본 사항을 이해한 후, 기존 구현을 이해하는 것이 중요합니다. 이는 작업 중인 모델에 대한 작동 구현과 모델에 대한 기대 사항을 미리 확인하는 좋은 기회입니다. 또한 TensorFlow 측면에서의 기술적 도전을 예상할 수 있습니다.

막대한 양의 정보를 처음으로 소화했을 때 압도당하는 것은 자연스러운 일입니다. 이 단계에서 모델의 모든 측면을 이해해야 하는 요구 사항은 전혀 없습니다. 그러나 우리는 Hugging Face의 [포럼](https://discuss.huggingface.co/)을 통해 질문이 있는 경우 대답을 구할 것을 권장합니다.

### 4. 모델 구현 [[4-model-implementation]]


이제 드디어 코딩을 시작할 시간입니다. 우리의 제안된 시작점은 PyTorch 파일 자체입니다: `modeling_brand_new_bert.py`의 내용을 
`src/transformers/models/brand_new_bert/`에 복사하여
`modeling_tf_brand_new_bert.py`로 업데이트합니다. 이 섹션의 목표는 파일을 수정하고 🤗 Transformers의 import 구조를 업데이트하여 `TFBrandNewBert` 및 `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`가 성공적으로 작동하는 TensorFlow *BrandNewBert* 모델을 가져올 수 있도록 하는 것입니다.

유감스럽게도, PyTorch 모델을 TensorFlow로 변환하는 규칙은 없습니다. 그러나 프로세스를 가능한한 원활하게 만들기 위해 다음 팁을 따를 수 있습니다.
- 모든 클래스 이름 앞에 `TF`를 붙입니다(예: `BrandNewBert`는 `TFBrandNewBert`가 됩니다).
- 대부분의 PyTorch 작업에는 직접적인 TensorFlow 대체가 있습니다. 예를 들어, `torch.nn.Linear`는 `tf.keras.layers.Dense`에 해당하고, `torch.nn.Dropout`은 `tf.keras.layers.Dropout`에 해당합니다. 특정 작업에 대해 확신이 없는 경우 [TensorFlow 문서](https://www.tensorflow.org/api_docs/python/tf)나 [PyTorch 문서](https://pytorch.org/docs/stable/)를 참조할 수 있습니다.
- 🤗 Transformers 코드베이스에서 패턴을 찾으세요. 직접적인 대체가 없는 특정 작업을 만나면 다른 사람이 이미 동일한 문제를 해결한 경우가 많습니다.
- 기본적으로 PyTorch와 동일한 변수 이름과 구조를 유지하세요. 이렇게 하면 디버깅과 문제 추적, 그리고 문제 해결 추가가 더 쉬워집니다.
- 일부 레이어는 각 프레임워크마다 다른 기본값을 가지고 있습니다. 대표적인 예로 배치 정규화 레이어의 epsilon은 [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)에서 `1e-5`이고 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)에서 `1e-3`입니다. 문서를 확인하세요!
- PyTorch의 `nn.Parameter` 변수는 일반적으로 TF 레이어의 `build()` 내에서 초기화해야 합니다. 다음 예를 참조하세요: [PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) /
   [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
- PyTorch 모델의 함수 상단에 `#copied from ...`가 있는 경우, TensorFlow 모델도 해당 함수를 복사한 아키텍처에서 사용할 수 있습니다(전제조건은 TensorFlow 아키텍처이어야 함).
- TensorFlow 함수에서 `name` 속성을 올바르게 할당하는 것은 `from_pt=True` 가중치 교차 로딩을 수행하는 데 중요합니다. `name`은 대부분 PyTorch 코드의 해당 변수의 이름입니다. `name`이 제대로 설정되지 않으면 모델 가중치를 로드할 때 오류 메시지에서 확인할 수 있습니다.
- 기본 모델 클래스인 `BrandNewBertModel`의 로직은 실제로 Keras 레이어 서브클래스인 `TFBrandNewBertMainLayer`에 있습니다([예시](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719)). `TFBrandNewBertModel`은 이 레이어를 감싸기만 하는 래퍼 역할을 합니다.
- Keras 모델은 사전 훈련된 가중치를 로드하기 위해 빌드되어야 합니다. 따라서 `TFBrandNewBertPreTrainedModel`은 모델의 입력 예제인 `dummy_inputs`를 유지해야 합니다([예시](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)).
- 도움이 필요한 경우 도움을 요청하세요. 우리는 여기 있어서 도움을 드리기 위해 있는 것입니다! 🤗

모델 파일 자체 외에도 모델 클래스 및 관련 문서 페이지에 대한 포인터를 추가해야 합니다. 이 부분은 다른 PR의 패턴을 따라 완전히 완료할 수 있습니다([예시](https://github.com/huggingface/transformers/pull/18020/files)). 다음은 필요한 수동 변경 목록입니다.

- `src/transformers/__init__.py`에 *BrandNewBert*의 모든 공개 클래스를 포함합니다.
- `src/transformers/models/auto/modeling_tf_auto.py`에서 *BrandNewBert* 클래스를 해당 Auto 클래스에 추가합니다.
- `utils/documentation_tests.txt`에 모델 파일을 문서화하는 테스트 파일 목록을 추가합니다.
- `src/transformers/utils/dummy_tf_objects.py`에 *BrandNewBert*와 관련된 레이지 로딩 클래스를 추가합니다.
- `src/transformers/models/brand_new_bert/__init__.py`에서 공개 클래스에 대한 import 구조를 업데이트합니다.
- `docs/source/en/model_doc/brand_new_bert.md`에서 *BrandNewBert*의 공개 메서드에 대한 문서 포인터를 추가합니다.
- `docs/source/en/model_doc/brand_new_bert.md`의 *BrandNewBert* 기여자 목록에 자신을 추가합니다.

🎉 축하합니다! TensorFlow 모델 아키텍처를 구현하는 데 성공했습니다.


### 5. 모델 테스트 구현 [[5-add-model-tests]]

이제 TensorFlow 모델을 테스트하는 구현을 작성할 차례입니다. 이를 통해 모델이 예상대로 작동하는지 확인할 수 있습니다.

🤗 Transformers는 다양한 유형의 테스트를 지원합니다. 다음은 추가할 수 있는 몇 가지 테스트 유형의 예입니다:
- `test_model_loading`: 특정 체크포인트를 사용하여 모델을 로드하고, 모델의 아웃풋을 기대값과 비교합니다.
- `test_model_forwarding`: 모델의 입력을 주고 결과를 얻고, 기대값과 비교합니다.
- `test_attention_is_consistent`: 어텐션 계산이 올바르게 이루어지는지 확인합니다.
- `test_output_all_attentions`: 모든 어텐션 값을 출력하고, 이전 버전의 출력값과 비교합니다.
- `test_hidden_states_are_reproducible`: 랜덤 시드를 고정하여 모델을 두 번 실행하고, 두 번째 실행의 출력값이 첫 번째 실행과 동일한지 확인합니다.

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

테스트를 추가하는 방법에 대한 자세한 내용은 [🤗 Transformers의 테스트 가이드](https://huggingface.co/transformers/contributing.html#running-tests)를 참조하세요.

테스트를 작성하는 동안 기존 모델 테스트를 참고하는 것이 도움이 될 수 있습니다. 기존 모델의 테스트 파일을 확인하고, 기존 테스트와 동일한 유형의 테스트를 작성하는 방법을 이해하세요. 이를 통해 코드를 작성하고 기존 모델의 동작을 신뢰할 수 있습니다.

### 6. 풀 요청 제출 [[67-ensure-everyone-can-use-your-model]]

모델 아키텍처 구현 및 테스트 구현이 완료되었다면, 이제 풀 요청(PR)을 제출할 차례입니다. 다음은 풀 요청을 제출하는 단계입니다:

1. 변경 사항을 본인의 포크 저장소에 푸시합니다.

2. GitHub 웹 페이지로 이동하여 포크된 저장소로 이동합니다.

3. "Compare & pull request" 버튼을 클릭하여 PR을 생성합니다.

4. 제목과 설명을 작성합니다. PR 설명에는 변경한 내용에 대한 상세한 정보를 포함해야 합니다. 또한 이전에 추가된 테스트가 실패하지 않는지 확인하세요.

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

5. Hugging Face 팀의 GitHub 핸들을 리뷰어로 추가하여 앞으로의 변경 사항에 대해 통지를 받을 수 있도록 합니다.

6. PR을 제출합니다.

축하합니다! 이제 풀 요청이 만들어졌습니다. Hugging Face 팀의 리뷰를 기다리며 코드에 대한 피드백을 받을 수 있습니다.

### 7. (선택 사항) 데모 빌드 및 공유 [[adding-tensorflow-weights-to-hub]]


모델 아키텍처가 구현되고 풀 요청이 완료되면, 🤗 Transformers의 데모에 해당 모델을 추가하여 다른 사용자들과 공유할 수 있습니다. 이를 통해 사용자들이 해당 모델을 사용하는 방법을 시도해 볼 수 있습니다.

모델을 데모에 추가하려면 다음 단계를 수행하세요:

1. 풀 요청이 승인되었는지 확인하세요.

2. 🤗 Transformers 데모 저장소를 로컬 디스크에 클론합니다.


3. `transformers/examples/legacy/run_generation.py`와 같은 기존 데모 코드에서 사용할 수 있도록 모델을 추가합니다. 모델 파일은 `huggingface.co/models`에 호스팅되어야 합니다.

4. 변경 사항을 포함하여 본인의 포크 저장소에 커밋 및 푸시합니다.

5. Hugging Face 팀에게 데모에 대한 PR을 제출합니다.

데모를 제공하면 다른 사용자들이 새로운 모델을 쉽게 사용하고 결과를 확인할 수 있습니다.

이제 🤗 Transformers에서 TensorFlow 모델을 구현하고 기여하기 위한 모든 단계를 알게 되었습니다. 커뮤니티에 더 많은 모델 아키텍처를 추가하여 다른 사용자들과 공유하십시오!