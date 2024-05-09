<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hugging Face Transformers를 추가하는 방법은 무엇인가요? [[how-to-add-a-model-to-transformers]]

Hugging Face Transformers 라이브러리는 커뮤니티 기여자들 덕분에 새로운 모델을 제공할 수 있는 경우가 많습니다. 하지만 이는 도전적인 프로젝트이며 Hugging Face Transformers 라이브러리와 구현할 모델에 대한 깊은 이해가 필요합니다. Hugging Face에서는 더 많은 커뮤니티 멤버가 모델을 적극적으로 추가할 수 있도록 지원하고자 하며, 이 가이드를 통해 PyTorch 모델을 추가하는 과정을 안내하고 있습니다 (PyTorch가 설치되어 있는지 확인해주세요).

이 과정을 진행하면 다음과 같은 내용을 이해하게 됩니다:

- 오픈 소스의 모범 사례에 대한 통찰력을 얻습니다.
- 가장 인기 있는 딥러닝 라이브러리의 설계 원칙을 이해합니다.
- 대규모 모델을 효율적으로 테스트하는 방법을 배웁니다.
- `black`, `ruff`, `make fix-copies`와 같은 Python 유틸리티를 통합하여 깔끔하고 가독성 있는 코드를 작성하는 방법을 배웁니다.

Hugging Face 팀은 항상 도움을 줄 준비가 되어 있으므로 혼자가 아니라는 점을 기억하세요. 🤗 ❤️

시작에 앞서 🤗 Transformers에 원하는 모델을 추가하기 위해 [New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml) 이슈를 열어야 합니다. 특정 모델을 기여하는 데 특별히 까다로운 기준을 가지지 않는 경우 [New model label](https://github.com/huggingface/transformers/labels/New%20model)을 필터링하여 요청되지 않은 모델이 있는지 확인하고 작업할 수 있습니다.

새로운 모델 요청을 열었다면 첫 번째 단계는 🤗 Transformers에 익숙해지는 것입니다!

## 🤗 Transformers의 전반적인 개요  [[general-overview-of-transformers]]

먼저 🤗 Transformers에 대한 전반적인 개요를 파악해야 합니다. 🤗 Transformers는 매우 주관적인 라이브러리이기 때문에 해당 라이브러리의 철학이나 설계 선택 사항에 동의하지 않을 수도 있습니다. 그러나 우리의 경험상 라이브러리의 기본적인 설계 선택과 철학은 🤗 Transformers의 규모를 효율적으로 확장하면서 유지 보수 비용을 합리적인 수준으로 유지하는 것입니다.

[라이브러리의 철학에 대한 문서](philosophy)를 읽는 것이 라이브러리를 더 잘 이해하는 좋은 시작점입니다. 모든 모델에 적용하려는 몇 가지 작업 방식에 대한 선택 사항이 있습니다:

- 일반적으로 추상화보다는 구성을 선호합니다.
- 코드를 복제하는 것이 항상 나쁜 것은 아닙니다. 코드의 가독성이나 접근성을 크게 향상시킨다면 복제하는 것은 좋습니다.
- 모델 파일은 가능한 한 독립적으로 유지되어야 합니다. 따라서 특정 모델의 코드를 읽을 때 해당 `modeling_....py` 파일만 확인하면 됩니다.

우리는 라이브러리의 코드가 제품을 제공하는 수단뿐만 아니라 개선하고자 하는 제품이라고도 생각합니다. 따라서 모델을 추가할 때, 사용자는 모델을 사용할 사람뿐만 아니라 코드를 읽고 이해하고 필요한 경우 조정할 수 있는 모든 사람까지도 포함한다는 점을 기억해야 합니다.

이를 염두에 두고 일반적인 라이브러리 설계에 대해 조금 더 자세히 알아보겠습니다.

### 모델 개요 [[overview-of-models]]

모델을 성공적으로 추가하려면 모델과 해당 구성인 [`PreTrainedModel`] 및 [`PretrainedConfig`] 간의 상호작용을 이해하는 것이 중요합니다. 예를 들어, 🤗 Transformers에 추가하려는 모델을 `BrandNewBert`라고 부르겠습니다.

다음을 살펴보겠습니다:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_overview.png"/>

보다시피, 🤗 Transformers에서는 상속을 사용하지만 추상화 수준을 최소한으로 유지합니다. 라이브러리의 어떤 모델에서도 두 수준 이상의 추상화가 존재하지 않습니다. `BrandNewBertModel`은 `BrandNewBertPreTrainedModel`에서 상속받고, 이 클래스는 [`PreTrainedModel`]에서 상속받습니다. 이로써 새로운 모델은 [`PreTrainedModel`]에만 의존하도록 하려고 합니다. 모든 새로운 모델에 자동으로 제공되는 중요한 기능은 [`~PreTrainedModel.from_pretrained`] 및 [`~PreTrainedModel.save_pretrained`]입니다. 이러한 기능 외에도 `BrandNewBertModel.forward`와 같은 다른 중요한 기능은 새로운 `modeling_brand_new_bert.py` 스크립트에서 완전히 정의되어야 합니다. 또한 `BrandNewBertForMaskedLM`과 같은 특정 헤드 레이어를 가진 모델은 `BrandNewBertModel`을 상속받지 않고 forward pass에서 호출할 수 있는 `BrandNewBertModel`을 사용하여 추상화 수준을 낮게 유지합니다. 모든 새로운 모델은 `BrandNewBertConfig`라는 구성 클래스를 필요로 합니다. 이 구성은 항상 [`PreTrainedModel`]의 속성으로 저장되며, 따라서 `BrandNewBertPreTrainedModel`을 상속받는 모든 클래스에서 `config` 속성을 통해 액세스할 수 있습니다:

```python
model = BrandNewBertModel.from_pretrained("brandy/brand_new_bert")
model.config  # model has access to its config
```

모델과 마찬가지로 구성은 [`PretrainedConfig`]에서 기본 직렬화 및 역직렬화 기능을 상속받습니다. 구성과 모델은 항상 *pytorch_model.bin* 파일과 *config.json* 파일로 각각 별도로 직렬화됩니다. [`~PreTrainedModel.save_pretrained`]를 호출하면 자동으로 [`~PretrainedConfig.save_pretrained`]도 호출되므로 모델과 구성이 모두 저장됩니다.


### 코드 스타일 [[code-style]]

새로운 모델을 작성할 때, Transformers는 주관적인 라이브러리이며 몇 가지 독특한 코딩 스타일이 있습니다:

1. 모델의 forward pass는 모델 파일에 완전히 작성되어야 합니다. 라이브러리의 다른 모델에서 블록을 재사용하려면 코드를 복사하여 위에 `# Copied from` 주석과 함께 붙여넣으면 됩니다 (예: [여기](https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L160)를 참조하세요).
2. 코드는 완전히 이해하기 쉬워야 합니다. 변수 이름을 명확하게 지정하고 약어를 사용하지 않는 것이 좋습니다. 예를 들어, `act`보다는 `activation`을 선호합니다. 한 글자 변수 이름은 루프의 인덱스인 경우를 제외하고 권장되지 않습니다.
3. 더 일반적으로, 짧은 마법 같은 코드보다는 길고 명시적인 코드를 선호합니다.
4. PyTorch에서 `nn.Sequential`을 하위 클래스로 만들지 말고 `nn.Module`을 하위 클래스로 만들고 forward pass를 작성하여 다른 사람이 코드를 빠르게 디버그할 수 있도록 합니다. print 문이나 중단점을 추가할 수 있습니다.
5. 함수 시그니처에는 타입 주석을 사용해야 합니다. 그 외에는 타입 주석보다 변수 이름이 훨씬 읽기 쉽고 이해하기 쉽습니다.

### 토크나이저 개요 [[overview-of-tokenizers]]
 
아직 준비되지 않았습니다 :-( 이 섹션은 곧 추가될 예정입니다!

## 🤗 Transformers에 모델 추가하는 단계별 방법  [[stepbystep-recipe-to-add-a-model-to-transformers]]

각자 모델을 이식하는 방법에 대한 선호가 다르기 때문에 다른 기여자들이 Hugging Face에 모델을 이식하는 방법에 대한 요약을 살펴보는 것이 매우 유용할 수 있습니다. 다음은 모델을 이식하는 방법에 대한 커뮤니티 블로그 게시물 목록입니다:

1. [GPT2 모델 이식하기](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) - [Thomas](https://huggingface.co/thomwolf)
2. [WMT19 MT 모델 이식하기](https://huggingface.co/blog/porting-fsmt) - [Stas](https://huggingface.co/stas)

경험상 모델을 추가할 때 주의해야 할 가장 중요한 사항은 다음과 같습니다:

-  같은 일을 반복하지 마세요! 새로운 🤗 Transformers 모델을 위해 추가할 코드의 대부분은 이미 🤗 Transformers 어딘가에 존재합니다. 이미 존재하는 복사할 수 있는 유사한 모델과 토크나이저를 찾는데 시간을 투자하세요. [grep](https://www.gnu.org/software/grep/)와 [rg](https://github.com/BurntSushi/ripgrep)를 참고하세요. 모델의 토크나이저가 한 모델을 기반으로 하고 모델링 코드가 다른 모델을 기반으로 하는 경우가 존재할 수도 있습니다. 예를 들어 FSMT의 모델링 코드는 BART를 기반으로 하고 FSMT의 토크나이저 코드는 XLM을 기반으로 합니다.
-  이것은 과학적인 도전보다는 공학적인 도전입니다. 논문의 모델의 모든 이론적 측면을 이해하려는 것보다 효율적인 디버깅 환경을 만드는 데 더 많은 시간을 소비해야 합니다.
-  막힐 때 도움을 요청하세요! 모델은 🤗 Transformers의 핵심 구성 요소이므로 Hugging Face의 우리는 당신이 모델을 추가하는 각 단계에서 기꺼이 도움을 줄 준비가 되어 있습니다. 진전이 없다고 느끼면 주저하지 말고 도움을 요청하세요.

다음에서는 모델을 🤗 Transformers로 이식하는 데 가장 유용한 일반적인 절차를 제공하려고 노력합니다.

다음 목록은 모델을 추가하는 데 수행해야 할 모든 작업의 요약이며 To-Do 목록으로 사용할 수 있습니다:

☐ (선택 사항) BrandNewBert의 이론적 측면 이해<br>
☐ Hugging Face 개발 환경 준비<br>
☐ 원본 리포지토리의 디버깅 환경 설정<br>
☐ 원본 리포지토리와 체크포인트를 사용하여 `forward()` pass가 성공적으로 실행되는 스크립트 작성<br>
☐ 🤗 Transformers에 모델 스켈레톤 성공적으로 추가<br>
☐ 원본 체크포인트를 🤗 Transformers 체크포인트로 성공적으로 변환<br>
☐ 🤗 Transformers에서 원본 체크포인트와 동일한 출력을 내주는 `forward()` pass 성공적으로 실행<br>
☐ 🤗 Transformers에서 모델 테스트 완료<br>
☐ 🤗 Transformers에 토크나이저 성공적으로 추가<br>
☐ 종단 간 통합 테스트 실행<br>
☐ 문서 작성 완료<br>
☐ 모델 가중치를 허브에 업로드<br>
☐ Pull request 제출<br>
☐ (선택 사항) 데모 노트북 추가

우선, 일반적으로는 `BrandNewBert`의 이론적인 이해로 시작하는 것을 권장합니다. 그러나 이론적 측면을 직접 이해하는 대신 *직접 해보면서* 모델의 이론적 측면을 이해하는 것을 선호하는 경우 바로 `BrandNewBert` 코드 베이스로 빠져드는 것도 괜찮습니다. 이 옵션은 엔지니어링 기술이 이론적 기술보다 더 뛰어난 경우, `BrandNewBert`의 논문을 이해하는 데 어려움이 있는 경우, 또는 과학적인 논문을 읽는 것보다 프로그래밍에 훨씬 더 흥미 있는 경우에 더 적합할 수 있습니다.

### 1. (선택 사항) BrandNewBert의 이론적 측면 [[1-optional-theoretical-aspects-of-brandnewbert]]

만약 그런 서술적인 작업이 존재한다면, *BrandNewBert*의 논문을 읽어보는 시간을 가져야 합니다. 이해하기 어려운 섹션이 많을 수 있습니다. 그렇더라도 걱정하지 마세요! 목표는 논문의 깊은 이론적 이해가 아니라 *BrandNewBert*를 🤗 Transformers에서 효과적으로 재구현하기 위해 필요한 정보를 추출하는 것입니다. 이를 위해 이론적 측면에 너무 많은 시간을 투자할 필요는 없지만 다음과 같은 실제적인 측면에 집중해야 합니다:

- *BrandNewBert*는 어떤 유형의 모델인가요? BERT와 유사한 인코더 모델인가요? GPT2와 유사한 디코더 모델인가요? BART와 유사한 인코더-디코더 모델인가요? 이들 간의 차이점에 익숙하지 않은 경우[model_summary](model_summary)를 참조하세요.
- *BrandNewBert*의 응용 분야는 무엇인가요? 텍스트 분류인가요? 텍스트 생성인가요? 요약과 같은 Seq2Seq 작업인가요?
- *brand_new_bert*와 BERT/GPT-2/BART의 차이점은 무엇인가요?
- *brand_new_bert*와 가장 유사한 [🤗 Transformers 모델](https://huggingface.co/transformers/#contents)은 무엇인가요?
- 어떤 종류의 토크나이저가 사용되나요? Sentencepiece 토크나이저인가요? Word piece 토크나이저인가요? BERT 또는 BART에 사용되는 동일한 토크나이저인가요?

모델의 아키텍처에 대해 충분히 이해했다는 생각이 든 후, 궁금한 사항이 있으면 Hugging Face 팀에 문의하십시오. 이는 모델의 아키텍처, 어텐션 레이어 등에 관한 질문을 포함할 수 있습니다. Hugging Face의 유지 관리자들은 보통 코드를 검토하는 것에 대해 매우 기뻐하므로 당신을 돕는 일을 매우 환영할 것입니다!

### 2. 개발 환경 설정 [[2-next-prepare-your-environment]]

1. 저장소 페이지에서 "Fork" 버튼을 클릭하여 저장소의 사본을 GitHub 사용자 계정으로 만듭니다.

2. `transformers` fork를 로컬 디스크에 클론하고 베이스 저장소를 원격 저장소로 추가합니다:

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 개발 환경을 설정합니다. 다음 명령을 실행하여 개발 환경을 설정할 수 있습니다:

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

각 운영 체제에 따라 Transformers의 선택적 의존성이 개수가 증가하면 이 명령이 실패할 수 있습니다. 그런 경우에는 작업 중인 딥 러닝 프레임워크 (PyTorch, TensorFlow 및/또는 Flax)을 설치한 후, 다음 명령을 수행하면 됩니다:

```bash
pip install -e ".[quality]"
```

대부분의 경우에는 이것으로 충분합니다. 그런 다음 상위 디렉토리로 돌아갑니다.

```bash
cd ..
```

4. Transformers에 *brand_new_bert*의 PyTorch 버전을 추가하는 것을 권장합니다. PyTorch를 설치하려면 다음 링크의 지침을 따르십시오: https://pytorch.org/get-started/locally/.

**참고:** CUDA를 설치할 필요는 없습니다. 새로운 모델이 CPU에서 작동하도록 만드는 것으로 충분합니다.

5. *brand_new_bert*를 이식하기 위해서는 해당 원본 저장소에 접근할 수 있어야 합니다:

```bash
git clone https://github.com/org_that_created_brand_new_bert_org/brand_new_bert.git
cd brand_new_bert
pip install -e .
```

이제 *brand_new_bert*를 🤗 Transformers로 이식하기 위한 개발 환경을 설정하였습니다.

### 3.-4. 원본 저장소에서 사전 훈련된 체크포인트 실행하기 [[3.-4.-run-a-pretrained-checkpoint-using-the-original-repository]]

먼저, 원본 *brand_new_bert* 저장소에서 작업을 시작합니다. 원본 구현은 보통 "연구용"으로 많이 사용됩니다. 즉, 문서화가 부족하고 코드가 이해하기 어려울 수 있습니다. 그러나 이것이 바로 *brand_new_bert*를 다시 구현하려는 동기가 되어야 합니다. Hugging Face에서의 주요 목표 중 하나는 **거인의 어깨 위에 서는 것**이며, 이는 여기에서 쉽게 해석되어 동작하는 모델을 가져와서 가능한 한 **접근 가능하고 사용자 친화적이며 아름답게** 만드는 것입니다. 이것은 🤗 Transformers에서 모델을 다시 구현하는 가장 중요한 동기입니다 - 새로운 복잡한 NLP 기술을 **모두에게** 접근 가능하게 만드는 것을 목표로 합니다.

따라서 원본 저장소에 대해 자세히 살펴보는 것으로 시작해야 합니다.

원본 저장소에서 공식 사전 훈련된 모델을 성공적으로 실행하는 것은 종종 **가장 어려운** 단계입니다. 우리의 경험에 따르면, 원본 코드 베이스에 익숙해지는 데 시간을 투자하는 것이 매우 중요합니다. 다음을 파악해야 합니다:

- 사전 훈련된 가중치를 어디서 찾을 수 있는지?
- 사전 훈련된 가중치를 해당 모델에로드하는 방법은?
- 모델과 독립적으로 토크나이저를 실행하는 방법은?
- 간단한 forward pass에 필요한 클래스와 함수를 파악하기 위해 forward pass를 한 번 추적해 보세요. 일반적으로 해당 함수들만 다시 구현하면 됩니다.
- 모델의 중요한 구성 요소를 찾을 수 있어야 합니다. 모델 클래스는 어디에 있나요? 모델 하위 클래스(*EncoderModel*, *DecoderModel* 등)가 있나요? self-attention 레이어는 어디에 있나요? self-attention, cross-attention 등 여러 가지 다른 어텐션 레이어가 있나요?
- 원본 환경에서 모델을 디버그할 수 있는 방법은 무엇인가요? *print* 문을 추가해야 하나요? *ipdb*와 같은 대화식 디버거를 사용할 수 있나요? PyCharm과 같은 효율적인 IDE를 사용해 모델을 디버그할 수 있나요?

원본 저장소에서 코드를 이식하는 작업을 시작하기 전에 원본 저장소에서 코드를 **효율적으로** 디버그할 수 있어야 합니다! 또한, 오픈 소스 라이브러리로 작업하고 있다는 것을 기억해야 합니다. 따라서 원본 저장소에서 issue를 열거나 pull request를 열기를 주저하지 마십시오. 이 저장소의 유지 관리자들은 누군가가 자신들의 코드를 살펴본다는 것에 대해 매우 기뻐할 것입니다!

현재 시점에서, 원래 모델을 디버깅하기 위해 어떤 디버깅 환경과 전략을 선호하는지는 당신에게 달렸습니다. 우리는 고가의 GPU 환경을 구축하는 것은 비추천합니다. 대신, 원래 저장소로 들어가서 작업을 시작할 때와 🤗 Transformers 모델의 구현을 시작할 때에도 CPU에서 작업하는 것이 좋습니다. 모델이 이미 🤗 Transformers로 성공적으로 이식되었을 때에만 모델이 GPU에서도 예상대로 작동하는지 확인해야합니다.

일반적으로, 원래 모델을 실행하기 위한 두 가지 가능한 디버깅 환경이 있습니다.

- [Jupyter 노트북](https://jupyter.org/) / [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- 로컬 Python 스크립트

Jupyter 노트북의 장점은 셀 단위로 실행할 수 있다는 것입니다. 이는 논리적인 구성 요소를 더 잘 분리하고 중간 결과를 저장할 수 있으므로 디버깅 사이클이 더 빨라질 수 있습니다. 또한, 노트북은 다른 기여자와 쉽게 공유할 수 있으므로 Hugging Face 팀의 도움을 요청하려는 경우 매우 유용할 수 있습니다. Jupyter 노트북에 익숙하다면 이를 사용하는 것을 강력히 추천합니다.

Jupyter 노트북의 단점은 사용에 익숙하지 않은 경우 새로운 프로그래밍 환경에 적응하는 데 시간을 할애해야 하며, `ipdb`와 같은 알려진 디버깅 도구를 더 이상 사용할 수 없을 수도 있다는 것입니다.

각 코드 베이스에 대해 좋은 첫 번째 단계는 항상 **작은** 사전 훈련된 체크포인트를 로드하고 더미 정수 벡터 입력을 사용하여 단일 forward pass를 재현하는 것입니다. 이와 같은 스크립트는 다음과 같을 수 있습니다(의사 코드로 작성):

```python
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = [0, 4, 5, 2, 3, 7, 9]  # vector of input ids
original_output = model.predict(input_ids)
```

다음으로, 디버깅 전략에 대해 일반적으로 다음과 같은 몇 가지 선택지가 있습니다:

- 원본 모델을 많은 작은 테스트 가능한 구성 요소로 분해하고 각각에 대해 forward pass를 실행하여 검증합니다.
- 원본 모델을 원본 *tokenizer*과 원본 *model*로만 분해하고 해당 부분에 대해 forward pass를 실행한 후 검증을 위해 중간 출력(print 문 또는 중단점)을 사용합니다.

다시 말하지만, 어떤 전략을 선택할지는 당신에게 달려 있습니다. 원본 코드 베이스에 따라 하나 또는 다른 전략이 유리할 수 있습니다.

원본 코드 베이스를 모델의 작은 하위 구성 요소로 분해할 수 있는지 여부, 예를 들어 원본 코드 베이스가 즉시 실행 모드에서 간단히 실행될 수 있는 경우, 그런 경우에는 그 노력이 가치가 있다는 것이 일반적입니다. 초기에 더 어려운 방법을 선택하는 것에는 몇 가지 중요한 장점이 있습니다.

- 원본 모델을 🤗 Transformers 구현과 비교할 때 각 구성 요소가 일치하는지 자동으로 확인할 수 있습니다. 즉, 시각적인 비교(print 문을 통한 비교가 아닌) 대신 🤗 Transformers 구현과 그에 대응하는 원본 구성 요소가 일치하는지 확인할 수 있습니다.
- 전체 모델을 모듈별로, 즉 작은 구성 요소로 분해함으로써 모델을 이식하는 큰 문제를 단순히 개별 구성 요소를 이식하는 작은 문제로 분해할 수 있으므로 작업을 더 잘 구조화할 수 있습니다.
- 모델을 논리적으로 의미 있는 구성 요소로 분리하는 것은 모델의 설계에 대한 더 나은 개요를 얻고 모델을 더 잘 이해하는 데 도움이 됩니다.
- 이러한 구성 요소별 테스트를 통해 코드를 변경하면서 회귀가 발생하지 않도록 보장할 수 있습니다.

[Lysandre의 ELECTRA 통합 검사](https://gist.github.com/LysandreJik/db4c948f6b4483960de5cbac598ad4ed)는 이를 수행하는 좋은 예제입니다.

그러나 원본 코드 베이스가 매우 복잡하거나 중간 구성 요소를 컴파일된 모드에서 실행하는 것만 허용하는 경우, 모델을 테스트 가능한 작은 하위 구성 요소로 분해하는 것이 시간이 많이 소요되거나 불가능할 수도 있습니다. [T5의 MeshTensorFlow](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow) 라이브러리는 매우 복잡하며 모델을 하위 구성 요소로 분해하는 간단한 방법을 제공하지 않습니다. 이러한 라이브러리의 경우, 보통 print 문을 통해 확인합니다.

어떤 전략을 선택하더라도 권장되는 절차는 동일합니다. 먼저 시작 레이어를 디버그하고 마지막 레이어를 마지막에 디버그하는 것이 좋습니다.

다음 순서로 각 레이어의 출력을 검색하는 것이 좋습니다:

1. 모델에 전달된 입력 ID 가져오기
2. 워드 임베딩 가져오기
3. 첫 번째 Transformer 레이어의 입력 가져오기
4. 첫 번째 Transformer 레이어의 출력 가져오기
5. 다음 n-1개의 Transformer 레이어의 출력 가져오기
6. BrandNewBert 모델의 출력 가져오기

입력 ID는 정수 배열로 구성되며, 예를 들어 `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`와 같을 수 있습니다.

다음 레이어의 출력은 종종 다차원 실수 배열로 구성되며, 다음과 같이 나타낼 수 있습니다:

```
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

🤗 Transformers에 추가되는 모든 모델은 통합 테스트를 통과해야 합니다. 즉, 원본 모델과 🤗 Transformers의 재구현 버전이 0.001의 정밀도로 정확히 동일한 출력을 내야 합니다! 동일한 모델이 다른 라이브러리에서 작성되었을 때 라이브러리 프레임워크에 따라 약간 다른 출력을 얻는 것은 정상이므로 1e-3(0.001)의 오차는 허용합니다. 거의 동일한 출력을 내는 것만으로는 충분하지 않으며, 완벽히 일치하는 수준이어야 합니다. 따라서 🤗 Transformers 버전의 중간 출력을 *brand_new_bert*의 원래 구현의 중간 출력과 여러 번 비교해야 합니다. 이 경우 원본 저장소의 **효율적인** 디버깅 환경이 절대적으로 중요합니다. 디버깅 환경을 가능한 한 효율적으로 만드는 몇 가지 조언을 제시합니다.

- 중간 결과를 디버그하는 가장 좋은 방법을 찾으세요. 원본 저장소가 PyTorch로 작성되었다면 원본 모델을 더 작은 하위 구성 요소로 분해하여 중간 값을 검색하는 긴 스크립트를 작성하는 것에 시간을 투자할 가치가 있습니다. 원본 저장소가 Tensorflow 1로 작성되었다면 [tf.print](https://www.tensorflow.org/api_docs/python/tf/print)와 같은 Tensorflow 출력 작업을 사용하여 중간 값을 출력해야 할 수도 있습니다. 원본 저장소가 Jax로 작성되었다면 forward pass를 실행할 때 모델이 **jit 되지 않도록** 해야 합니다. 예를 들어 [이 링크](https://github.com/google/jax/issues/196)를 확인해 보세요.
- 사용 가능한 가장 작은 사전 훈련된 체크포인트를 사용하세요. 체크포인트가 작을수록 디버그 사이클이 더 빨라집니다. 전반적으로 forward pass에 10초 이상이 걸리는 경우 효율적이지 않습니다. 매우 큰 체크포인트만 사용할 수 있는 경우, 새 환경에서 임의로 초기화된 가중치로 더미 모델을 만들고 해당 가중치를 🤗 Transformers 버전과 비교하기 위해 저장하는 것이 더 의미가 있을 수 있습니다.
- 디버깅 설정에서 가장 쉽게 forward pass를 호출하는 방법을 사용하세요. 원본 저장소에서 **단일** forward pass만 호출하는 함수를 찾는 것이 이상적입니다. 이 함수는 일반적으로 `predict`, `evaluate`, `forward`, `__call__`과 같이 호출됩니다. `autoregressive_sample`과 같은 텍스트 생성에서 `forward`를 여러 번 호출하여 텍스트를 생성하는 등의 작업을 수행하는 함수를 디버그하고 싶지 않을 것입니다.
- 토큰화 과정을 모델의 *forward* pass와 분리하려고 노력하세요. 원본 저장소에서 입력 문자열을 입력해야 하는 예제가 있는 경우, 입력 문자열이 입력 ID로 변경되는 순간을 찾아서 시작하세요. 이 경우 직접 ID를 입력할 수 있도록 작은 스크립트를 작성하거나 원본 코드를 수정해야 할 수도 있습니다.
- 디버깅 설정에서 모델이 훈련 모드가 아니라는 것을 확인하세요. 훈련 모드에서는 모델의 여러 드롭아웃 레이어 때문에 무작위 출력이 생성될 수 있습니다. 디버깅 환경에서 forward pass가 **결정론적**이도록 해야 합니다. 또는 동일한 프레임워크에 있는 경우 *transformers.utils.set_seed*를 사용하세요.

다음 섹션에서는 *brand_new_bert*에 대해 이 작업을 수행하는 데 더 구체적인 세부 사항/팁을 제공합니다.

### 5.-14. 🤗 Transformers에 BrandNewBert를 이식하기 [[5.-14.-port-brandnewbert-to-transformers]]

이제, 마침내 🤗 Transformers에 새로운 코드를 추가할 수 있습니다. 🤗 Transformers 포크의 클론으로 이동하세요:

```bash
cd transformers
```

다음과 같이 이미 존재하는 모델의 모델 아키텍처와 정확히 일치하는 모델을 추가하는 특별한 경우에는 [이 섹션](#write-a-conversion-script)에 설명된대로 변환 스크립트만 추가하면 됩니다. 이 경우에는 이미 존재하는 모델의 전체 모델 아키텍처를 그대로 재사용할 수 있습니다.

그렇지 않으면 새 모델 생성을 시작하겠습니다. 다음 스크립트를 사용하여 다음에서 시작하는 모델을 추가하는 것이 좋습니다.
기존 모델:

```bash
transformers-cli add-new-model-like
```

모델의 기본 정보를 입력하는 설문지가 표시됩니다.

**huggingface/transformers 메인 저장소에 Pull Request 열기**

자동으로 생성된 코드를 수정하기 전에, 지금은 "작업 진행 중 (WIP)" 풀 리퀘스트를 열기 위한 시기입니다. 예를 들어, 🤗 Transformers에 "*brand_new_bert* 추가"라는 제목의 "[WIP] Add *brand_new_bert*" 풀 리퀘스트를 엽니다. 이렇게 하면 당신과 Hugging Face 팀이 🤗 Transformers에 모델을 통합하는 작업을 함께할 수 있습니다.

다음을 수행해야 합니다:

1. 메인 브랜치에서 작업을 잘 설명하는 이름으로 브랜치 생성

```bash
git checkout -b add_brand_new_bert
```

2. 자동으로 생성된 코드 커밋

```bash
git add .
git commit
```

3. 현재 메인을 가져오고 리베이스

```bash
git fetch upstream
git rebase upstream/main
```

4. 변경 사항을 계정에 푸시

```bash
git push -u origin a-descriptive-name-for-my-changes
```

5. 만족스럽다면, GitHub에서 자신의 포크한 웹 페이지로 이동합니다. "Pull request"를 클릭합니다. Hugging Face 팀의 일부 멤버의 GitHub 핸들을 리뷰어로 추가하여 Hugging Face 팀이 앞으로의 변경 사항에 대해 알림을 받을 수 있도록 합니다.

6. GitHub 풀 리퀘스트 웹 페이지 오른쪽에 있는 "Convert to draft"를 클릭하여 PR을 초안으로 변경합니다.

다음으로, 어떤 진전을 이루었다면 작업을 커밋하고 계정에 푸시하여 풀 리퀘스트에 표시되도록 해야 합니다. 또한, 다음과 같이 현재 메인과 작업을 업데이트해야 합니다:

```bash
git fetch upstream
git merge upstream/main
```

일반적으로, 모델 또는 구현에 관한 모든 질문은 자신의 PR에서 해야 하며, PR에서 토론되고 해결되어야 합니다. 이렇게 하면 Hugging Face 팀이 새로운 코드를 커밋하거나 질문을 할 때 항상 알림을 받을 수 있습니다. Hugging Face 팀에게 문제 또는 질문을 효율적으로 이해할 수 있도록 추가한 코드를 명시하는 것이 도움이 될 때가 많습니다.

이를 위해, 변경 사항을 모두 볼 수 있는 "Files changed" 탭으로 이동하여 질문하고자 하는 줄로 이동한 다음 "+" 기호를 클릭하여 코멘트를 추가할 수 있습니다. 질문이나 문제가 해결되면, 생성된 코멘트의 "Resolve" 버튼을 클릭할 수 있습니다.

마찬가지로, Hugging Face 팀은 코드를 리뷰할 때 코멘트를 남길 것입니다. 우리는 PR에서 대부분의 질문을 GitHub에서 묻는 것을 권장합니다. 공개에 크게 도움이 되지 않는 매우 일반적인 질문의 경우, Slack이나 이메일을 통해 Hugging Face 팀에게 문의할 수 있습니다.

**5. brand_new_bert에 대해 생성된 모델 코드를 적용하기**

먼저, 우리는 모델 자체에만 초점을 맞추고 토크나이저에 대해서는 신경 쓰지 않을 것입니다. 모든 관련 코드는 다음의 생성된 파일에서 찾을 수 있습니다: `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` 및 `src/transformers/models/brand_new_bert/configuration_brand_new_bert.py`.

이제 마침내 코딩을 시작할 수 있습니다 :). `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`의 생성된 코드는 인코더 전용 모델인 경우 BERT와 동일한 아키텍처를 가지거나, 인코더-디코더 모델인 경우 BART와 동일한 아키텍처를 가질 것입니다. 이 시점에서, 모델의 이론적 측면에 대해 배운 내용을 다시 상기해야 합니다: *모델이 BERT 또는 BART와 어떻게 다른가요?*. 자주 변경해야 하는 것은 *self-attention* 레이어, 정규화 레이어의 순서 등을 변경하는 것입니다. 다시 말하지만, 자신의 모델을 구현하는 데 도움이 되도록 Transformers에서 이미 존재하는 모델의 유사한 아키텍처를 살펴보는 것이 유용할 수 있습니다.

**참고로** 이 시점에서, 코드가 완전히 정확하거나 깨끗하다고 확신할 필요는 없습니다. 오히려 처음에는 원본 코드의 첫 번째 *불완전하고* 복사된 버전을 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`에 추가하는 것이 좋습니다. 필요한 모든 코드가 추가될 때까지 이러한 작업을 진행한 후, 다음 섹션에서 설명한 변환 스크립트를 사용하여 코드를 점진적으로 개선하고 수정하는 것이 훨씬 효율적입니다. 이 시점에서 작동해야 하는 유일한 것은 다음 명령이 작동하는 것입니다:

```python
from transformers import BrandNewBertModel, BrandNewBertConfig

model = BrandNewBertModel(BrandNewBertConfig())
```

위의 명령은 `BrandNewBertConfig()`에 정의된 기본 매개변수에 따라 무작위 가중치로 모델을 생성하며, 이로써 모든 구성 요소의 `init()` 메서드가 작동함을 보장합니다.

모든 무작위 초기화는 `BrandnewBertPreTrainedModel` 클래스의 `_init_weights` 메서드에서 수행되어야 합니다. 이 메서드는 구성 설정 변수에 따라 모든 리프 모듈을 초기화해야 합니다. BERT의 `_init_weights` 메서드 예제는 다음과 같습니다:

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

몇 가지 모듈에 대해 특별한 초기화가 필요한 경우 사용자 정의 방식을 사용할 수도 있습니다. 예를 들어, `Wav2Vec2ForPreTraining`에서 마지막 두 개의 선형 레이어는 일반적인 PyTorch `nn.Linear`의 초기화를 가져야 하지만, 다른 모든 레이어는 위와 같은 초기화를 사용해야 합니다. 이는 다음과 같이 코드화됩니다:

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, Wav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
        module.project_hid._is_hf_initialized = True
        module.project_q._is_hf_initialized = True
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
```

`_is_hf_initialized` 플래그는 서브모듈을 한 번만 초기화하도록 내부적으로 사용됩니다. `module.project_q` 및 `module.project_hid`에 대해 `True`로 설정함으로써, 우리가 수행한 사용자 정의 초기화가 이후에 덮어쓰이지 않도록 합니다. 즉, `_init_weights` 함수가 이들에게 적용되지 않습니다.

**6. 변환 스크립트 작성하기**

다음으로, 디버그에 사용한 체크포인트를 기존 저장소에서 만든 🤗 Transformers 구현과 호환되는 체크포인트로 변환할 수 있는 변환 스크립트를 작성해야 합니다. 변환 스크립트를 처음부터 작성하는 것보다는 *brand_new_bert*와 동일한 프레임워크로 작성된 유사한 모델을 변환한 기존 변환 스크립트를 찾아보는 것이 좋습니다. 일반적으로 기존 변환 스크립트를 복사하여 사용 사례에 맞게 약간 수정하는 것으로 충분합니다. 모델에 대해 유사한 기존 변환 스크립트를 어디에서 찾을 수 있는지 Hugging Face 팀에게 문의하는 것을 망설이지 마세요.

- TensorFlow에서 PyTorch로 모델을 이전하는 경우, 좋은 참고 자료로 BERT의 변환 스크립트 [여기](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91)를 참조할 수 있습니다.
- PyTorch에서 PyTorch로 모델을 이전하는 경우, 좋은 참고 자료로 BART의 변환 스크립트 [여기](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py)를 참조할 수 있습니다.

다음에서는 PyTorch 모델이 레이어 가중치를 저장하고 레이어 이름을 정의하는 방법에 대해 간단히 설명하겠습니다. PyTorch에서 레이어의 이름은 레이어에 지정한 클래스 속성의 이름으로 정의됩니다. 다음과 같이 PyTorch에서 `SimpleModel`이라는 더미 모델을 정의해 봅시다:

```python
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        self.intermediate = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)
```

이제 이 모델 정의의 인스턴스를 생성할 수 있으며 `dense`, `intermediate`, `layer_norm` 등의 가중치가 랜덤하게 할당됩니다. 모델을 출력하여 아키텍처를 확인할 수 있습니다.

```python
model = SimpleModel()

print(model)
```

이는 다음과 같이 출력됩니다:

```
SimpleModel(
  (dense): Linear(in_features=10, out_features=10, bias=True)
  (intermediate): Linear(in_features=10, out_features=10, bias=True)
  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)
)
```

우리는 레이어의 이름이 PyTorch에서 클래스 속성의 이름으로 정의되어 있는 것을 볼 수 있습니다. 특정 레이어의 가중치 값을 출력하여 확인할 수 있습니다:

```python
print(model.dense.weight.data)
```

가중치가 무작위로 초기화되었음을 확인할 수 있습니다.

```
tensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,
         -0.2077,  0.2157],
        [ 0.1044,  0.0201,  0.0990,  0.2482,  0.3116,  0.2509,  0.2866, -0.2190,
          0.2166, -0.0212],
        [-0.2000,  0.1107, -0.1999, -0.3119,  0.1559,  0.0993,  0.1776, -0.1950,
         -0.1023, -0.0447],
        [-0.0888, -0.1092,  0.2281,  0.0336,  0.1817, -0.0115,  0.2096,  0.1415,
         -0.1876, -0.2467],
        [ 0.2208, -0.2352, -0.1426, -0.2636, -0.2889, -0.2061, -0.2849, -0.0465,
          0.2577,  0.0402],
        [ 0.1502,  0.2465,  0.2566,  0.0693,  0.2352, -0.0530,  0.1859, -0.0604,
          0.2132,  0.1680],
        [ 0.1733, -0.2407, -0.1721,  0.1484,  0.0358, -0.0633, -0.0721, -0.0090,
          0.2707, -0.2509],
        [-0.1173,  0.1561,  0.2945,  0.0595, -0.1996,  0.2988, -0.0802,  0.0407,
          0.1829, -0.1568],
        [-0.1164, -0.2228, -0.0403,  0.0428,  0.1339,  0.0047,  0.1967,  0.2923,
          0.0333, -0.0536],
        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,
          0.2220,  0.2358]]).
```

변환 스크립트에서는 이러한 무작위로 초기화된 가중치를 체크포인트의 해당 레이어의 정확한 가중치로 채워야 합니다. 예를 들면 다음과 같습니다:

```python
# retrieve matching layer weights, e.g. by
# recursive algorithm
layer_name = "dense"
pretrained_weight = array_of_dense_layer

model_pointer = getattr(model, "dense")

model_pointer.weight.data = torch.from_numpy(pretrained_weight)
```

이렇게 하면 PyTorch 모델의 무작위로 초기화된 각 가중치와 해당 체크포인트 가중치가 **모양과 이름** 모두에서 정확히 일치하는지 확인해야 합니다. 이를 위해 모양에 대한 assert 문을 추가하고 체크포인트 가중치의 이름을 출력해야 합니다. 예를 들어 다음과 같은 문장을 추가해야 합니다:

```python
assert (
    model_pointer.weight.shape == pretrained_weight.shape
), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"
```

또한 두 가중치의 이름을 출력하여 일치하는지 확인해야 합니다. *예시*:

```python
logger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")
```

모양 또는 이름이 일치하지 않는 경우, 랜덤으로 초기화된 레이어에 잘못된 체크포인트 가중치를 할당한 것으로 추측됩니다.

잘못된 모양은 `BrandNewBertConfig()`의 구성 매개변수 설정이 변환하려는 체크포인트에 사용된 설정과 정확히 일치하지 않기 때문일 가능성이 가장 큽니다. 그러나 PyTorch의 레이어 구현 자체에서 가중치를 전치해야 할 수도 있습니다.

마지막으로, **모든** 필요한 가중치가 초기화되었는지 확인하고 초기화에 사용되지 않은 모든 체크포인트 가중치를 출력하여 모델이 올바르게 변환되었는지 확인해야 합니다. 잘못된 모양 문장이나 잘못된 이름 할당으로 인해 변환 시도가 실패하는 것은 완전히 정상입니다. 이는 `BrandNewBertConfig()`에서 잘못된 매개변수를 사용하거나 🤗 Transformers 구현에서 잘못된 아키텍처, 🤗 Transformers 구현의 구성 요소 중 하나의 `init()` 함수에 버그가 있는 경우이거나 체크포인트 가중치 중 하나를 전치해야 하는 경우일 가능성이 가장 높습니다.

이 단계는 이전 단계와 함께 반복되어야 하며 모든 체크포인트의 가중치가 Transformers 모델에 올바르게 로드되었을 때까지 계속되어야 합니다. 🤗 Transformers 구현에 체크포인트를 올바르게 로드한 후에는 `/path/to/converted/checkpoint/folder`와 같은 원하는 폴더에 모델을 저장할 수 있어야 합니다. 해당 폴더에는 `pytorch_model.bin` 파일과 `config.json` 파일이 모두 포함되어야 합니다.

```python
model.save_pretrained("/path/to/converted/checkpoint/folder")
```

**7. 순방향 패스 구현하기**

🤗 Transformers 구현에 사전 훈련된 가중치를 정확하게 로드한 후에는 순방향 패스가 올바르게 구현되었는지 확인해야 합니다. [원본 저장소에 익숙해지기](#3-4-run-a-pretrained-checkpoint-using-the-original-repository)에서 이미 원본 저장소를 사용하여 모델의 순방향 패스를 실행하는 스크립트를 만들었습니다. 이제 원본 대신 🤗 Transformers 구현을 사용하는 유사한 스크립트를 작성해야 합니다. 다음과 같이 작성되어야 합니다:

```python
model = BrandNewBertModel.from_pretrained("/path/to/converted/checkpoint/folder")
input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
output = model(input_ids).last_hidden_states
```

🤗 Transformers 구현과 원본 모델 구현이 처음부터 정확히 동일한 출력을 제공하지 않거나 순방향 패스에서 오류가 발생할 가능성이 매우 높습니다. 실망하지 마세요. 예상된 일입니다! 먼저, 순방향 패스에서 오류가 발생하지 않도록 해야 합니다. 종종 잘못된 차원이 사용되어 *차원 불일치* 오류가 발생하거나 잘못된 데이터 유형 개체가 사용되는 경우가 있습니다. 예를 들면 `torch.long` 대신에 `torch.float32`가 사용된 경우입니다. 해결할 수 없는 오류가 발생하면 Hugging Face 팀에 도움을 요청하는 것이 좋습니다.

🤗 Transformers 구현이 올바르게 작동하는지 확인하는 마지막 단계는 출력이 `1e-3`의 정밀도로 동일한지 확인하는 것입니다. 먼저, 출력 모양이 동일하도록 보장해야 합니다. 즉, 🤗 Transformers 구현 스크립트와 원본 구현 사이에서 `outputs.shape`는 동일한 값을 반환해야 합니다. 그 다음으로, 출력 값이 동일하도록 해야 합니다. 이는 새로운 모델을 추가할 때 가장 어려운 부분 중 하나입니다. 출력이 동일하지 않은 일반적인 실수 사례는 다음과 같습니다:

- 일부 레이어가 추가되지 않았습니다. 즉, *활성화* 레이어가 추가되지 않았거나 잔차 연결이 빠졌습니다.
- 단어 임베딩 행렬이 연결되지 않았습니다.
- 잘못된 위치 임베딩이 사용되었습니다. 원본 구현에서는 오프셋을 사용합니다.
- 순방향 패스 중에 Dropout이 적용되었습니다. 이를 수정하려면 *model.training이 False*인지 확인하고 순방향 패스 중에 Dropout 레이어가 잘못 활성화되지 않도록 하세요. 즉, [PyTorch의 기능적 Dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout)에 *self.training*을 전달하세요.

문제를 해결하는 가장 좋은 방법은 일반적으로 원본 구현과 🤗 Transformers 구현의 순방향 패스를 나란히 놓고 차이점이 있는지 확인하는 것입니다. 이상적으로는 순방향 패스의 중간 출력을 디버그/출력하여 원본 구현과 🤗 Transformers 구현의 정확한 위치를 찾을 수 있어야 합니다. 먼저, 두 스크립트의 하드코딩된 `input_ids`가 동일한지 확인하세요. 다음으로, `input_ids`의 첫 번째 변환의 출력(일반적으로 단어 임베딩)이 동일한지 확인하세요. 그런 다음 네트워크의 가장 마지막 레이어까지 진행해보세요. 어느 시점에서 두 구현 사이에 차이가 있는 것을 알게 되는데, 이는 🤗 Transformers 구현의 버그 위치를 가리킬 것입니다. 저희 경험상으로는 원본 구현과 🤗 Transformers 구현 모두에서 동일한 위치에 많은 출력 문을 추가하고 이들의 중간 표현에 대해 동일한 값을 보이는 출력 문을 연속적으로 제거하는 것이 간단하고 효과적인 방법입니다.

`torch.allclose(original_output, output, atol=1e-3)`로 출력을 확인하여 두 구현이 동일한 출력을 하는 것을 확신한다면, 가장 어려운 부분은 끝났습니다! 축하드립니다. 남은 작업은 쉬운 일이 될 것입니다 😊.

**8. 필요한 모든 모델 테스트 추가하기**

이 시점에서 새로운 모델을 성공적으로 추가했습니다. 그러나 해당 모델이 요구되는 디자인에 완전히 부합하지 않을 수도 있습니다. 🤗 Transformers와 완벽하게 호환되는 구현인지 확인하기 위해 모든 일반 테스트를 통과해야 합니다. Cookiecutter는 아마도 모델을 위한 테스트 파일을 자동으로 추가했을 것입니다. 아마도 `tests/models/brand_new_bert/test_modeling_brand_new_bert.py`와 같은 경로에 위치할 것입니다. 이 테스트 파일을 실행하여 일반 테스트가 모두 통과하는지 확인하세요.

```bash
pytest tests/models/brand_new_bert/test_modeling_brand_new_bert.py
```

모든 일반 테스트를 수정한 후, 이제 수행한 작업을 충분히 테스트하여 다음 사항을 보장해야 합니다.

- a) 커뮤니티가 *brand_new_bert*의 특정 테스트를 살펴봄으로써 작업을 쉽게 이해할 수 있도록 함
- b) 모델에 대한 향후 변경 사항이 모델의 중요한 기능을 손상시키지 않도록 함

먼저 통합 테스트를 추가해야 합니다. 이러한 통합 테스트는 이전에 모델을 🤗 Transformers로 구현하기 위해 사용한 디버깅 스크립트와 동일한 작업을 수행합니다. Cookiecutter에 이미 이러한 모델 테스트의 템플릿인 `BrandNewBertModelIntegrationTests`가 추가되어 있으며, 여러분이 작성해야 할 내용으로만 채워 넣으면 됩니다. 이러한 테스트가 통과하는지 확인하려면 다음을 실행하세요.

```bash
RUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests
```

<Tip>

Windows를 사용하는 경우 `RUN_SLOW=1`을 `SET RUN_SLOW=1`로 바꿔야 합니다.

</Tip>

둘째로, *brand_new_bert*에 특화된 모든 기능도 별도의 테스트에서 추가로 테스트해야 합니다. 이 부분은 종종 잊히는데, 두 가지 측면에서 굉장히 유용합니다.

- *brand_new_bert*의 특수 기능이 어떻게 작동해야 하는지 보여줌으로써 커뮤니티에게 모델 추가 과정에서 습득한 지식을 전달하는 데 도움이 됩니다.
- 향후 기여자는 이러한 특수 테스트를 실행하여 모델에 대한 변경 사항을 빠르게 테스트할 수 있습니다.


**9. 토크나이저 구현하기**

다음으로, *brand_new_bert*의 토크나이저를 추가해야 합니다. 보통 토크나이저는 🤗 Transformers의 기존 토크나이저와 동일하거나 매우 유사합니다.

토크나이저가 올바르게 작동하는지 확인하기 위해 먼저 원본 리포지토리에서 문자열을 입력하고 `input_ids`를 반환하는 스크립트를 생성하는 것이 좋습니다. 다음과 같은 유사한 스크립트일 수 있습니다 (의사 코드로 작성):

```python
input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = model.tokenize(input_str)
```

원본 리포지토리를 자세히 살펴보고 올바른 토크나이저 함수를 찾거나, 복제본에서 변경 사항을 적용하여 `input_ids`만 출력하도록 해야 합니다. 원본 리포지토리를 사용하는 기능적인 토큰화 스크립트를 작성한 후, 🤗 Transformers의 유사한 스크립트를 생성해야 합니다. 다음과 같이 작성되어야 합니다:

```python
from transformers import BrandNewBertTokenizer

input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."

tokenizer = BrandNewBertTokenizer.from_pretrained("/path/to/tokenizer/folder/")

input_ids = tokenizer(input_str).input_ids
```

두 개의 `input_ids`가 동일한 값을 반환할 때, 마지막 단계로 토크나이저 테스트 파일도 추가해야 합니다.

*brand_new_bert*의 모델링 테스트 파일과 유사하게, *brand_new_bert*의 토크나이제이션 테스트 파일에는 몇 가지 하드코딩된 통합 테스트가 포함되어야 합니다.

**10. 종단 간 통합 테스트 실행**

토크나이저를 추가한 후에는 모델과 토크나이저를 사용하여 몇 가지 종단 간 통합 테스트를 추가해야 합니다. `tests/models/brand_new_bert/test_modeling_brand_new_bert.py`에 추가해주세요. 이러한 테스트는 🤗 Transformers 구현이 예상대로 작동하는지를 의미 있는 text-to-text 예시로 보여줘야 합니다. 그 예시로는 *예를 들어* source-to-target 번역 쌍, article-to-summary 쌍, question-to-answer 쌍 등이 포함될 수 있습니다. 불러온 체크포인트 중 어느 것도 다운스트림 작업에서 미세 조정되지 않았다면, 모델 테스트만으로 충분합니다. 모델이 완전히 기능을 갖추었는지 확인하기 위해 마지막 단계로 GPU에서 모든 테스트를 실행하는 것이 좋습니다. 모델의 내부 텐서의 일부에 `.to(self.device)` 문을 추가하는 것을 잊었을 수 있으며, 이 경우 테스트에서 오류로 표시됩니다. GPU에 액세스할 수 없는 경우, Hugging Face 팀이 테스트를 대신 실행할 수 있습니다.

**11. 기술문서 추가**

이제 *brand_new_bert*에 필요한 모든 기능이 추가되었습니다. 거의 끝났습니다! 추가해야 할 것은 멋진 기술문서과 기술문서 페이지입니다. Cookiecutter가 `docs/source/model_doc/brand_new_bert.md`라는 템플릿 파일을 추가해줬을 것입니다. 이 페이지를 사용하기 전에 모델을 사용하는 사용자들은 일반적으로 이 페이지를 먼저 확인합니다. 따라서 문서는 이해하기 쉽고 간결해야 합니다. 모델을 사용하는 방법을 보여주기 위해 *팁*을 추가하는 것이 커뮤니티에 매우 유용합니다. 독스트링에 관련하여 Hugging Face 팀에 문의하는 것을 주저하지 마세요.

다음으로, `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`에 추가된 독스트링이 올바르며 필요한 모든 입력 및 출력을 포함하도록 확인하세요. [여기](writing-documentation)에서 우리의 문서 작성 가이드와 독스트링 형식에 대한 상세 가이드가 있습니다. 문서는 일반적으로 커뮤니티와 모델의 첫 번째 접점이기 때문에, 문서는 적어도 코드만큼의 주의를 기울여야 합니다.

**코드 리팩토링**

좋아요, 이제 *brand_new_bert*를 위한 모든 필요한 코드를 추가했습니다. 이 시점에서 다음을 실행하여 잠재적으로 잘못된 코드 스타일을 수정해야 합니다:

그리고 코딩 스타일이 품질 점검을 통과하는지 확인하기 위해 다음을 실행하고 확인해야 합니다:

```bash
make style
```

🤗 Transformers에는 여전히 실패할 수 있는 몇 가지 매우 엄격한 디자인 테스트가 있습니다. 이는 독스트링에 누락된 정보나 잘못된 명명 때문에 종종 발생합니다. 여기서 막히면 Hugging Face 팀이 도움을 줄 것입니다.

```bash
make quality
```

마지막으로, 코드가 정확히 작동하는 것을 확인한 후에는 항상 코드를 리팩토링하는 것이 좋은 생각입니다. 모든 테스트가 통과된 지금은 추가한 코드를 다시 검토하고 리팩토링하는 좋은 시기입니다.

이제 코딩 부분을 완료했습니다. 축하합니다! 🎉 멋져요! 😎

**12. 모델을 모델 허브에 업로드하세요**

이 마지막 파트에서는 모든 체크포인트를 변환하여 모델 허브에 업로드하고 각 업로드된 모델 체크포인트에 대한 모델 카드를 추가해야 합니다. [Model sharing and uploading Page](model_sharing)를 읽고 허브 기능에 익숙해지세요. *brand_new_bert*의 저자 조직 아래에 모델을 업로드할 수 있는 필요한 액세스 권한을 얻기 위해 Hugging Face 팀과 협업해야 합니다. `transformers`의 모든 모델에 있는 `push_to_hub` 메서드는 체크포인트를 허브에 빠르고 효율적으로 업로드하는 방법입니다. 아래에 작은 코드 조각이 붙여져 있습니다:

각 체크포인트에 적합한 모델 카드를 만드는 데 시간을 할애하는 것은 가치가 있습니다. 모델 카드는 체크포인트의 특성을 강조해야 합니다. *예를 들어* 이 체크포인트는 어떤 데이터셋에서 사전 훈련/세부 훈련되었는지? 이 모델은 어떤 하위 작업에서 사용해야 하는지? 그리고 모델을 올바르게 사용하는 방법에 대한 몇 가지 코드도 포함해야 합니다.

```python
brand_new_bert.push_to_hub("brand_new_bert")
# Uncomment the following line to push to an organization.
# brand_new_bert.push_to_hub("<organization>/brand_new_bert")
```

**13. (선택 사항) 노트북 추가**

*brand_new_bert*를 다운스트림 작업에서 추론 또는 미세 조정에 사용하는 방법을 자세히 보여주는 노트북을 추가하는 것이 매우 유용합니다. 이것은 PR을 병합하는 데 필수적이지는 않지만 커뮤니티에 매우 유용합니다.

**14. 완료된 PR 제출**

이제 프로그래밍을 마쳤으며, 마지막 단계로 PR을 메인 브랜치에 병합해야 합니다. 보통 Hugging Face 팀은 이미 여기까지 도움을 주었을 것입니다. 그러나 PR에 멋진 설명을 추가하고 리뷰어에게 특정 디자인 선택 사항을 강조하려면 완료된 PR에 약간의 설명을 추가하는 시간을 할애하는 것이 가치가 있습니다.

### 작업물을 공유하세요!! [[share-your-work]]

이제 커뮤니티에서 작업물을 인정받을 시간입니다! 모델 추가 작업을 완료하는 것은 Transformers와 전체 NLP 커뮤니티에 큰 기여입니다. 당신의 코드와 이식된 사전 훈련된 모델은 수백, 심지어 수천 명의 개발자와 연구원에 의해 확실히 사용될 것입니다. 당신의 작업에 자랑스러워해야 하며 이를 커뮤니티와 공유해야 합니다.

**당신은 커뮤니티 내 모든 사람들에게 매우 쉽게 접근 가능한 또 다른 모델을 만들었습니다! 🤯**
