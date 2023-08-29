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
-->

# 🤗 트랜스포머에 기여하기 [[contribute-to-transformers]]

누구든 기여할 수 있으며, 우리는 모든 사람의 기여를 소중히 생각합니다. 코드 기여는 커뮤니티를 돕는 유일한 방법이 아닙니다. 질문에 답하거나 다른 사람을 도와 문서를 개선하는 것도 매우 가치 있습니다.

또한 당신이 알려주시면 우리에게 큰 도움이 됩니다! 이 멋진 프로젝트들이 가능하게 한 라이브러리에 대해 블로그 글에서 언급하거나, 도움이 되었을 때마다 Twitter에서 소개해 주시거나, 단순히 저장소에 ⭐️ 를 표시하여 감사의 뜻을 표현해주세요.

기여하는 방식에 상관없이 우리의 [행동 규칙](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md)을 숙지하고 존중해주시기 바랍니다.

**이 안내서는 멋진 [scikit-learn 기여 안내서](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md)에서 큰 영감을 받았습니다.**

## 기여하는 방법들 [[ways-to-contribute]]

여러 가지 방법으로 🤗 트랜스포머에 기여할 수 있습니다:

* 기존 코드의 미해결된 문제를 수정합니다.
* 버그 또는 원하는 새로운 기능과 관련된 이슈를 제출합니다.
* 새로운 모델을 구현합니다.
* 예제나 문서에 기여합니다.

어디서부터 시작할지 모르겠다면, [처음으로 좋은 이슈](https://github.com/huggingface/transformers/contribute) 목록을 확인해보세요. 이 목록은 초보자도 참여하기 쉬운 오픈 이슈 목록을 제공하며, 해당 이슈에서 작업하고 싶다는 의사를 댓글로 달아주시면 됩니다.

더 조금 도전적인 작업을 원한다면, [두 번째로 좋은 이슈](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue) 목록도 확인해보세요. 일반적으로, 당신이 무엇을 하고 있는지 알고 있다고 생각한다면, 도전해보세요. 우리는 당신이 달성하는 데 도움을 드릴 것입니다! 🚀

> 모든 기여는 커뮤니티에게 동등한 가치가 있습니다. 🥰

## 미해결된 문제 수정하기 [[fixing-outstanding-issues]]

기존 코드에 문제점을 발견하고 해결책이 떠오른 경우, 언제든지 [기여를 시작](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#create-a-pull-request)하고 풀 리퀘스트를 열어주세요!

## 버그 관련 이슈나 기능 요청 제출하기 [[submitting-a-bugrelated-issue-or-feature-request]]

버그 관련 이슈나 새로운 기능 요청을 제출할 때는 다음 가이드라인을 최대한 따르시기 바랍니다. 이렇게 하면 우리가 빠르고 좋은 피드백과 함께 빠르게 답변해 드릴 수 있습니다.

### 버그를 발견하셨나요? [[did-you-find-a-bug]]

🤗 트랜스포머 라이브러리는 사용자들이 겪는 문제들을 보고해주는 사용자들 덕분에 견고하고 신뢰할 수 있습니다.

이슈를 보고하기 전에, 버그가 이미 [보고되지 않았는지](https://github.com/huggingface/transformers/contribute) 확인해 주시기를 부탁드립니다 (GitHub의 이슈 탭 아래의 검색 바를 사용하세요). 당신의 이슈는 라이브러리 자체의 버그와 관련되어야 하며, 당신의 코드와 관련된 것이 아니어야 합니다. 버그가 코드인지 라이브러리인지 확신이 없다면, 먼저 [포럼](https://discuss.huggingface.co/)에서 물어보세요. 이는 라이브러리와 관련된 문제를 해결할 때 우리가 일반적인 질문보다 빠르게 대응할 수 있도록 도와줍니다.

버그가 이미 보고되지 않았다는 것을 확인한 후에, 다음 정보를 이슈에 포함해 주세요. 그러면 우리가 빠르게 해결할 수 있습니다:

* 사용 중인 **운영체제 종류와 버전**, 그리고 **Python**, **PyTorch** 및 **TensorFlow** 버전(해당되는 경우).
* 버그를 30초 이내로 재현할 수 있는 간단하고 자체 포함된 코드 스니펫.
* 예외가 발생한 경우 *전체* 트레이스백.
* 도움이 될 것으로 생각되는 스크린샷과 같은 추가 정보를 첨부해 주세요.

운영체제와 소프트웨어 버전을 자동으로 가져오려면 다음 명령을 실행하세요:

```bash
transformers-cli env
```

저장소 루트에서도 같은 명령을 실행할 수 있습니다:

```bash
python src/transformers/commands/transformers_cli.py env
```


### 새로운 기능을 원하시나요? [[do-you-want-a-new-feature]]

🤗 트랜스포머에서 보고 싶은 새로운 기능이 있다면, 이슈를 열어서 다음을 설명해 주세요:

1. 이 기능의 *동기*는 무엇인가요? 이 라이브러리와 관련된 문제나 당신이 프로젝트에 필요한 기능인가요? 당신이 작업한 내용 중 라이브러리 커뮤니티에 도움이 될 수 있는 기능인가요?

   무엇이든지 듣고 싶습니다!

2. 가능한 한 자세하게 요청하는 기능을 설명해 주세요. 더 많은 정보를 제공할수록 더 나은 도움을 드릴 수 있습니다.
3. 해당 기능 사용법을 보여주는 *코드 스니펫*을 제공해 주세요.
4. 기능이 논문과 관련된 경우 링크를 포함해 주세요.

당신이 잘 쓴 이슈라면 이미 80% 정도의 작업이 완료된 것입니다. 

당신이 이슈를 작성하는 시점에서 [템플릿](https://github.com/huggingface/transformers/tree/main/templates)을 추가해 놓았습니다.

## 새로운 모델을 구현하고 싶으신가요? [[do-you-want-to-implement-a-new-model]]

새로운 모델들은 계속해서 출시되며, 만약 새로운 모델을 구현하고 싶다면, 다음 정보를 제공해 주세요.

* 모델에 대한 간단한 설명과 논문 링크.
* 구현이 공개되어 있다면 구현 링크.
* 모델 가중치가 사용 가능하다면 가중치 링크.

만약 모델을 직접 기여하려는 의사가 있다면, 알려주시면 우리가 당신을 도와 🤗 트랜스포머에 추가할 수 있도록 도와드리겠습니다!

새로운 모델을 추가하는 방법에 대한 [상세 안내서와 템플릿](https://github.com/huggingface/transformers/tree/main/templates)을 제공하였으며, [🤗 트랜스포머에 모델 추가하는 방법](https://huggingface.co/docs/transformers/add_new_model)에 대한 더 기술적인 안내서도 있습니다.

## 문서를 추가하고 싶으신가요? [[do-you-want-to-add-documentation]]

우리는 언제나 더 명확하고 정확한 문서를 위한 개선을 찾고 있습니다. 오탈자나 부족한 내용, 분명하지 않거나 부정확한 내용 등을 알려주시면 개선하는 데 도움이 됩니다. 변경 사항을 만들거나 관심 있으신 분들의 기여를 도와드릴 준비가 되어 있습니다!

문서를 생성, 빌드 및 작성하는 방법에 대한 자세한 내용은 문서의 [README](https://github.com/huggingface/transformers/tree/main/docs)를 확인해 주세요.

## 풀 리퀘스트(Pull Request) 생성하기 [[create-a-pull-request]]

코드를 작성하기 전에 기존의 풀 리퀘스트나 이슈를 검색하여 누구도 이미 동일한 작업을 하고 있는지 확인하는 것이 좋습니다. 확실하지 않다면 피드백을 받기 위해 이슈를 열어보는 것이 항상 좋은 아이디어입니다.

🤗 트랜스포머에 기여하기 위해서는 기본적인 `git` 사용 능력이 필요합니다. `git`은 사용하기 쉬운 도구는 아니지만, 가장 훌륭한 매뉴얼을 제공합니다. 쉘(shell)에서 `git --help`을 입력하여 확인하세요! 만약 책을 선호한다면, [Pro Git](https://git-scm.com/book/en/v2)은 매우 좋은 참고 자료입니다.

🤗 트랜스포머에 기여하려면 **[Python 3.8]((https://github.com/huggingface/transformers/blob/main/setup.py#L426))** 버전 이상이 필요합니다. 기여를 시작하려면 아래 단계를 따르세요:

1. 저장소 페이지에서 **[Fork](https://github.com/huggingface/transformers/fork)** 버튼을 클릭하여 저장소를 포크하세요. 이렇게 하면 코드의 복사본이 귀하의 GitHub 사용자 계정 아래에 생성됩니다.

2. 포크한 저장소를 로컬 디스크로 클론하고, 기본 저장소를 원격(remote)으로 추가하세요:

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. 개발 변경 사항을 보유할 새 브랜치를 생성하세요:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 `main` 브랜치에서 작업하지 **마세요!**

4. 다음 명령을 가상 환경에서 실행하여 개발 환경을 설정하세요:

   ```bash
   pip install -e ".[dev]"
   ```

   만약 이미 가상 환경에 🤗 트랜스포머가 설치되어 있다면, `-e` 플래그를 사용하여 설치하기 전에 `pip uninstall transformers`로 제거해주세요.
   
   여러분의 운영체제에 따라서, 그리고 트랜스포머의 선택적 의존성이 증가하면서, 이 명령이 실패할 수도 있습니다. 그럴 경우 사용하려는 딥 러닝 프레임워크(PyTorch, TensorFlow, 그리고/또는 Flax)를 설치한 다음 다음 명령을 실행해주세요:

   ```bash
   pip install -e ".[quality]"
   ```

   대부분의 사용 사례에는 충분할 것입니다.

5. 브랜치에서 기능을 개발하세요.

   코드를 작업하는 동안 테스트 스위트가 통과하는지 확인하세요. 다음과 같이 변경 사항에 영향을 받는 테스트를 실행하세요:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   테스트에 대한 더 많은 정보는 [테스팅](https://huggingface.co/docs/transformers/testing) 가이드를 확인하세요.

   🤗 트랜스포머는 `black`과 `ruff`를 사용하여 소스 코드를 일관되게 서식 지정하고 확인합니다. 변경 사항을 만든 후에는 다음 명령으로 자동 스타일 교정 및 코드 확인을 적용하세요:

   ```bash
   make fixup
   ```

   이 목표는 주로 PR에서 수정된 파일에만 작동합니다.

   하나씩 검사를 실행하려는 경우, 다음 명령으로 스타일 교정을 적용할 수 있습니다:

   ```bash
   make style
   ```

   🤗 트랜스포머는 또한 `ruff`와 몇 가지 사용자 정의 스크립트를 사용하여 코딩 실수를 확인합니다. 품질 컨트롤은 CI에서 실행되지만, 다음 명령으로 동일한 검사를 실행할 수 있습니다:

   ```bash
   make quality
   ```

   마지막으로, 새로운 모델을 추가할 때 어떤 파일을 업데이트하지 않았는지 확인하는 스크립트가 많이 있습니다. 다음 명령으로 이러한 스크립트를 실행할 수 있습니다:

   ```bash
   make repo-consistency
   ```

   이러한 검사에 대해 더 알아보고 문제를 해결하는 방법은 [풀 리퀘스트에서의 검사](https://huggingface.co/docs/transformers/pr_checks) 가이드를 확인하세요.

   만약 `docs/source` 디렉터리 아래의 문서를 수정하는 경우, 문서가 여전히 빌드될 수 있는지 확인하세요. 이 검사는 풀 리퀘스트를 열 때 CI에서도 실행됩니다. 로컬 검사를 실행하려면 문서 빌더를 설치해야 합니다:
   
   ```bash
   pip install ".[docs]"
   ```

   저장소 루트에서 다음 명령을 실행하세요:

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   이 명령은 `~/tmp/test-build` 폴더에 문서를 빌드하며, 생성된 Markdown 파일을 선호하는 편집기로 확인할 수 있습니다. 풀 리퀘스트를 열 때 문서를 GitHub에서도 미리 볼 수 있습니다.

   변경 사항에 만족하면 `git add`로 변경된 파일을 추가하고, `git commit`으로 변경 사항을 로컬에 기록하세요:

   ```bash
   git add modified_file.py
   git commit
   ```

   부디 [좋은 커밋 메시지](https://chris.beams.io/posts/git-commit/)를 작성하여 변경 사항을 명확하게 전달하세요!

   변경 사항을 프로젝트 원본 저장소와 동기화하려면, PR을 *열기 전에* 브랜치를 `upstream/branch` 위로 리베이스(rebase)하세요. 또는 유지 관리자의 요청에 따라 진행할 수 있습니다:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
   
   변경 사항을 브랜치에 푸시하세요:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   이미 PR을 열었다면, `--force` 플래그와 함께 강제 푸시해야 합니다. 그렇지 않은 경우, 아직 PR이 열리지 않았다면 일반적으로 변경 사항을 푸시할 수 있습니다.

6. 이제 GitHub에서 저장소 포크로 이동하여 **Pull request(풀 리퀘스트)**를 클릭하여 풀 리퀘스트를 열 수 있습니다. 아래의 [체크리스트](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#pull-request-checklist)에서 모든 항목에 체크하도록 하세요. 준비가 되었다면, 변경 사항을 프로젝트 유지 관리자에게 검토 요청할 수 있습니다.

7. 유지 관리자가 변경 사항을 요청하면 괜찮습니다. 우리의 핵심 기여자들도 동일한 상황을 겪기 때문입니다! 모두가 변경 사항을 풀 리퀘스트에서 볼 수 있도록, 로컬 브랜치에서 작업하고 변경 사항을 포크로 푸시하세요. 그러면 변경 사항이 자동으로 풀 리퀘스트에 나타납니다.

### 풀 리퀘스트 체크리스트 [[pull-request-checklist]]

☐ 풀 리퀘스트 제목은 기여 내용을 요약해야 합니다.<br>
☐ 풀 리퀘스트가 이슈를 해결하는 경우, 풀 리퀘스트 설명에 이슈 번호를 언급하여 연결됨을 보장하세요 (이슈를 보는 사람들이 해당 작업을 수행 중임을 알 수 있게 합니다).<br>
☐ 진행 중인 작업을 나타내려면 제목 앞에 `[WIP]`를 붙여주세요. 이것은 중복 작업을 피하고 병합 준비가 된 PR과 구분하기에 유용합니다.<br>
☐ 기존 테스트가 통과되는지 확인하세요.<br>
☐ 새로운 기능을 추가하는 경우, 해당 기능에 대한 테스트도 추가하세요.<br>
   - 새 모델을 추가하는 경우, `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)`을 사용하여 일반적인 테스트를 활성화하세요.
   - 새 `@slow` 테스트를 추가하는 경우, 다음 명령으로 테스트가 통과되는지 확인하세요: `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`.
   - 새 토크나이저를 추가하는 경우, 테스트를 작성하고 다음 명령으로 테스트가 통과되는지 확인하세요: `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py`. CircleCI에서는 느린 테스트를 실행하지 않지만, GitHub Actions에서는 매일 밤 실행됩니다!<br>

☐ 모든 공개 메서드는 유용한 docstring을 가져야 합니다 (예: [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) 참조).<br>
☐ 빠르게 성장하는 저장소로 인해, 저장소의 무게를 크게 증가시킬 수 있는 이미지, 비디오 및 기타 비텍스트 파일을 추가하지 마세요. 대신 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)과 같은 Hub 저장소를 사용하여 이러한 파일을 호스팅하고 URL을 통해 참조하세요. 관련 문서 이미지를 다음 저장소에 배치하는 것을 권장합니다: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images). 이 데이터셋 저장소에서 PR을 열어서 Hugging Face 멤버에게 병합을 요청할 수 있습니다.

풀 리퀘스트에서 실행되는 검사에 대한 자세한 정보는 [풀 리퀘스트에서의 검사](https://huggingface.co/docs/transformers/pr_checks) 가이드를 확인하세요.

### 테스트 [[tests]]

풀 리퀘스트를 위한 포괄적인 테스트 스위트가 포함되어 있으며, 라이브러리 동작 및 여러 예제를 테스트합니다. 라이브러리 테스트는 [tests](https://github.com/huggingface/transformers/tree/main/tests) 폴더에, 예제 테스트는 [examples](https://github.com/huggingface/transformers/tree/main/examples) 폴더에 있습니다.

빠른 실행과 분산 실행을 지원하는 `pytest`와 `pytest-xdist`를 선호합니다. 저장소 루트에서 테스트를 실행하려면 *하위 폴더 경로 또는 테스트 파일 경로*를 지정하세요.

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

마찬가지로 `examples` 디렉토리에서도 *하위 폴더 경로 또는 테스트 파일 경로*를 지정하세요. 예를 들어, 다음 명령은 PyTorch `examples` 디렉토리의 텍스트 분류 하위 폴더를 테스트합니다:

```bash
pip install -r examples/xxx/requirements.txt  # only needed the first time
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

사실, 이것이 실제로 `make test` 및 `make test-examples` 명령의 구현 방식입니다 (`pip install`은 제외합니다)!

또한 특정 기능만 테스트하려면 더 작은 일부 테스트를 지정할 수 있습니다.

기본적으로 느린 테스트는 건너뛰지만 `RUN_SLOW` 환경 변수를 `yes`로 설정하여 실행할 수 있습니다. 이렇게 하면 많은 기가바이트의 모델이 다운로드되므로 충분한 디스크 공간, 좋은 인터넷 연결 또는 많은 인내가 필요합니다!

<Tip warning={true}>

테스트를 실행하려면 *하위 폴더 경로 또는 테스트 파일 경로*를 지정하세요. 그렇지 않으면 `tests` 또는 `examples` 폴더의 모든 테스트를 실행하게 되어 매우 긴 시간이 걸립니다!

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

느린 테스트와 마찬가지로, 다음과 같이 테스트 중에 기본적으로 활성화되지 않는 다른 환경 변수도 있습니다:
- `RUN_CUSTOM_TOKENIZERS`: 사용자 정의 토크나이저 테스트를 활성화합니다.
- `RUN_PT_FLAX_CROSS_TESTS`: PyTorch + Flax 통합 테스트를 활성화합니다.
- `RUN_PT_TF_CROSS_TESTS`: TensorFlow + PyTorch 통합 테스트를 활성화합니다.

더 많은 환경 변수와 추가 정보는 [testing_utils.py](src/transformers/testing_utils.py)에서 찾을 수 있습니다.

🤗 Transformers는 테스트 실행기로 `pytest`를 사용합니다. 그러나 테스트 스위트 자체에서는 `pytest` 특정 기능을 사용하지 않습니다.

이것은 `unittest`가 완전히 지원된다는 것을 의미합니다. 다음은 `unittest`로 테스트를 실행하는 방법입니다:

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### 스타일 가이드 [[style-guide]]

문서 문자열은 [Google Python 스타일 가이드](https://google.github.io/styleguide/pyguide.html)를 따릅니다. 자세한 정보는 [문서 작성 가이드](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)를 확인하세요.

### Windows에서 개발 [[develop-on-windows]]

Windows에서 (Windows Subsystem for Linux 또는 WSL에서 작업하지 않는 한) Windows `CRLF` 줄 바꿈을 Linux `LF` 줄 바꿈으로 변환하도록 git을 구성해야 합니다:

```bash
git config core.autocrlf input
```

Windows에서 `make` 명령을 실행하는 한 가지 방법은 MSYS2를 사용하는 것입니다:

1. [MSYS2 다운로드](https://www.msys2.org/)하고 `C:\msys64`에 설치되었다고 가정합니다.
2. 명령 줄에서 `C:\msys64\msys2.exe`를 열어봅니다 (시작 메뉴에서 사용 가능해야 함).
3. 쉘에서 실행: `pacman -Syu` 및 `pacman -S make`로 `make`를 설치합니다.
4. 환경 변수 PATH에 `C:\msys64\usr\bin`을 추가하세요.

이제 모든 터미널 (Powershell, cmd.exe 등)에서 `make`를 사용할 수 있습니다! 🎉

### 포크한 저장소를 상위 원본 브랜치(main)과 동기화하기 (Hugging Face 저장소) [[sync-a-forked-repository-with-upstream-main-the-hugging-face-repository]]

포크한 저장소의 main 브랜치를 업데이트할 때, 다음 단계를 따라 upstream 저장소에 참조 노트를 추가하지 않고 동기화하세요. 이렇게 하면 각 upstream PR에 참조 노트가 추가되는 것을 피하고 이러한 PR에 관여하는 개발자들에게 불필요한 알림이 전송되는 것을 방지할 수 있습니다.

1. 가능한 경우, 포크된 main에 직접 병합하면서 upstream과 동기화하는 것을 피하세요.
2. PR이 반드시 필요한 경우, 브랜치를 확인한 후 다음 단계를 사용하세요:

```bash
git checkout -b your-branch-for-syncing
git pull --squash --no-commit upstream main
git commit -m '<your message without GitHub references>'
git push --set-upstream origin your-branch-for-syncing
```