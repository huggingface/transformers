# 모듈식 트랜스포머 [[modular-transformers]]

`transformers`는 opinionated(자기 의견이 강한) 프레임워크이며, 우리의 철학은 다음의 [개념 가이드](./philosophy)에 정의되어 있습니다.

이 철학의 핵심은 라이브러리의 [단일 모델, 단일 파일](https://huggingface.co/blog/transformers-design-philosophy) 측면에서 잘 나타납니다. 이 구성 요소의 단점은 파일 간에 구성 요소의 상속과 임포트 가능성을 제한한다는 것입니다.

그 결과, 모델 구성 요소가 여러 파일에 걸쳐 반복되는 경향이 있습니다. `transformers`에는 모델 수만큼 많은 어텐션 레이어가 정의되어 있으며, 그 중 상당수는 서로 동일합니다. 안타깝게도, 수정과 변경 사항이 코드의 특정 부분에 적용되면서 독립적인 구현들이 서로 분기되는 경향이 있습니다.

이 문제를 적절히 해결하기 위해, 우리는 라이브러리 전체에 "복사본"의 개념을 도입했습니다. 코드가 다른 코드의 복사본임을 나타내는 주석을 추가함으로써, CI 및 로컬 명령을 통해 복사본이 분기되지 않도록 강제할 수 있습니다. 그러나 복잡성이 낮더라도 이는 종종 매우 번거로운 작업입니다.

마지막으로, 이 방식은 우리가 줄이고자 하는 상당한 오버헤드를 모델 기여 과정에 추가하게 됩니다. 이 접근 방식은 종종 모델 기여에 모델링 코드(~1,000줄), 프로세서(~500줄), 테스트, 문서 등을 추가해야 합니다. 모델 기여 PR은 대부분 3,000~5,000줄 이상의 코드를 추가하며, 이 중 많은 부분이 보일러플레이트(boilerplate) 코드입니다.

이는 기여의 장벽을 높이며, 모듈식 트랜스포머를 통해 우리는 이러한 장벽을 훨씬 더 수용 가능한 수준으로 낮추고자 합니다.

## 무엇인가요 [[what-is-it]]

모듈식 트랜스포머는 모델 폴더에 "모듈식" 파일의 개념을 도입합니다. 이 모듈식 파일은 일반적으로 모델링/프로세싱 파일에서 허용되지 않는 코드를 허용하며, 이는 인접한 모델로부터의 임포트와 클래스 간의 상속을 허용합니다.

이 모듈식 파일은 각각의 별도의 모듈에서 정의되었을 모델, 프로세서 및 구성 클래스를 정의합니다.

마지막으로, 이 기능은 모듈식 파일을 "풀어내어" 단일 모델, 단일 파일 디렉토리 구조로 변환하는 새로운 `linter`를 도입합니다. 이 파일들은 스크립트가 실행될 때마다 자동으로 생성되며, 기여해야 할 내용을 모듈식 파일, 그리고 기여된 모델과 다른 모델 간의 차이점으로만 줄여줍니다.

모델 사용자는 단일 파일 인터페이스를 임포트하고 사용하게 되므로, 여기에는 변화가 없을 것입니다. 이를 통해 간단한 기여를 가능하게 하면서도 우리의 철학을 유지하는 양쪽의 장점을 결합하고자 합니다.

따라서 이는 `# Copied from` 마커의 대체품이며, 이전에 기여된 모델은 앞으로 몇 달 내에 새로운 모듈식 트랜스포머 형식으로 전환될 예정입니다.

### 자세한 내용 [[details]]

“linter”는 상속 구조를 풀어서 모듈화된 파일로부터 모든 단일 파일을 생성하며, Python 사용자들에게는 그 과정이 보이지 않도록 동작합니다. 현재 linter는 **단일** 수준의 상속만을 평탄화합니다.

예를 들어:
- 구성 클래스가 다른 클래스를 상속하고 인자를 추가/삭제하는 경우, 생성된 파일은 직접 참조(추가의 경우)하거나 완전히 제거합니다(삭제의 경우).
- 클래스가 다른 클래스를 상속하는 경우, 예를 들어 class GemmaModel(LlamaModel): 의 경우, 종속성이 자동으로 추론됩니다. 모든 서브모듈은 슈퍼클래스로부터 자동으로 추론됩니다.

토크나이저, 이미지 프로세서, 모델, 구성 등을 이 `modular` 파일에 모두 작성할 수 있으며, 해당 파일들이 자동으로 생성됩니다.

### 시행 [[enforcement]]

[TODO] 우리는 새로운 테스트를 도입하여 생성된 콘텐츠가 `modular_xxxx.py`에 있는 내용과 일치하는지 확인합니다.

### 예시 [[examples]]

여기 BERT와 RoBERTa의 간단한 예가 있습니다. 두 모델은 밀접하게 관련되어 있으며, 모델 구현의 차이는 임베딩 레이어의 변경에서만 있습니다.

모델을 완전히 재정의하는 대신, `modular_roberta.py` 파일은 모델링 및 구성 클래스를 위해 다음과 같이 생겼습니다. (예시를 위해, 토크나이저는 매우 다르므로 일단 무시합니다.)

```python
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# RoBERTa 구성은 BERT의 구성과 동일합니다
class RobertaConfig(BertConfig):
    model_type = 'roberta'

# 여기서 패딩 ID 차이를 강조하기 위해 임베딩을 재정의하고, 위치 임베딩을 재정의합니다
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# RoBERTa 모델은 임베딩 레이어를 제외하면 BERT 모델과 동일합니다.
# 위에서 임베딩을 재정의했으므로, 여기서는 추가 작업이 필요 없습니다
class RobertaModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)

# 헤드는 이제 내부에서 올바른 `RobertaModel`을 재정의하기만 하면 됩니다
class RobertaForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = RobertaModel(config)
```

정의한 종속성을 사용하지 않으면 다음과 같은 오류가 발생합니다:

```bash
ValueError: You defined `RobertaEmbeddings` in the modular_roberta.py, it should be used
                                        when you define `BertModel`, as it is one of it's direct dependencies. Make sure
                                        you use it in the `__init__` function.
```

또한, 다음에서 예시 목록을 찾을 수 있습니다:

## 무엇이 아닌가요 [[what-it-is-not]]

(아직은?) 모델링 코드를 대체하는 것은 아닙니다. 그리고 여러분의 모델이 지금까지 존재했던 다른 어떤 것에도 기반하지 않는다면, 기존과 같이 `modeling` 파일을 추가할 수 있습니다.
