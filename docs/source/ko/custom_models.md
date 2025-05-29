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

# 사용자 정의 모델 공유하기[[sharing-custom-models]]

🤗 Transformers 라이브러리는 쉽게 확장할 수 있도록 설계되었습니다. 
모든 모델은 추상화 없이 저장소의 지정된 하위 폴더에 완전히 코딩되어 있으므로, 손쉽게 모델링 파일을 복사하고 필요에 따라 조정할 수 있습니다.

완전히 새로운 모델을 만드는 경우에는 처음부터 시작하는 것이 더 쉬울 수 있습니다.
이 튜토리얼에서는 Transformers 내에서 사용할 수 있도록 사용자 정의 모델과 구성을 작성하는 방법과 
🤗 Transformers 라이브러리에 없는 경우에도 누구나 사용할 수 있도록 (의존성과 함께) 커뮤니티에 공유하는 방법을 배울 수 있습니다.

[timm 라이브러리](https://github.com/rwightman/pytorch-image-models)의 ResNet 클래스를 [`PreTrainedModel`]로 래핑한 ResNet 모델을 예로 모든 것을 설명합니다.

## 사용자 정의 구성 작성하기[[writing-a-custom-configuration]]

모델에 들어가기 전에 먼저 구성을 작성해보도록 하겠습니다.
모델의 `configuration`은 모델을 만들기 위해 필요한 모든 중요한 것들을 포함하고 있는 객체입니다.
다음 섹션에서 볼 수 있듯이, 모델은 `config`를 사용해서만 초기화할 수 있기 때문에 완벽한 구성이 필요합니다.

아래 예시에서는 ResNet 클래스의 인수(argument)를 조정해보겠습니다.
다른 구성은 가능한 ResNet 중 다른 유형을 제공합니다.
그런 다음 몇 가지 유효성을 확인한 후 해당 인수를 저장합니다.

```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

사용자 정의 `configuration`을 작성할 때 기억해야 할 세 가지 중요한 사항은 다음과 같습니다:
- `PretrainedConfig`을 상속해야 합니다.
- `PretrainedConfig`의 `__init__`은 모든 kwargs를 허용해야 하고,
- 이러한 `kwargs`는 상위 클래스 `__init__`에 전달되어야 합니다.

상속은 🤗 Transformers 라이브러리에서 모든 기능을 가져오는 것입니다.
이러한 점으로부터 비롯되는 두 가지 제약 조건은 `PretrainedConfig`에 설정하는 것보다 더 많은 필드가 있습니다.
`from_pretrained` 메서드로 구성을 다시 로드할 때 해당 필드는 구성에서 수락한 후 상위 클래스로 보내야 합니다.

모델을 auto 클래스에 등록하지 않는 한, `configuration`에서 `model_type`을 정의(여기서 `model_type="resnet"`)하는 것은 필수 사항이 아닙니다 (마지막 섹션 참조).

이렇게 하면 라이브러리의 다른 모델 구성과 마찬가지로 구성을 쉽게 만들고 저장할 수 있습니다.
다음은 resnet50d 구성을 생성하고 저장하는 방법입니다:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

이렇게 하면 `custom-resnet` 폴더 안에 `config.json`이라는 파일이 저장됩니다.
그런 다음 `from_pretrained` 메서드를 사용하여 구성을 다시 로드할 수 있습니다.

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

구성을 Hub에 직접 업로드하기 위해 [`PretrainedConfig`] 클래스의 [`~PretrainedConfig.push_to_hub`]와 같은 다른 메서드를 사용할 수 있습니다.


## 사용자 정의 모델 작성하기[[writing-a-custom-model]]

이제 ResNet 구성이 있으므로 모델을 작성할 수 있습니다.
실제로는 두 개를 작성할 것입니다. 하나는 이미지 배치에서 hidden features를 추출하는 것([`BertModel`]과 같이), 다른 하나는 이미지 분류에 적합한 것입니다([`BertForSequenceClassification`]과 같이).

이전에 언급했듯이 이 예제에서는 단순하게 하기 위해 모델의 느슨한 래퍼(loose wrapper)만 작성할 것입니다.
이 클래스를 작성하기 전에 블록 유형과 실제 블록 클래스 간의 매핑 작업만 하면 됩니다.
그런 다음 `ResNet` 클래스로 전달되어 `configuration`을 통해 모델이 선언됩니다:

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

이미지 분류 모델을 만들기 위해서는 forward 메소드만 변경하면 됩니다:

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

두 경우 모두 `PreTrainedModel`를 상속받고, `config`를 통해 상위 클래스 초기화를 호출하다는 점을 기억하세요 (일반적인 `torch.nn.Module`을 작성할 때와 비슷함).
모델을 auto 클래스에 등록하고 싶은 경우에는 `config_class`를 설정하는 부분이 필수입니다 (마지막 섹션 참조).

<Tip>

라이브러리에 존재하는 모델과 굉장히 유사하다면, 모델을 생성할 때 구성을 참조해 재사용할 수 있습니다.

</Tip>

원하는 것을 모델이 반환하도록 할 수 있지만, `ResnetModelForImageClassification`에서 했던 것 처럼
레이블을 통과시켰을 때 손실과 함께 사전 형태로 반환하는 것이 [`Trainer`] 클래스 내에서 직접 모델을 사용하기에 유용합니다.
자신만의 학습 루프 또는 다른 학습 라이브러리를 사용할 계획이라면 다른 출력 형식을 사용해도 좋습니다.

이제 모델 클래스가 있으므로 하나 생성해 보겠습니다:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

다시 말하지만, [`~PreTrainedModel.save_pretrained`]또는 [`~PreTrainedModel.push_to_hub`]처럼 [`PreTrainedModel`]에 속하는 모든 메소드를 사용할 수 있습니다.
다음 섹션에서 두 번째 메소드를 사용해 모델 코드와 모델 가중치를 업로드하는 방법을 살펴보겠습니다.
먼저, 모델 내부에 사전 훈련된 가중치를 로드해 보겠습니다.

이 예제를 활용할 때는, 사용자 정의 모델을 자신만의 데이터로 학습시킬 것입니다.
이 튜토리얼에서는 빠르게 진행하기 위해 사전 훈련된 resnet50d를 사용하겠습니다.
아래 모델은 resnet50d의 래퍼이기 때문에, 가중치를 쉽게 로드할 수 있습니다.


```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

이제 [`~PreTrainedModel.save_pretrained`] 또는 [`~PreTrainedModel.push_to_hub`]를 사용할 때 모델 코드가 저장되는지 확인해봅시다.

## Hub로 코드 업로드하기[[sending-the-code-to-the-hub]]

<Tip warning={true}>

이 API는 실험적이며 다음 릴리스에서 약간의 변경 사항이 있을 수 있습니다.

</Tip>

먼저 모델이 `.py` 파일에 완전히 정의되어 있는지 확인하세요.
모든 파일이 동일한 작업 경로에 있기 때문에 상대경로 임포트(relative import)에 의존할 수 있습니다 (transformers에서는 이 기능에 대한 하위 모듈을 지원하지 않습니다).
이 예시에서는 작업 경로 안의 `resnet_model`에서 `modeling_resnet.py` 파일과 `configuration_resnet.py` 파일을 정의합니다.
구성 파일에는 `ResnetConfig`에 대한 코드가 있고 모델링 파일에는 `ResnetModel` 및 `ResnetModelForImageClassification`에 대한 코드가 있습니다.

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

Python이 `resnet_model`을 모듈로 사용할 수 있도록 감지하는 목적이기 때문에 `__init__.py`는 비어 있을 수 있습니다.

<Tip warning={true}>

라이브러리에서 모델링 파일을 복사하는 경우,
모든 파일 상단에 있는 상대 경로 임포트(relative import) 부분을 `transformers` 패키지에서 임포트 하도록 변경해야 합니다.

</Tip>

기존 구성이나 모델을 재사용(또는 서브 클래스화)할 수 있습니다.

커뮤니티에 모델을 공유하기 위해서는 다음 단계를 따라야 합니다:
먼저, 새로 만든 파일에 ResNet 모델과 구성을 임포트합니다:

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

다음으로 `save_pretrained` 메소드를 사용해 해당 객체의 코드 파일을 복사하고, 
복사한 파일을 Auto 클래스로 등록하고(모델인 경우) 실행합니다:

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

`configuration`에 대한 auto 클래스를 지정할 필요는 없지만(`configuration` 관련 auto 클래스는 AutoConfig 클래스 하나만 있음), 모델의 경우에는 지정해야 합니다.
사용자 지정 모델은 다양한 작업에 적합할 수 있으므로, 모델에 맞는 auto 클래스를 지정해야 합니다.

다음으로, 이전에 작업했던 것과 마찬가지로 구성과 모델을 작성합니다:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

이제 모델을 Hub로 업로드하기 위해 로그인 상태인지 확인하세요. 
터미널에서 다음 코드를 실행해 확인할 수 있습니다:

```bash
huggingface-cli login
```

주피터 노트북의 경우에는 다음과 같습니다:

```py
from huggingface_hub import notebook_login

notebook_login()
```

그런 다음 이렇게 자신의 네임스페이스(또는 자신이 속한 조직)에 업로드할 수 있습니다:
```py
resnet50d.push_to_hub("custom-resnet50d")
```

On top of the modeling weights and the configuration in json format, this also copied the modeling and
configuration `.py` files in the folder `custom-resnet50d` and uploaded the result to the Hub. You can check the result
in this [model repo](https://huggingface.co/sgugger/custom-resnet50d).
json 형식의 모델링 가중치와 구성 외에도 `custom-resnet50d` 폴더 안의 모델링과 구성 `.py` 파일을 복사하해 Hub에 업로드합니다.
[모델 저장소](https://huggingface.co/sgugger/custom-resnet50d)에서 결과를 확인할 수 있습니다.

[sharing tutorial](model_sharing) 문서의 `push_to_hub` 메소드에서 자세한 내용을 확인할 수 있습니다.


## 사용자 정의 코드로 모델 사용하기[[using-a-model-with-custom-code]]

auto 클래스와 `from_pretrained` 메소드를 사용하여 사용자 지정 코드 파일과 함께 모든 구성, 모델, 토크나이저를 사용할 수 있습니다.
Hub에 업로드된 모든 파일 및 코드는 멜웨어가 있는지 검사되지만 (자세한 내용은 [Hub 보안](https://huggingface.co/docs/hub/security#malware-scanning) 설명 참조),
자신의 컴퓨터에서 모델 코드와 작성자가 악성 코드를 실행하지 않는지 확인해야 합니다.
사용자 정의 코드로 모델을 사용하려면 `trust_remote_code=True`로 설정하세요:

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

모델 작성자가 악의적으로 코드를 업데이트하지 않았다는 점을 확인하기 위해, 커밋 해시(commit hash)를 `revision`으로 전달하는 것도 강력히 권장됩니다 (모델 작성자를 완전히 신뢰하지 않는 경우).

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Hub에서 모델 저장소의 커밋 기록을 찾아볼 때, 모든 커밋의 커밋 해시를 쉽게 복사할 수 있는 버튼이 있습니다.

## 사용자 정의 코드로 만든 모델을 auto 클래스로 등록하기[[registering-a-model-with-custom-code-to-the-auto-classes]]

🤗 Transformers를 상속하는 라이브러리를 작성하는 경우 사용자 정의 모델을 auto 클래스에 추가할 수 있습니다.
사용자 정의 모델을 사용하기 위해 해당 라이브러리를 임포트해야 하기 때문에, 이는 Hub로 코드를 업로드하는 것과 다릅니다 (Hub에서 자동적으로 모델 코드를 다운로드 하는 것과 반대).

구성에 기존 모델 유형과 다른 `model_type` 속성이 있고 모델 클래스에 올바른 `config_class` 속성이 있는 한,
다음과 같이 auto 클래스에 추가할 수 있습니다:

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

사용자 정의 구성을 [`AutoConfig`]에 등록할 때 사용되는 첫 번째 인수는 사용자 정의 구성의 `model_type`과 일치해야 합니다.
또한, 사용자 정의 모델을 auto 클래스에 등록할 때 사용되는 첫 번째 인수는 해당 모델의 `config_class`와 일치해야 합니다.