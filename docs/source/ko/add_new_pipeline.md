<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 커스텀 파이프라인을 어떻게 생성하나요? [[how-to-create-a-custom-pipeline]]

이 가이드에서는 커스텀 파이프라인을 어떻게 생성하고 [허브](hf.co/models)에 공유하거나 🤗 Transformers 라이브러리에 추가하는 방법을 살펴보겠습니다.

먼저, 파이프라인이 수용할 수 있는 원시 엔트리를 결정해야 합니다.
문자열, 바이트, 사전 또는 가장 원하는 입력에 가장 적합한 것을 선택할 수 있습니다.
이 입력을 가능한 한 순수한 Python 형식으로 유지하는 것이 좋습니다.
이렇게 하면 호환성이 쉬워집니다(다른 언어를 통한 JSON을 통해 가능).
이러한 것들은 파이프라인의 `inputs`(전처리)이 될 것입니다.

그런 다음 `outputs`를 정의합니다.
`inputs`와 같은 정책을 따릅니다.
간단할수록 좋습니다.
이것들은 `postprocess` 메서드의 출력이 될 것입니다.

먼저 4개의 메서드(`preprocess`, `_forward`, `postprocess` 및 `_sanitize_parameters`)를 구현하기 위해 기본 클래스 `Pipeline`을 상속하여 시작합니다.


```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

이 분해 구조의 목적은 CPU/GPU에 대한 비교적 원활한 지원을 제공하면서, CPU에서 다른 스레드에서 사전/후처리를 수행할 수 있는 지원을 제공하는 것입니다.

`preprocess`는 최초에 정의된 입력을 가져와 모델에 피드할 수 있는 형식으로 변환합니다.
더 많은 정보를 포함할 수 있으며 일반적으로 `Dict` 형태입니다.

`_forward`는 구현 세부 사항이며 직접 호출되지 않도록 설계되었습니다. 
`forward`가 호출될 때 모든 것이 예상된 장치에서 작동되는지 확인하기 위한 보호 장치가 포함되어 있습니다.
실제 모델과 관련된 것은 `_forward` 메서드에 속하며, 나머지는 전처리/후처리에 속합니다.

`postprocess` 메서드는 `_forward`의 출력을 가져와 이전에 결정한 최종 출력 형식으로 변환합니다.

`_sanitize_parameters`는 사용자가 원하는 경우 언제든지 매개변수를 전달할 수 있도록 허용합니다. 초기화 시간에 `pipeline(...., maybe_arg=4)`이나 호출 시간에 `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`과 같이 사용할 수 있습니다.

`_sanitize_parameters`의 반환 값은 `preprocess`, `_forward`, `postprocess`에 직접 전달되는 3개의 kwargs 딕셔너리입니다.
호출자가 추가 매개변수로 호출하지 않았다면 아무것도 채우지 마십시오.
이렇게 하면 함수 정의의 기본 인수를 유지할 수 있습니다.
이것이 항상 더 "자연스러운" 것입니다.

분류 작업에서 `top_k` 매개변수가 대표적인 예입니다.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

이를 달성하기 위해 우리는 `postprocess` 메서드를 기본 매개변수인 `5`로 업데이트하고 `_sanitize_parameters`를 수정하여 이 새 매개변수를 허용합니다.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

입력/출력을 가능한한 간단하고 이상적으로 JSON 직렬화 가능한 형식으로 유지하려고 노력하십시오.
이렇게 하면 사용자가 새로운 종류의 개체를 이해하지 않고도 파이프라인을 쉽게 사용할 수 있습니다.
또한 사용 용이성을 위해 여러 가지 유형의 인수를 지원하는 것이 상대적으로 흔한 방법입니다(오디오 파일은 파일 이름, URL 또는 순수한 바이트일 수 있음).



## 지원되는 작업 목록에 추가하기 [[adding-it-to-the-list-of-supported-tasks]]

`new-task`를 지원되는 작업 목록에 등록하려면 `PIPELINE_REGISTRY`에 추가해야 합니다:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

원하는 경우 기본 모델을 지정할 수 있으며, 이 경우 특정 리비전(분기 이름 또는 커밋 해시일 수 있음, 여기서는 "abcdef")과 유형을 함께 가져와야 합니다:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## 허브에 파이프라인 공유하기 [[share-your-pipeline-on-the-hub]]

허브에 사용자 지정 파이프라인을 공유하려면 `Pipeline` 하위 클래스의 사용자 지정 코드를 Python 파일에 저장하기만 하면 됩니다.
예를 들어, 다음과 같이 문장 쌍 분류를 위한 사용자 정의 파이프라인을 사용하려는 경우:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

구현은 프레임워크에 독립적이며 PyTorch와 TensorFlow 모델 모두에서 작동합니다.
이를 `pair_classification.py`라는 파일에 저장한 경우 다음과 같이 가져오고 등록할 수 있습니다:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

이 작업이 완료되면 사전 훈련된 모델과 함께 사용할 수 있습니다.
예를 들어, `sgugger/finetuned-bert-mrpc`은 MRPC 데이터셋에서 미세 조정된 모델로 문장 쌍을 패러프레이즈로 분류합니다.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

그런 다음 `Repository`의 `save_pretrained` 메서드를 사용하여 허브에 공유할 수 있습니다:

```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

이렇게 하면 "test-dynamic-pipeline" 폴더 내에 `PairClassificationPipeline`을 정의한 파일이 복사되며, 파이프라인의 모델과 토크나이저도 저장된 다음 모두 리포지토리 `{your_username}/test-dynamic-pipeline`에 푸시됩니다.
이후에는 누구나 `trust_remote_code=True` 옵션을 제공하는 한 사용할 수 있습니다.

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## 파이프라인을 🤗 Transformers에 추가하기 [[add-the-pipeline-to-transformers]]

사용자 정의 파이프라인을 🤗 Transformers에 기여하려면 `pipelines` 하위 모듈에 새 모듈을 추가한 다음, `pipelines/__init__.py`에서 정의된 작업 목록에 추가해야 합니다.

그런 다음 테스트를 추가해야 합니다.
`tests/test_pipelines_MY_PIPELINE.py`라는 새 파일을 만들고 다른 테스트와 예제를 함께 작성합니다.

`run_pipeline_test` 함수는 매우 일반적이며, `model_mapping` 및 `tf_model_mapping`에서 정의한 모든 가능한 아키텍처에서 작은 무작위 모델에서 실행됩니다.

이는 미래 호환성을 테스트하는 데 매우 중요합니다.
즉, 누군가가 `XXXForQuestionAnswering`을 위한 새 모델을 추가하면 파이프라인 테스트는 해당 모델에서 실행하려고 시도합니다.
모델이 무작위이기 때문에 실제 값을 확인할 수 없으므로 출력 형식을 일치시키기 위한 도우미 `ANY`가 있습니다.

또한 2개(이상하게 4개)의 테스트를 구현해야 합니다.

- `test_small_model_pt`: 이 파이프라인에 대한 작은 모델 1개를 정의(결과가 의미 없더라도 상관없음)하고 파이프라인 출력을 테스트합니다.
결과는 `test_small_model_tf`와 동일해야 합니다.
- `test_small_model_tf`: 이 파이프라인에 대한 작은 모델 1개를 정의(결과가 의미 없더라도 상관없음)하고 파이프라인 출력을 테스트합니다.
결과는 `test_small_model_pt`와 동일해야 합니다.
- `test_large_model_pt`(`선택사항`): 결과가 의미 있는 실제 파이프라인에서 파이프라인을 테스트합니다.
이러한 테스트는 느리며 그렇게 표시되어야 합니다.
여기서의 목표는 파이프라인을 쇼케이스하고 미래 릴리스에서의 변화가 없는지 확인하는 것입니다.
- `test_large_model_tf`(`선택사항`): 결과가 의미 있는 실제 파이프라인에서 파이프라인을 테스트합니다.
이러한 테스트는 느리며 그렇게 표시되어야 합니다.
여기서의 목표는 파이프라인을 쇼케이스하고 미래 릴리스에서의 변화가 없는지 확인하는 것입니다.
