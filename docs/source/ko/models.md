<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 모델 로드하기[[loading-models]]

Transformers는 한 줄의 코드로 사용할 수 있는 많은 사전 훈련된 모델을 제공합니다. 모델 클래스와 [`~PreTrainedModel.from_pretrained`] 메소드가 필요합니다.

[`~PreTrainedModel.from_pretrained`]를 호출하여 Hugging Face [Hub](https://hf.co/models)에 저장된 모델의 가중치와 구성을 다운로드하고 로드하세요.

> [!TIP]
> [`~PreTrainedModel.from_pretrained`] 메소드는 [safetensors](https://hf.co/docs/safetensors/index) 파일 형식으로 저장된 가중치가 있으면 이를 로드합니다. 전통적으로 PyTorch 모델 가중치는 보안에 취약한 것으로 알려진 [pickle](https://docs.python.org/3/library/pickle.html) 유틸리티로 직렬화됩니다. Safetensor 파일은 더 안전하고 로드 속도가 빠릅니다.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map="auto")
```

이 가이드는 모델을 불러오는 방법, 다양한 로딩 방식, 매우 큰 모델에서 발생할 수 있는 메모리 문제를 해결하는 방법, 그리고 사용자 정의 모델을 불러오는 방법을 설명합니다.

## 모델과 구성[[models-and-configurations]]

모든 모델에는 은닉 레이어 수, 어휘 사전 크기, 활성화 함수 등과 같은 특정 속성이 포함된 `configuration.py` 파일이 있습니다. 또한 각 레이어의 정의와 각각의 레이어 안에서 일어나는 수학적 연산을 정의하는 `modeling.py` 파일도 있습니다. `modeling.py` 파일은 `configuration.py`에 정의된 모델 속성을 바탕으로 모델을 구축합니다. 이 단계에서는 아직 학습되지 않은 무작위 가중치를 가진 상태이기 때문에, 의미 있는 출력을 얻기 위해서는 학습이 필요합니다.

<!-- insert diagram of model and configuration -->

> [!TIP]
> *아키텍처(Architecture)*는 모델의 골격을 의미하고 *체크포인트(checkpoint)*는 주어진 아키텍처에 대한 모델의 가중치를 의미합니다. 예를 들어, [BERT](./model_doc/bert)는 아키텍처이고 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)는 해당 아키텍처의 체크포인트(checkpoint)입니다. *모델*이라는 용어는 아키텍처 및 체크포인트(checkpoint)와 혼용하여 사용되는 것을 볼 수 있습니다.

로드할 수 있는 모델은 일반적으로 두 가지 타입이 있습니다.

1. 은닉 상태를 출력하는 [`AutoModel`] 또는 [`LlamaModel`]과 같은 기본 모델입니다.
2. 특정 작업을 수행하기 위해 특정 *헤드*가 붙은 [`AutoModelForCausalLM`] 또는 [`LlamaForCausalLM`]과 같은 모델입니다.

각 모델 타입마다, 각각의 기계학습 프레임워크(PyTorch, TensorFlow, Flax)를 위한 별도의 클래스가 있습니다. 사용 중인 프레임워크에 해당하는 접두어(prefix)를 선택하세요.

<hfoptions id="backend">
<hfoption id="PyTorch">

```py
from transformers import AutoModelForCausalLM, MistralForCausalLM

# AutoClass 또는 모델별 클래스(model-specific class) 를 이용해 로드
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")
```

</hfoption>
<hfoption id="TensorFlow">

```py
from transformers import TFAutoModelForCausalLM, TFMistralForCausalLM

# AutoClass 또는 모델별 클래스(model-specific class) 를 이용해 로드
model = TFAutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = TFMistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

</hfoption>
<hfoption id="Flax">

```py
from transformers import FlaxAutoModelForCausalLM, FlaxMistralForCausalLM

# AutoClass 또는 모델별 클래스(model-specific class) 를 이용해 로드
model = FlaxAutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = FlaxMistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

</hfoption>
</hfoptions>

## 모델 클래스[[model-classes]]

사전 훈련된 모델을 가져오려면 모델에 가중치를 로드해야 합니다. 이는 Hugging Face Hub 또는 로컬 디렉터리에서 가중치를 받아들이는 [`~PreTrainedModel.from_pretrained`]를 호출하여 수행됩니다.

두 가지 모델 클래스로 [AutoModel](./model_doc/auto) 클래스와 모델별 클래스가 있습니다.

<hfoptions id="model-classes">
<hfoption id="AutoModel">

<Youtube id="AhChOFRegn4"/>

[AutoModel](./model_doc/auto) 클래스는 정확한 모델 클래스 이름을 몰라도 아키텍처를 불러올 수 있는 편리한 방법입니다. 많은 모델이 제공되기 때문에, 이 클래스는 구성 파일을 기반으로 올바른 모델 클래스를 자동으로 선택해 줍니다. 원하는 작업과 사용하려는 체크포인트만 알고 있으면 됩니다.

주어진 작업을 아키텍처가 지원하는 한, 모델이나 작업을 쉽게 전환할 수 있습니다.

예를 들어, 동일한 모델을 서로 다른 작업에 사용할 수 있습니다.

```py
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

# 동일한 API를 3가지 다른 작업에 사용
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Llama-2-7b-hf")
```

다른 경우에는, 하나의 작업에 대해 여러 가지 모델을 빠르게 시험해보고 싶을 수도 있습니다.

```py
from transformers import AutoModelForCausalLM

# 동일한 API를 사용하여 3가지 다른 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
```

</hfoption>
<hfoption id="model-specific class">

[AutoModel](./model_doc/auto) 클래스는 모델별 클래스들을 기반으로 구축됩니다. 특정 작업을 지원하는 모든 모델 클래스들은 각각의 `AutoModelFor` 작업 클래스에 매핑됩니다.

이미 사용하려는 모델 클래스를 알고 있다면 해당 모델별 클래스를 직접 사용할 수 있습니다.

```py
from transformers import LlamaModel, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

</hfoption>
</hfoptions>

## 대규모 모델[[large-models]]

대규모 사전 훈련된 모델은 로드하는 데 많은 메모리가 필요합니다. 로드 과정은 다음을 포함합니다:

1. 무작위 가중치로 모델 생성
2. 사전 훈련된 가중치 로드
3. 사전 훈련된 가중치를 모델에 적용

모델 가중치의 복사본 두 가지(무작위 가중치와 사전 훈련된 가중치)를 보관할 수 있는 충분한 메모리가 필요하며, 이는 보유한 하드웨어에 따라 불가능할 수 있습니다. 분산 학습 환경에서는 각 프로세스가 사전 훈련된 모델을 로드하기 때문에 이는 더욱 어려운 과제입니다.

transformers는 빠른 초기화, 분할된 체크포인트, Accelerate의 [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling) 기능, 그리고 더 낮은 비트 데이터 타입 지원을 통해 이러한 메모리 관련 문제들을 일부 줄여줍니다.


### 분할된 체크포인트[[sharded-checkpoints]]

[`~PreTrainedModel.save_pretrained`] 메소드는 10GB보다 큰 체크포인트를 자동으로 샤드합니다.

각 샤드(shard)는 이전 샤드가 로드된 후 순차적으로 로드되어, 메모리 사용량을 모델 크기와 가장 큰 샤드 크기로만 제한합니다.

`max_shard_size` 매개변수는 각 샤드에 대해 기본적으로 5GB로 설정되어 있는데, 이는 메모리 부족 없이 무료 등급 GPU 인스턴스에서 더 쉽게 실행할 수 있기 때문입니다.

예를 들어, [`~PreTrainedModel.save_pretrained`]에서 [BioMistral/BioMistral-7B](https://hf.co/BioMistral/BioMistral-7B)에 대한 분할된 체크포인트를 생성해보겠습니다.

```py
from transformers import AutoModel
import tempfile
import os

model = AutoModel.from_pretrained("biomistral/biomistral-7b")
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    print(sorted(os.listdir(tmp_dir)))
```

[`~PreTrainedModel.from_pretrained`]로 분할된 체크포인트를 다시 로드합니다.

```py
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir)
    new_model = AutoModel.from_pretrained(tmp_dir)
```

분할된 체크포인트는 [`~transformers.modeling_utils.load_sharded_checkpoint`]로도 직접 불러올 수 있습니다.

```py
from transformers.modeling_utils import load_sharded_checkpoint

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    load_sharded_checkpoint(model, tmp_dir)
```

[`~PreTrainedModel.save_pretrained`] 메소드는 매개변수 이름을 저장된 파일에 매핑하는 인덱스 파일을 생성합니다. 인덱스 파일에는 `metadata`와 `weight_map`이라는 두 개의 키가 있습니다.

```py
import json

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print(index.keys())
```

`metadata` 키는 전체 모델 크기를 제공합니다.

```py
index["metadata"]
{'total_size': 28966928384}
```

`weight_map` 키는 각 매개변수를 저장된 샤드에 매핑합니다.

```py
index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

### 대형 모델 추론[[big-model-inference]]

> [!TIP]
> 이 기능을 사용하려면 Accelerate v0.9.0 및 PyTorch v1.9.0 이상이 설치되어 있는지 확인하세요!

<Youtube id="MWCSGj9jEAo"/>

[`~PreTrainedModel.from_pretrained`]는 Accelerate의 [대형 모델 추론](https://hf.co/docs/accelerate/usage_guides/big_modeling) 기능으로 강화되었습니다.

대형 모델 추론은 PyTorch [meta](https://pytorch.org/docs/main/meta.html) 장치에서 *모델 스켈레톤*을 생성합니다. meta 장치는 실제 데이터를 저장하지 않고 메타데이터만 저장합니다.

무작위로 초기화된 가중치는 사전 훈련된 가중치가 로드될 때만 생성되어 메모리에 동시에 모델의 두 복사본을 유지하는 것을 방지합니다. 최대 메모리 사용량은 모델 크기만큼입니다.

> [!TIP]
> 장치 할당에 대한 자세한 내용은 [장치 맵 설계하기](https://hf.co/docs/accelerate/v0.33.0/en/concept_guides/big_model_inference#designing-a-device-map)를 참조하세요.

대형 모델 추론의 두 번째 기능은 불러온 가중치가 모델 스켈레톤에 할당되는 방식과 관련이 있습니다. 모델 가중치는 사용 가능한 모든 디바이스에 분산되며, 가장 빠른 디바이스(보통 GPU)부터 시작해 나머지 가중치는 느린 디바이스(CPU 및 하드 디스크)로 순차적으로 할당됩니다.

두 기능을 결합하면 대형 사전 훈련된 모델의 메모리 사용량과 로딩 시간이 줄어듭니다.

대형 모델 추론을 활성화하려면 [device_map](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3061)을 `"auto"`로 설정합니다.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
```

`device_map`을 사용하여 레이어를 원하는 디바이스에 수동으로 할당할 수도 있습니다. 레이어 전체가 동일한 디바이스에 할당되어 있다면, 해당 레이어의 모든 서브모듈이 어디에 배치되는지 일일이 지정할 필요는 없습니다.

모델이 각 디바이스에 어떻게 분산되어 있는지 확인하려면 `hf_device_map` 속성을 조회합니다.

```py
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
model.hf_device_map
```

### 모델 데이터 타입[[model-data-type]]

PyTorch 모델 가중치는 기본적으로 `torch.float32`로 초기화됩니다. `torch.float16`과 같이 다른 데이터 타입으로 모델을 로드하면 모델이 원하는 데이터 타입으로 다시 로드되기 때문에 추가 메모리가 필요합니다.

[torch_dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) 매개변수를 명시적으로 설정하여 가중치를 두 번 로드하는 대신(`torch.float32` 후 `torch.float16`) 원하는 데이터 타입으로 모델을 직접 초기화하세요. 또한 `torch_dtype="auto"`를 설정하여 가중치를 저장된 데이터 타입으로 자동으로 로드할 수도 있습니다.

<hfoptions id="dtype">
<hfoption id="specific dtype">

```py
import torch
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.float16)
```

</hfoption>
<hfoption id="auto dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype="auto")
```

</hfoption>
</hfoptions>

모델을 처음부터 인스턴스화하는 경우, `torch_dtype` 파라미터는 [`AutoConfig`]에서 설정할 수도 있습니다.

```py
import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", torch_dtype=torch.float16)
model = AutoModel.from_config(my_config)
```

## 커스텀 모델[[custom-models]]

커스텀 모델은 트랜스포머의 구성 및 모델링 클래스를 기반으로 구축되며, [AutoClass](#autoclass) API를 지원하고 [`~PreTrainedModel.from_pretrained`]로 로드됩니다. 차이점은 모델링 코드가 트랜스포머에서 제공되는 것이 *아니라는* 점입니다.

커스텀 모델을 로드할 때는 특별히 주의해야 합니다. Hub에는 모든 저장소에 대한 [악성코드 스캔](https://hf.co/docs/hub/security-malware#malware-scanning)이 포함되어 있지만, 여전히 실수로 악성코드를 실행하지 않도록 주의해야 합니다.

커스텀 모델을 로드하려면 [`~PreTrainedModel.from_pretrained`]에서 `trust_remote_code=True`를 설정하세요.

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

추가적인 보안 조치로, 변경되었을 수도 있는 모델 코드를 로드하는 것을 피하기 위해 특정 리비전에서 커스텀 모델을 로드합니다. 커밋 해시는 모델의 [커밋 기록](https://hf.co/sgugger/custom-resnet50d/commits/main)에서 복사할 수 있습니다.

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

자세한 내용은 [사용자 정의 모델 공유하기](./custom_models) 가이드를 참조하세요.
