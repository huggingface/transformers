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

# GPTQ [[gptq]]

<Tip>

PEFT를 활용한 GPTQ 양자화를 사용해보시려면 이 [노트북](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb)을 참고하시고, 자세한 내용은 이 [블로그 게시물](https://huggingface.co/blog/gptq-integration)에서 확인하세요!

</Tip>

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 라이브러리는 GPTQ 알고리즘을 구현합니다. 이는 훈련 후 양자화 기법으로, 가중치 행렬의 각 행을 독립적으로 양자화하여 오차를 최소화하는 가중치 버전을 찾습니다. 이 가중치는 int4로 양자화되지만, 추론 중에는 실시간으로 fp16으로 복원됩니다. 이는 int4 가중치가 GPU의 전역 메모리 대신 결합된 커널에서 역양자화되기 때문에 메모리 사용량을 4배 절약할 수 있으며, 더 낮은 비트 너비를 사용함으로써 통신 시간이 줄어들어 추론 속도가 빨라질 것으로 기대할 수 있습니다.

시작하기 전에 다음 라이브러리들이 설치되어 있는지 확인하세요:

```bash
pip install auto-gptq
pip install --upgrade accelerate optimum transformers
```

모델을 양자화하려면(현재 텍스트 모델만 지원됨) [`GPTQConfig`] 클래스를 생성하고 양자화할 비트 수, 양자화를 위한 가중치 교정 데이터셋, 그리고 데이터셋을 준비하기 위한 토크나이저를 설정해야 합니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

자신의 데이터셋을 문자열 리스트 형태로 전달할 수도 있지만, GPTQ 논문에서 사용한 동일한 데이터셋을 사용하는 것을 강력히 권장합니다.

```py
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

양자화할 모델을 로드하고 `gptq_config`을 [`~AutoModelForCausalLM.from_pretrained`] 메소드에 전달하세요. 모델을 메모리에 맞추기 위해 `device_map="auto"`를 설정하여 모델을 자동으로 CPU로 오프로드하고, 양자화를 위해 모델 모듈이 CPU와 GPU 간에 이동할 수 있도록 합니다.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```

데이터셋이 너무 커서 메모리가 부족한 경우를 대비한 디스크 오프로드는 현재 지원하지 않고 있습니다. 이럴 때는 `max_memory` 매개변수를 사용하여 디바이스(GPU 및 CPU)에서 사용할 메모리 양을 할당해 보세요:

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"}, quantization_config=gptq_config)
```

<Tip warning={true}>

하드웨어와 모델 매개변수량에 따라 모델을 처음부터 양자화하는 데 드는 시간이 서로 다를 수 있습니다. 예를 들어, 무료 등급의 Google Colab GPU로 비교적 가벼운 [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) 모델을 양자화하는 데 약 5분이 걸리지만, NVIDIA A100으로 175B에 달하는 매개변수를 가진 모델을 양자화하는 데는 약 4시간에 달하는 시간이 걸릴 수 있습니다. 모델을 양자화하기 전에, Hub에서 해당 모델의 GPTQ 양자화 버전이 이미 존재하는지 확인하는 것이 좋습니다.

</Tip>

모델이 양자화되면, 모델과 토크나이저를 Hub에 푸시하여 쉽게 공유하고 접근할 수 있습니다. [`GPTQConfig`]를 저장하기 위해 [`~PreTrainedModel.push_to_hub`] 메소드를 사용하세요:

```py
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

양자화된 모델을 로컬에 저장하려면 [`~PreTrainedModel.save_pretrained`] 메소드를 사용할 수 있습니다. 모델이 `device_map` 매개변수로 양자화되었을 경우, 저장하기 전에 전체 모델을 GPU나 CPU로 이동해야 합니다. 예를 들어, 모델을 CPU에 저장하려면 다음과 같이 합니다:

```py
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

# device_map이 설정된 상태에서 양자화된 경우
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

양자화된 모델을 다시 로드하려면 [`~PreTrainedModel.from_pretrained`] 메소드를 사용하고, `device_map="auto"`를 설정하여 모든 사용 가능한 GPU에 모델을 자동으로 분산시켜 더 많은 메모리를 사용하지 않으면서 모델을 더 빠르게 로드할 수 있습니다.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

## ExLlama [[exllama]]

[ExLlama](https://github.com/turboderp/exllama)은 [Llama](model_doc/llama) 모델의 Python/C++/CUDA 구현체로, 4비트 GPTQ 가중치를 사용하여 더 빠른 추론을 위해 설계되었습니다(이 [벤치마크](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark)를 참고하세요). ['GPTQConfig'] 객체를 생성할 때 ExLlama 커널이 기본적으로 활성화됩니다. 추론 속도를 더욱 높이기 위해, `exllama_config` 매개변수를 구성하여 [ExLlamaV2](https://github.com/turboderp/exllamav2) 커널을 사용할 수 있습니다:

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=gptq_config)
```

<Tip warning={true}>

4비트 모델만 지원되며, 양자화된 모델을 PEFT로 미세 조정하는 경우 ExLlama 커널을 비활성화할 것을 권장합니다.

</Tip>

ExLlama 커널은 전체 모델이 GPU에 있을 때만 지원됩니다. AutoGPTQ(버전 0.4.2 이상)로 CPU에서 추론을 수행하는 경우 ExLlama 커널을 비활성화해야 합니다. 이를 위해 config.json 파일의 양자화 설정에서 ExLlama 커널과 관련된 속성을 덮어써야 합니다.

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig
gptq_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="cpu", quantization_config=gptq_config)
```