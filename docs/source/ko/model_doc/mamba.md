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

# 맘바[[mamba]]

## 개요[[overview]]

맘바(Mamba) 모델은 Albert Gu, Tri Dao가 제안한 [맘바: 선택적 상태 공간을 이용한 선형 시간 시퀀스 모델링](https://arxiv.org/abs/2312.00752)라는 논문에서 소개 되었습니다.

이 모델은 `state-space-models`을 기반으로 한 새로운 패러다임 아키텍처입니다. 직관적인 이해를 얻고 싶다면 [이곳](https://srush.github.io/annotated-s4/)을 참고 하세요.

해당 논문의 초록입니다:

*현재 딥러닝에서 흥미로운 응용 프로그램을 구동하는 대부분의 기초 모델들은 거의 보편적으로 트랜스포머 아키텍처와 그 핵심 어텐션 모듈을 기반으로 합니다. 선형 어텐션, 게이트된 컨볼루션과 순환 모델, 구조화된 상태 공간 모델(SSM) 등 많은 준이차시간(subquadratic-time) 아키텍처가 긴 시퀀스에 대한 트랜스포머의 계산 비효율성을 해결하기 위해 개발되었지만, 언어와 같은 중요한 양식에서는 어텐션만큼 성능을 내지 못했습니다. 우리는 이러한 모델의 주요 약점이 내용 기반 추론을 수행하지 못한다는 점임을 알고 몇 가지를 개선했습니다. 첫째, SSM 매개변수를 입력의 함수로 만드는 것만으로도 이산 모달리티(discrete modalities)의 약점을 해결할 수 있어, 현재 토큰에 따라 시퀀스 길이 차원을 따라 정보를 선택적으로 전파하거나 잊을 수 있게 합니다. 둘째, 이러한 변경으로 효율적인 컨볼루션을 사용할 수 없게 되었지만, 우리는 순환 모드에서 하드웨어를 인식하는 병렬 알고리즘을 설계했습니다. 우리는 이러한 선택적 SSM을 어텐션이나 MLP 블록도 없는 단순화된 종단간 신경망 아키텍처인 맘바에 통합시켰습니다. 맘바는 빠른 추론(트랜스포머보다 5배 높은 처리량)과 시퀀스 길이에 대한 선형 확장성을 누리며, 백만 길이 시퀀스까지 실제 데이터에서 성능이 향상됩니다. 일반적인 시퀀스 모델 백본으로서 맘바는 언어, 오디오, 유전체학과 같은 여러 양식에서 최첨단 성능을 달성합니다. 언어 모델링에서 우리의 맘바-3B 모델은 같은 크기의 트랜스포머를 능가하고 두 배 크기의 트랜스포머와 맞먹는 성능을 보이며, 사전 훈련과 다운스트림 평가 모두에서 성능을 나타납니다.*

팁:

- 맘바는 고전적인 트랜스포머와 견줄 만한 새로운 `상태 공간 모델` 아키텍처입니다. 이는 구조화된 상태 공간 모델의 발전 선상에 있으며, [플래시어텐션](https://github.com/Dao-AILab/flash-attention)의 정신을 따르는 효율적인 하드웨어 인식 설계와 구현을 특징으로 합니다.
- 맘바는 `어텐션` 레이어와 동등한 `믹서(mixer)` 레이어를 쌓습니다. `맘바`의 핵심 로직은 `MambaMixer` 클래스에 있습니다.
- 두 가지 구현이 공존합니다: 하나는 최적화되어 빠른 cuda커널을 사용하고, 다른 하나는 단순하지만 모든 장치에서 실행할 수 있습니다!
- 현재 구현은 원본 cuda커널을 활용합니다: 맘바를 위한 플래시 어텐션의 역할을 하는 것은 [`mamba-ssm`](https://github.com/state-spaces/mamba)와 [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d) 저장소에 호스팅되어 있습니다. 하드웨어가 지원한다면 반드시 설치하세요!
- cuda 커널을 최적화하는 방향 보다는, 단순하지만 모든 장치에서 실행가능하도록하는 방향인 '단순구현'의 성능을 빠르게 향상시키는 기여를 더 환영하고 있습니다. 🤗

이 모델은 [ArthurZ](https://huggingface.co/ArthurZ)에 의해 기여되었습니다.
원본 코드는 [이곳](https://github.com/state-spaces/mamba)에서 확인할 수 있습니다.

# 사용

### 간단한 생성 예제
 
```python 
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

### Peft 파인튜닝
느린 버전은 학습에서 아주 안정적이진 않습니다. 빠른 버전은 `float32`가 필요합니다!

```python 
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()
```

## MambaConfig

[[autodoc]] MambaConfig

## MambaModel

[[autodoc]] MambaModel
    - forward

## MambaLMHeadModel

[[autodoc]] MambaForCausalLM
    - forward
