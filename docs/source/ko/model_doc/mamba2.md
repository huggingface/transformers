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

# 맘바2[[mamba-2]]

## 개요[[overview]]

맘바2 모델은 Tri Dao, Albert Gu가 제안한 [트랜스포머는 SSM이다: 구조화된 상태 공간 이중성을 통한 일반화된 모델과 효율적인 알고리즘](https://arxiv.org/abs/2405.21060)라는 논문에서 소개되었습니다. 맘바2는 맘바1과 유사한 상태 공간 모델로, 단순화된 아키텍처에서 더 나은 성능을 보입니다.

해당 논문의 초록입니다:

*트랜스포머는 언어 모델링에서 딥러닝 성공의 주요 아키텍처였지만, 맘바와 같은 상태 공간 모델(SSM)이 최근 소규모 혹은 중간 규모에서 트랜스포머와 대등하거나 더 나은 성능을 보이는 것으로 나타났습니다. 우리는 이러한 모델 계열들이 실제로 매우 밀접하게 연관되어 있음을 파악했습니다. 그리고 구조화된 준분리(semiseparable) 행렬 중 연구가 잘 이루어진 클래스의 다양한 분해를 통해 연결된 SSM과 어텐션 변형 사이의 풍부한 이론적 연결 프레임워크를 개발했습니다. 상태 공간 이중성(SSD) 프레임워크를 통해 맘바1의 선택적 SSM을 개선한 새로운 아키텍처를 설계할 수 있었고, 트랜스포머와 경쟁력을 유지하면서도 속도는 2~8배 더 빠른 성능을 냅니다.*

팁:

이 버전은 맘바2 구현을 지원해야 하며, 특히 Mistral AI의 [Mamba-2 codestral](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1)을 지원합니다. 특히, mamba 2 codestral은 8개의 `groups`로 출시되었는데, 이는 어텐션 기반 모델의 KV 헤드 수와 유사하다고 판단 가능합니다.

이 모델은 `torch_forward`와 `cuda_kernels_forward`라는 두 가지 다른 전방 패스를 가집니다. `cuda_kernels_forward`는 환경에서 cuda 커널을 찾으면 이를 사용하며, prefill에서는 더 느립니다. 즉, 높은 CPU 오버헤드로 인해 "웜업 실행"이 필요하기 때문입니다. 관련 내용은 [이곳](https://github.com/state-spaces/mamba/issues/389#issuecomment-2171755306)과 [이곳](https://github.com/state-spaces/mamba/issues/355#issuecomment-2147597457)을 참고하세요. 

컴파일 없이는 `torch_forward` 구현이 3~4배 빠릅니다. 또한, 이 모델에는 위치 임베딩이 없지만 `attention_mask`와 배치 생성의 경우 두 곳에서 은닉 상태(hidden state)를 마스킹하는 특정 로직이 있습니다. 관련 내용은 [이곳](https://github.com/state-spaces/mamba/issues/66#issuecomment-1863563829)을 참고하세요. 

이로인해 맘바2 커널의 재구현과 함께 배치 생성 및 캐시된 생성에서 약간의 차이가 예상됩니다. 또한 cuda 커널 또는 torch forward가 제공하는 결과가 약간 다를 것으로 예상됩니다. SSM 알고리즘은 텐서 수축에 크게 의존하는데, 이는 matmul과 동등하지만 연산 순서가 약간 다르며, 이로 인해 더 작은 정밀도에서 차이가 더 커집니다.

또 다른 참고사항으로, 패딩 토큰에 해당하는 은닉 상태(hidden state)의 종료는 두 곳에서 이루어지며 주로 왼쪽 패딩으로 테스트되었습니다. 오른쪽 패딩은 노이즈를 전파하므로 만족스러운 결과를 보장하지 않습니다. `tokenizer.padding_side = "left"`를 사용하면 올바른 패딩 방향을 사용할 수 있습니다.

이 모델은 [Molbap](https://huggingface.co/Molbap)이 기여했으며, [Anton Vlasjuk](https://github.com/vasqu)의 큰 도움을 받았습니다.
원본 코드는 [이곳](https://github.com/state-spaces/mamba)에서 확인할 수 있습니다.


# 사용

### 간단한 생성 예: 
```python 
from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer
import torch
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

이곳은 미세조정을 위한 초안 스크립트입니다: 
```python 
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, Mamba2ForCausalLM, TrainingArguments
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #왼쪽 패딩으로 설정

model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
dataset = load_dataset("Abirate/english_quotes", split="train")
# CUDA 커널없이는, 배치크기 2가 80GB 장치를 하나 차지합니다.
# 하지만 정확도는 감소합니다.
# 실험과 시도를 환영합니다!
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["embeddings", "in_proj", "out_proj"],
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


## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
    - forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
    - forward
