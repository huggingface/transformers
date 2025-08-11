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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Bamba[[bamba]]

[Bamba](https://huggingface.co/blog/bamba)는 [Mamba-2](./mamba2) 아키텍처를 기반으로 구축된 90억 개 매개변수를 가진 디코더 전용 언어 모델입니다. 이 모델은 두 단계로 사전 학습됩니다. 먼저 [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) 데이터 세트에서 2T 토큰으로 학습한 후, [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)과 [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)에서 추가로 200B 토큰으로 학습합니다.

모든 원본 Bamba 체크포인트는 [Bamba](https://huggingface.co/collections/ibm-ai-platform/bamba-674f1388b9bbc98b413c7bab) 컬렉션에서 찾을 수 있습니다.

> [!TIP]
> 이 모델은 [ani300](https://github.com/ani300)과 [fabianlim](https://github.com/fabianlim)에 의해 기여되었습니다.
>
> 다양한 텍스트 생성 작업에 Bamba를 적용하는 방법에 대한 더 많은 예제를 보려면 오른쪽 사이드바의 Bamba 모델을 클릭하세요.

아래 예제는 [`Pipeline`], [`AutoModel`], 그리고 명령줄에서 텍스트를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="ibm-ai-platform/Bamba-9B-v2",
    torch_dtype=torch.bfloat16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-ai-platform/Bamba-9B-v2")
model = AutoModelForCausalLM.from_pretrained("ibm-ai-platform/Bamba-9B-v2", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>

<hfoption id="transformers CLI">
```bash
echo "Plants create energy through a process known as" | transformers-cli run --task text-generation --model ibm-ai-platform/Bamba-9B-v2 --device 0
```
</hfoption>
</hfoptions>

양자화는 가중치를 더 낮은 정밀도로 표현하여 대형 모델의 메모리 부담을 줄입니다. 사용할 수 있는 양자화 백엔드에 대한 자세한 내용은 [양자화](../quantization/overview) 개요를 참조하세요.

아래 예제는 [torchao](../quantization/torchao)를 사용하여 가중치만 int4로 양자화하는 방법을 보여줍니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
tokenizer = AutoTokenizer.from_pretrained("ibm-ai-platform/Bamba-9B-v2")
model = AutoModelForCausalLM.from_pretrained(
   "ibm-ai-platform/Bamba-9B-v2",
   quantization_config=quantization_config,
   device_map="auto",
   attn_implementation="sdpa"
)

inputs = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 참고사항[[notes]]

- Bamba는 별개의 학습 예제들을 연결하면서도 입력을 별도의 배치로 처리하는 패딩 없는 학습을 지원합니다. 이는 패딩 토큰으로 인한 불필요한 계산과 메모리 오버헤드를 피함으로써 다양한 길이의 예제가 있을 때 추론을 [~2배](https://github.com/huggingface/transformers/pull/35861#issue-2807873129) 가속화하고(모델과 데이터 분포에 따라 다름) 메모리 사용량을 줄일 수 있습니다.

  패딩 없는 학습에는 `flash-attn`, `mamba-ssm`, `causal-conv1d` 패키지가 필요하며, `input_ids`와 `labels` 외에 다음 인수를 모델에 전달해야 합니다.

  - `position_ids: torch.LongTensor`: 각 시퀀스에서 각 토큰의 위치 인덱스입니다.
  - `seq_idx: torch.IntTensor`: 배치에서 각 시퀀스의 인덱스입니다.
  - [`FlashAttentionKwargs`]의 각 항목:
    - `cu_seq_lens_q: torch.LongTensor`: 모든 쿼리의 누적 시퀀스 길이입니다.
    - `cu_seq_lens_k: torch.LongTensor`: 모든 키의 누적 시퀀스 길이입니다.
    - `max_length_q: int`: 배치에서 가장 긴 쿼리 길이입니다.
    - `max_length_k: int`: 배치에서 가장 긴 키 길이입니다.

  `attention_mask` 입력은 제공하지 말아야 합니다. [`DataCollatorWithFlattening`]은 `return_seq_idx=True`와 `return_flash_attn_kwargs=True`를 사용하여 위의 추가 인수를 프로그래밍 방식으로 생성합니다. 자세한 정보는 [Flash Attention을 활용한 패킹으로 Hugging Face 학습 효율성 향상](https://huggingface.co/blog/packing-with-FA2) 블로그 포스트를 참조하세요.

  ```python
  from transformers import DataCollatorWithFlattening

  # 패딩 없는 학습 사용 예제
  data_collator = DataCollatorWithFlattening(
      tokenizer=tokenizer,
      return_seq_idx=True,
      return_flash_attn_kwargs=True
  )
  ```

## BambaConfig

[[autodoc]] BambaConfig

## BambaModel

[[autodoc]] BambaModel
    - forward

## BambaForCausalLM

[[autodoc]] BambaForCausalLM
    - forward