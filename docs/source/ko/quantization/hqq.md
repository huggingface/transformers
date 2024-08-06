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


# HQQ [[hqq]]

Half-Quadratic Quantization (HQQ)는 빠르고 견고한 최적화를 통해 실시간으로 양자화를 구현합니다. 이 방법은 교정 데이터가 필요 없으며, 어떤 모델이든 양자화를 적용할 수 있습니다.  
자세한 내용은 <a href="https://github.com/mobiusml/hqq/">공식 패키지</a>를 참조하세요.
최신 버전을 설치하고 CUDA 커널을 빌드하려면 다음 방법을 사용하는 것을 추천드립니다.
```
pip install hqq
```

모델을 양자화하려면 먼저 [`HqqConfig`]를 생성해야 합니다. 다음과 같이 두 가지 방법이 있습니다:
``` Python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# 방법 1: 모든 선형 레이어가 동일한 양자화 구성을 사용
quant_config  = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False, axis=0) # axis=0은 기본값
```

``` Python
# 방법 2: 동일한 태그를 가진 각 선형 레이어가 전용 양자화 구성을 사용
q4_config = {'nbits':4, 'group_size':64, 'quant_zero':False, 'quant_scale':False}
q3_config = {'nbits':3, 'group_size':32, 'quant_zero':False, 'quant_scale':False}
quant_config  = HqqConfig(dynamic_config={
  'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
})
```

두 번째 방법은 Mixture-of-Experts (MoEs)를 양자화하는 데 유용합니다. 왜냐하면 experts는 낮은 양자화 설정의 영향을 덜 받기 때문입니다.


그 후, 다음과 같이 모델을 양자화합니다:
``` Python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```

## 최적화된 런타임 [[optimized-runtime]]

HQQ는 순수 Pytorch 및 맞춤형 역양자화(Dequantization) CUDA 커널을 포함한 다양한 백엔드를 지원합니다. 이는 오래된 GPU 및 peft/QLoRA 훈련에 적합합니다. 
HQQ는 4비트 융합 커널(TorchAO 및 Marlin)을 지원하여 더 빠른 추론이 가능하며, 속도는 한 개의 4090 GPU에서 최대 200 토큰/초에 달합니다. ```
백엔드 사용에 대한 자세한 내용은 https://github.com/mobiusml/hqq/?tab=readme-ov-file#backend를 참조하세요.
