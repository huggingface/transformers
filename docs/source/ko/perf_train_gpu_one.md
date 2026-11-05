<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPU[[gpu]]

GPU는 높은 메모리 대역폭과 병렬 처리 능력 덕분에 딥러닝 모델 학습에 널리 사용됩니다. GPU 사양과 모델 크기에 따라 수십억 개 매개변수를 가진 모델도 학습할 수 있습니다. 핵심은 GPU 메모리 활용도(데이터 처리량/학습 시간)와 학습 속도 사이에서 최적의 균형을 찾는 것입니다.

이 가이드는 Transformers와 PyTorch에서 GPU를 활용해 모델을 효율적으로 학습하기 위해 제공하는 기능을 소개합니다. 대부분의 경우, 이 기능들을 조합해서 학습을 최적화하는 것이 좋습니다.

아래 표를 참고하면 자신의 학습 시나리오에 적합한 기능을 빠르게 파악할 수 있습니다.

| 기능                        | 학습 속도 가속 | 메모리 사용량 절약 |
| --------------------------- | --------- | ------------- |
| 배치 크기                   | 예        | 예            |
| 그레이디언트 누적           | 아니요    | 예            |
| 그레이디언트 체크포인팅     | 아니요    | 예            |
| 혼합 정밀도                 | 예        | 조건부        |
| 옵티마이저                  | 예        | 예            |
| 데이터 사전 적재            | 예        | 아니요        |
| torch_empty_cache_steps     | 아니요    | 예            |
| torch.compile               | 예        | 아니요        |
| 스케일된 내적 어텐션 (SDPA) | 예        | 예            |

## Trainer[[trainer]]

Trainer는 [`TrainingArguments`]로 설정할 수 있는 다양한 학습 기능을 제공합니다. 이번 섹션에서는 학습 최적화에 특히 유용한 주요 기능 몇 가지를 살펴봅니다.

### 배치 크기[[batch-size]]

배치 크기는 GPU 학습 효율을 좌우하는 가장 중요한 하이퍼파라미터 중 하나로, 메모리 사용량과 학습 속도에 직접적인 영향을 줍니다. 배치 크기를 크게 하면 GPU의 병렬 처리 능력을 극대화하여 학습 속도를 높일 수 있습니다. 일반적으로 8, 64, 128, 256, 512처럼 2의 거듭제곱 값을 사용하는 것이 좋습니다. 적절한 배치 크기는 GPU 사양과 모델의 데이터 타입에 따라 달라집니다.

배치 크기는 [`TrainingArguments`]의 [`~TrainingArguments.per_device_train_batch_size`] 옵션으로 설정합니다.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
)
```

성능, 입력 피처 수와 출력 뉴런 수, 배치 크기가 성능에 미치는 영향에 대해서는 NVIDIA [Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features) 가이드를 참고하세요. 이 매개변수들은 GPU에서 실행되는 General Matrix Multiplications(GEMMs)에 사용됩니다. 매개변수가 클수록 병렬화와 효율성이 향상됩니다.

데이터 타입과 GPU에 따른 최적의 배치 크기를 선택해 텐서 곱셈 속도를 극대화하려면, [Tensor Core Requirements](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) 섹션을 참고하는 것이 유용합니다. 그 예시로, fp16에서는 8의 배수가 권장되지만, A100 GPU에서는 64의 배수가 더 적합하다는 사실을 확인할 수 있습니다.

마지막으로, 작은 매개변수를 사용할 때는 [Dimension Quantization Effects](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)를 고려하세요. 행렬 차원이 GPU 스레드 블록의 타일 크기로 나누어지지 않으면 타일 양자화가 발생하여 GPU 자원을 충분히 활용하지 못합니다. 행렬이 타일 크기로 정확히 나뉘도록 올바른 배치 크기 배수를 선택하며 학습 속도가 크게 향상됩니다.

### 그레이디언트 누적[[gradient-accumulation]]

그레이디언트 누적은 메모리 제약을 극복하는 방법으로, 단일 GPU에 맞지 않는 매우 큰 모델을 학습할 때 유용합니다. 이는 매개변수를 업데이트하기 전에 여러 미니 배치에 걸쳐 그레이디언트를 누적하는 방식입니다. 그 결과, 저장해야 하는 그레이디언트 수가 줄어 메모리 사용량이 줄어들고, 일반적으로 하나의 배치에서만 매개변수를 갱신하는 방식보다 더 큰 유효 배치 크기로 학습할 수 있습니다. 다만, 추가적인 순전파와 역전파가 필요하기 때문에 학습 속도가 느려질 수 있습니다.

그레이디언트 누적을 활성화하려면 [`TrainingArguments`]에서 [`TrainingArguments.per_device_train_batch_size`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

# 효율적인 배치 크기 64
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```

학습 속도가 느려질 수 있기 때문에 그레이디언트 누적 단계를 너무 크게 설정하지 않는 것이 좋습니다. 아래 예시를 참고하세요, GPU에 담을 수 있는 최대 배치 크기가 4라면 GPU의 효율적인 사용을 위해 배치 크기를 4로 유지하는 것이 좋습니다.

| 배치 크기 | 그레이디언트 누적 단계 | 효율적인 배치 크기 |     |
| --------- | ---------------------- | ------------------ | --- |
| 1         | 64                     | 64                 | 👎  |
| 4         | 16                     | 64                 | 👍  |

### 그레이디언트 체크포인팅[[gradient-checkpointing]]

그레이디언트 체크포인팅은 역전파 과정에서 일부 중간 활성화 값만 저장하고 나머지는 다시 계산해 메모리 사용량을 줄입니다. 이를 통해 순전파 과정에서 모든 중간 활성화 값을 저장하지 않아도 되어 메모리 오버헤드를 크게 줄일 수 있습니다. 다만, 학습 속도가 약 20% 느려지는 한계가 있습니다.

그레이디언트 누적을 활성화하려면 [`TrainingArguments`]에서 [`~TrainingArguments.gradient_checkpointing`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
)
```

### 혼합 정밀도[[mixed-precision]]

혼합 정밀도는 일부 계산을 반정밀도(fp16)로, 나머지를 전정밀도(fp32)로 수행해 학습 속도를 높이는 기법입니다. 반정밀도 계산은 전정밀도보다 계산량이 적어 더 빠르게 수행됩니다. 한편, 전정밀도로 일부 계산을 수행하면 정확도를 유지할 수 있습니다.

혼합 정밀도 학습을 위해 여러 자료형을 사용할 수 있습니다.

<hfoptions id="mixed-precision">
<hfoption id="fp16">

혼합 정밀도 학습의 주요 장점은 활성화 값을 fp16으로 저장할 수 있다는 것입니다.

fp16 자료형으로 혼합 정밀도 학습을 활성화하려면 [`TrainingArguments`]에서 [`~TrainingArguments.fp16`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True.
)
```

fp16은 메모리 사용에 최적화된 방식이 아닙니다. 이는 fp16으로 계산된 그레이디언트가 최적화 단계에서 fp32로 다시 변환되기 때문입니다. 특히 배치 크기가 작을 때는, GPU에 두 가지 자료형(fp16, fp32)이 적재되어 있기 때문에 더 많은 GPU 메모리를 사용하게 됩니다.
</hfoption>
<hfoption id="bf16">

[bf16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)은 일부 정밀도를 포기하는 대신, 훨씬 더 넓은 동적 범위를 제공하여 오버플로와 언더플로 오류를 방지하는 데 도움이 됩니다. bf16은 fp16과 달리 손실 스케일링 기법을 추가하지 않고도 사용할 수 있습니다. bf16은 NVIDIA의 Ampere 아키텍처 이상에서 지원됩니다.

bf16 자료형으로 혼합 정밀도 학습을 활성화하려면 [`TrainingArguments`]에서 [`~TrainingArguments.bf16`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
)
```

</hfoption>
<hfoption id="tf32">

[tf32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)는 NVIDIA Ampere GPU에서 합성곱과 행렬곱 입력을 tf32로 변환하는 모드입니다. 다른 모든 저장과 연산은 fp32로 유지됩니다. 이를 통해 tf32는 fp32와 동일한 범위, fp16과 동일한 정밀도, 그리고 bf16보다 더 높은 정밀도를 유지할 수 있습니다. tf32를 fp16 또는 bf16 혼합 정밀도 학습과 결합하면 처리량을 16배 향상할 수 있습니다.

tf32는 NVIDIA Ampere GPU에서 기본적으로 활성화되어 있지만, fp32 학습 또는 추론 코드에 아래 코드를 추가하여 명시적으로 활성화할 수도 있습니다.

```py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

tf32 모드에서 혼합 정밀도 학습을 활성화하려면 [`TrainingArguments`]에서 [tf32()](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.tf32) 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True.
    tf32=True,
)
```

</hfoption>
</hfoptions>

### 옵티마이저[[optimizers]]

Transformers는 기본적으로 PyTorch의 [AdamW (adamw_torch)](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 옵티마이저를 사용합니다. 하지만, 이 옵티마이저는 과거 그레이디언트의 가중 평균을 저장하기 때문에, 그레이디언트를 저장하기 위해 모델 매개변수 수에 비례한 추가 메모리가 필요합니다. 이는 매우 큰 모델을 학습할 때 문제가 될 수 있으며, 이러면 다른 옵티마이저를 선택하는 것을 고려해야 합니다. 예를 들어, [NVIDIA](https://github.com/NVIDIA/apex) 또는 [AMD](https://github.com/ROCm/apex)에 [Apex](https://nvidia.github.io/apex/index.html)가 설치되어 있다면, 모든 AdamW 옵티마이저 중 `adamw_apex_fused` 옵티마이저를 사용하는 것이 가장 빠른 학습 속도를 얻을 수 있습니다.

옵티마이저를 선택하기 위해서는 [`TrainingArguments`]에서 [`~TrainingArguments.optim`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit"
)
```
학습 시나리오에 맞게 선택할 수 있는 다양한 옵티마이저가 있습니다. (전체 지원 목록은 [OptimizerNames](https://github.com/huggingface/transformers/blob/34f4080ff59b1668d919a1ba9f8bc4a3a2a3f478/src/transformers/training_args.py#L145)를 참고하세요) 예를 들어 Adafactor는 행렬의 각 요소 대신 행 또는 열 단위의 가중 평균만 저장해 메모리 요구량을 크게 줄일 수 있지만, 수렴 속도는 느려질 수 있습니다. 또 다른 예로, bitandbytes의 [8-bit AdamW optimizer](https://huggingface.co/docs/bitsandbytes)를 사용하면 옵티마이저의 상태를 8비트로 양자화할 수 있습니다. 옵티마이저 상태는 낮은 정밀도로 저장되었다가 옵티마이저 단계에서 사용되기 전에 역 양자화됩니다.

특화된 옵티마이저에 대해 더 알고 싶다면 [optimizer](./optimizers) 가이드를 참고하세요.

### 데이터 사전 적재[[data-preloading]]

데이터 사전 적재(Data preloading)는 GPU가 지속적으로 작업할 수 있도록 CPU에서 미리 배치 단위의 데이터를 적재하고 준비하는 기능입니다. 이를 통해 GPU 유휴 시간을 줄이고 활용도를 높일 수 있습니다. GPU가 항상 작업을 계속하도록 하려면 다음 데이터 사전 적재를 위한 두 가지 방법을 사용할 수 있습니다.

1. 데이터를 저장할 고정 메모리를 CPU에 할당한 뒤, 이를 GPU로 직접 전송합니다.
2. CPU 스레드 및 워커 수를 늘려 데이터를 더 빠르게 사전 적재합니다.

고정 메모리를 할당하고 워커 수를 늘리기 위해서는 [`TrainingArguments`]에서 [`~TrainingArguments.dataloader_pin_memory`]와 [`~TrainingArguments.dataloader_num_workers`] 옵션을 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)
```

## PyTorch[[pytorch]]

PyTorch는 메모리 요구사항을 줄이고 학습 속도를 높이기 위한 여러 기능을 제공합니다. 이러한 기능들은 Transformers에서 몇 줄의 코드만 추가하여 활성화할 수 있습니다.

### torch.empty_cache_steps[[torchemptycachesteps]]

[torch.cuda.empty_cache](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache) 함수는 사용하지 않는 캐시 메모리를 해제하여 메모리 부족(OOM) 오류를 방지할 수 있지만, 학습 속도가 약 10% 느려질 수 있습니다.

특정 학습 단계 이후에 이 기능을 활성화하고 싶다면, [`TrainingArguments`]에서 [torch_empty_cache_steps()](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.torch_empty_cache_steps)를 설정하세요.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
)
```

### torch.compile[[torchcompile]]

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)은 PyTorch 코드를 최적화된 커널로 컴파일해 학습 속도를 크게 높여줍니다. 이 기능은 TorchDynamo를 사용해 프레임 평가 API로부터 PyTorch 그래프를 캡처하며, 이렇게 캡처한 그래프는 다양한 백엔드에 추가로 최적화된 커널로 컴파일될 수 있습니다.

이를 활성화하려면 [`TrainingArguments`]에서 [`~TrainingArguments.torch_compile`]를 설정하세요. 백엔드는 [torch_compile_backend()](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.torch_compile_backend)를 통해 선택할 수 있습니다.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
    torch_compile=True,
    torch_compile_backend="inductor"
)
```

아래 표를 참고하여 학습 시나리오에 적합한 백엔드를 선택하세요.

| 백엔드         | 설명                                                                                                                                                                   | 목표         |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| eager          | PyTorch를 사용해 추출된 GraphModule을 실행합니다                                                                                                                       | 디버깅       |
| aot_eager      | AOTAutograd로 추출된 순전파 및 역전파 그래프를 Pytorch eager 모드로 실행합니다                                                                                         | 디버깅       |
| inductor       | Triton 커널을 활용하는 TorchInductor와 AOTAutograd, CUDA Graphs를 사용합니다                                                                                           | 학습 및 추론 |
| nvfuser        | TorchScript와 함께 nvFuser를 사용합니다                                                                                                                                | 학습 및 추론 |
| aot_nvfuser    | AOTAutograd와 함께 nvFuser를 사용합니다                                                                                                                                | 학습 및 추론 |
| aot_cudagraphs | AOTAutograd와 함께 CUDA Graphs를 사용합니다                                                                                                                            | 학습 및 추론 |
| ofi            | TorchScripts의 [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch-jit-optimize-for-inference)를 사용합니다 | 추론         |
| fx2trt         | [Torch-TensorRT](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html)를 사용합니다                                                                | 추론         |
| onnxrt         | CPU 및 GPU 추론을 위해 [ONNX-RT](https://onnxruntime.ai/)를 사용합니다                                                                                                 | 추론         |
| ipex           | CPU 추론을 위해 [IPEX](https://github.com/intel/intel-extension-for-pytorch)를 사용합니다                                                                              | 추론         |

### 스케일된 내적 어텐션[[scaled-dot-production-attention]]

[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA)는 스케일된 내적 어텐션 메커니즘을 PyTorch에 내장해 구현한 함수입니다. SDPA는 트랜스포머 모델의 기존 어텐션 메커니즘보다 더 효율적이고 최적화되어 있습니다. 세 가지 유형의 스케일된 내적 어텐션을 지원합니다.

- [FlashAttention2](https://github.com/Dao-AILab/flash-attention)는 fp16 또는 bf16 torch 타입 모델에서 자동으로 활성화됩니다. 먼저 모델을 적절한 타입으로 캐스팅했는지 확인하세요.
- [xFormers](https://github.com/facebookresearch/xformers) 또는 Memory-Efficient Attention은 fp32 torch 타입 모델을 지원합니다.
- C++로 구현된 스케일된 내적 어텐션입니다.

SDPA는 PyTorch 2.1.1 버전 이상에서 기본적으로 활성화되어 있지만, [`~PreTrainedModel.from_pretrained`]에서 `attn_implementation="sdpa"`를 설정해 명시적으로 활성화할 수 있습니다.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", attn_implementation="sdpa")
```
