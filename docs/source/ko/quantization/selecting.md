<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ 이 파일은 Markdown이지만 문서 빌더용 특수 문법(일부는 MDX와 유사)을 포함합니다. 일반적인 Markdown 뷰어에서는 올바르게 렌더링되지 않을 수 있습니다.
-->

# 양자화 방법 선택 가이드

Transformers에는 추론과 파인튜닝을 위한 다양한 양자화 방법이 있습니다. 이 가이드는 사용 사례에 따라 가장 일반적이고 운영 환경에서 바로 사용할 수 있는 된 양자화 기법을 선택하는 데 도움을 주고, 각 기법의 장단점을 소개합니다.

지원되는 모든 방법과 기능을 포괄적으로 보려면 [개요](./overview)의 표를 참고하세요.

## 추론(Inference)

추론 용도로는 아래 양자화 방법을 고려하세요.

| 양자화 방법             | 사용 사례                                  |
| ------------------ | -------------------------------------- |
| bitsandbytes       | 쉬운 사용성과 NVIDIA/Intel GPU에서의 QLoRA 파인튜닝 |
| compressed-tensors | 특정 양자화 형식(FP8, 희소성 등) 로딩               |
| GPTQModel 또는 AWQ   | 사전 보정(calibration) 기반의 우수한 4비트 정확도     |
| HQQ                | 보정 없이 빠른 온더플라이 양자화                     |
| torchao            | `torch.compile`과 함께 유연하고 빠른 추론         |

### 보정 불필요(온더플라이 양자화)

이 방법들은 별도의 보정 데이터셋이나 단계가 필요 없어 대체로 사용이 간편합니다.

#### bitsandbytes

| 장점                           | 단점                         |
| ---------------------------- | -------------------------- |
| 매우 간단하며, 추론에 보정 데이터셋이 필요 없음. | 주로 NVIDIA GPU(CUDA)에 최적화됨. |
| 커뮤니티 지원이 좋고 널리 사용됨.          | 추론 가속이 항상 보장되지는 않음.        |

자세한 내용은 [bitsandbytes 문서](./bitsandbytes)를 참고하세요.

#### HQQ (Half-Quadratic Quantization)

| 장점                           | 단점                                                   |
| ---------------------------- | ---------------------------------------------------- |
| 보정 데이터 없이도 빠른 양자화.           | 4비트 미만에서는 정확도가 크게 저하될 수 있음.                          |
| 빠른 추론을 위한 다수 백엔드 제공.         | `torch.compile` 또는 특정 백엔드를 쓰지 않으면 속도가 기대에 못 미칠 수 있음. |
| `torch.compile`과 호환.         |                                                      |
| 폭넓은 비트폭(8, 4, 3, 2, 1비트) 지원. |                                                      |

자세한 내용은 [HQQ 문서](./hqq)를 참고하세요.

#### torchao

| 장점                                   | 단점                                     |
| ------------------------------------ | -------------------------------------- |
| `torch.compile`과 강하게 통합되어 속도 향상 잠재력. | 비교적 새로운 라이브러리로 생태계가 아직 성장 중.           |
| CPU 양자화 지원이 준수함.                     | 성능이 `torch.compile`의 성과에 좌우될 수 있음.     |
| 다양한 양자화 스킴(int8, int4, fp8) 제공.      | 4비트(int4wo)의 정확도는 GPTQ/AWQ에 못 미칠 수 있음. |

자세한 내용은 [torchao 문서](./torchao)를 참고하세요.

### 보정 기반 양자화

이 방법들은 더 높은 정확도를 위해 데이터셋을 사용한 사전 보정 단계가 필요합니다.

#### GPTQ/GPTQModel

8B 모델 보정은 A100 GPU 1장으로 약 20분 소요됩니다.

| 장점                                                                                    | 단점                      |
| ------------------------------------------------------------------------------------- | ----------------------- |
| 높은 정확도 달성 사례가 많음.                                                                     | 보정 데이터셋과 별도의 보정 단계가 필요. |
| 추론 가속으로 이어질 수 있음.                                                                     | 보정 데이터에 과적합될 가능성.       |
| 사전 양자화된 GPTQ 모델이 [Hugging Face Hub](https://huggingface.co/models?other=gptq)에 다수 존재. |                         |

자세한 내용은 [GPTQ 문서](./gptq)를 참고하세요.

#### AWQ (Activation-aware Weight Quantization)

8B 모델 보정은 A100 GPU 1장으로 약 10분 소요.

| 장점                                                                                  | 단점              |
| ----------------------------------------------------------------------------------- | --------------- |
| 4비트에서 높은 정확도(특정 작업에서는 GPTQ를 능가하기도 함).                                               | 자체 양자화 시 보정 필요. |
| 추론 가속으로 이어질 수 있음.                                                                   |                 |
| GPTQ보다 보정 시간이 짧은 편.                                                                 |                 |
| 사전 양자화된 AWQ 모델이 [Hugging Face Hub](https://huggingface.co/models?other=awq)에 다수 존재. |                 |

자세한 내용은 [AWQ 문서](./awq)를 참고하세요.

### 특정 형식 로딩

#### compressed-tensors

| 장점                     | 단점                                  |
| ---------------------- | ----------------------------------- |
| FP8 및 희소성 등 유연한 형식 지원. | 주로 사전 양자화된 모델 로딩용.                  |
|                        | Transformers 내부에서 직접 양자화를 수행하지는 않음. |

자세한 내용은 [compressed-tensors 문서](./compressed_tensors)를 참고하세요.

## 파인튜닝(Fine-tuning)

메모리 절약을 위해 파인튜닝에서는 아래 양자화 방법을 고려하세요.

### bitsandbytes[[training]]

* **설명:** PEFT를 통한 QLoRA 파인튜닝의 표준 방법입니다.
* **장점:** 대형 모델을 소비자용 GPU에서도 파인튜닝 가능하고 PEFT 측에서 널리 지원·문서화되어 있습니다.
* **단점:** 주로 NVIDIA GPU 대상으로 최적화되어 있습니다.

다른 방법들도 PEFT 호환성을 제공하지만, bitsandbytes가 QLoRA 경로로는 가장 정립되어 있고 간단합니다.

자세한 내용은 [bitsandbytes 문서](./bitsandbytes#qlora)와 [PEFT 문서](https://huggingface.co/docs/peft/developer_guides/quantization#aqlm-quantization)를 참고하세요.

## 연구(Research)

[AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs) 등의 방법은 2비트 미만의 극단적 압축이나 새로운 기법을 탐구합니다.

* 다음과 같은 경우 고려할 수 있습니다:

  * 4비트 미만의 극한 압축이 필요할 때
  * 연구 목적이거나 각 논문에서 제시하는 최신 결과가 필요한 경우
  * 복잡한 양자화 절차에 필요한 충분한 컴퓨트 자원이 있을 때

프로덕션 사용 전에는 각 방법의 문서와 관련 논문을 반드시 면밀히 검토할 것을 권장합니다.

## 벤치마크 비교

여러 인기 양자화 기법을 Llama 3.1 8B 및 70B 모델에서 벤치마크하여 정량 비교를 제공합니다. 아래 표는 정확도(높을수록 좋음), 토큰/초로 측정한 추론 처리량(높을수록 좋음), 피크 VRAM 사용량(GB, 낮을수록 좋음), 양자화 시간을 보여줍니다.

Llama 3.1 70B(bfloat16)는 NVIDIA A100 80GB 2장, FP8 방법은 NVIDIA H100 80GB 1장, 그 외는 NVIDIA A100 80GB 1장에서 측정했습니다. 처리량은 배치 크기 1, 64 토큰 생성을 기준으로 측정했습니다.
적용 가능한 곳에는 `torch.compile`과 Marlin 커널 결과도 포함했습니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

핵심 요약은 다음과 같습니다.

| 양자화 & 방법                                      | 메모리 절감(대비 bf16) | 정확도             | 기타 메모                                                  |
| --------------------------------------------- | --------------- | --------------- | ------------------------------------------------------ |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8) | 약 2배            | bf16 기준선과 매우 유사 |                                                        |
| **4비트** (AWQ, GPTQ, HQQ, bnb-nf4)             | 약 4배            | 비교적 높은 정확도      | AWQ/GPTQ는 정확도 선도 사례 많으나 보정 필요. HQQ/bnb-nf4는 온더플라이로 간편. |
| **4비트 미만** (VPTQ, AQLM, 2-bit GPTQ)           | 극단적(>4배)        | 특히 2비트에서 뚜렷한 하락 | AQLM, VPTQ는 양자화 시간이 매우 길 수 있음. 성능 편차 큼.                |

> [!TIP]
> 양자화된 모델의 성능(정확도와 속도)은 항상 여러분의 **특정 작업과 하드웨어**에서 벤치마크하세요. 자세한 사용법은 위에 연결한 각 방법의 문서 페이지를 참고하세요.
