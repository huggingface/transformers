<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU에서 효율적인 훈련 [[efficient-training-on-cpu]]

이 가이드는 CPU에서 대규모 모델을 효율적으로 훈련하는 데 초점을 맞춥니다.

## IPEX와 혼합 정밀도 [[mixed-precision-with-ipex]]

IPEX는 AVX-512 이상을 지원하는 CPU에 최적화되어 있으며, AVX2만 지원하는 CPU에도 기능적으로 작동합니다. 따라서 AVX-512 이상의 Intel CPU 세대에서는 성능상 이점이 있을 것으로 예상되지만, AVX2만 지원하는 CPU (예: AMD CPU 또는 오래된 Intel CPU)의 경우에는 IPEX 아래에서 더 나은 성능을 보일 수 있지만 이는 보장되지 않습니다. IPEX는 Float32와 BFloat16를 모두 사용하여 CPU 훈련을 위한 성능 최적화를 제공합니다. BFloat16의 사용은 다음 섹션의 주요 초점입니다.

저정밀도 데이터 타입인 BFloat16은 3세대 Xeon® Scalable 프로세서 (코드명: Cooper Lake)에서 AVX512 명령어 집합을 네이티브로 지원해 왔으며, 다음 세대의 Intel® Xeon® Scalable 프로세서에서 Intel® Advanced Matrix Extensions (Intel® AMX) 명령어 집합을 지원하여 성능을 크게 향상시킬 예정입니다. CPU 백엔드의 자동 혼합 정밀도 기능은 PyTorch-1.10부터 활성화되었습니다. 동시에, Intel® Extension for PyTorch에서 BFloat16에 대한 CPU의 자동 혼합 정밀도 및 연산자의 BFloat16 최적화를 대규모로 활성화하고, PyTorch 마스터 브랜치로 부분적으로 업스트림을 반영했습니다. 사용자들은 IPEX 자동 혼합 정밀도를 사용하여 더 나은 성능과 사용자 경험을 얻을 수 있습니다.

[자동 혼합 정밀도](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html)에 대한 자세한 정보를 확인하십시오.

### IPEX 설치: [[ipex-installation]]

IPEX 릴리스는 PyTorch를 따라갑니다. pip를 통해 설치하려면:

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 1.13              |  1.13.0+cpu    |
| 1.12              |  1.12.300+cpu  |
| 1.11              |  1.11.200+cpu  |
| 1.10              |  1.10.100+cpu  |

```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

[IPEX 설치](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)에 대한 더 많은 접근 방법을 확인하십시오.

### Trainer에서의 사용법 [[usage-in-trainer]]
Trainer에서 IPEX의 자동 혼합 정밀도를 활성화하려면 사용자는 훈련 명령 인수에 `use_ipex`, `bf16`, `no_cuda`를 추가해야 합니다.

[Transformers 질문-응답](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)의 사용 사례를 살펴보겠습니다.

- CPU에서 BF16 자동 혼합 정밀도를 사용하여 IPEX로 훈련하기:
<pre> python run_qa.py \
--model_name_or_path google-bert/bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda</b></pre> 

### 실습 예시 [[practice-example]]

블로그: [Intel Sapphire Rapids로 PyTorch Transformers 가속화](https://huggingface.co/blog/intel-sapphire-rapids)