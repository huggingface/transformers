<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU에서 효율적인 추론하기 [[efficient-inference-on-cpu]]

이 가이드는 CPU에서 대규모 모델을 효율적으로 추론하는 방법에 중점을 두고 있습니다.

## 더 빠른 추론을 위한 `BetterTransformer` [[bettertransformer-for-faster-inference]]

우리는 최근 CPU에서 텍스트, 이미지 및 오디오 모델의 빠른 추론을 위해 `BetterTransformer`를 통합했습니다. 이 통합에 대한 더 자세한 내용은 [이 문서](https://huggingface.co/docs/optimum/bettertransformer/overview)를 참조하세요.

## PyTorch JIT 모드 (TorchScript) [[pytorch-jitmode-torchscript]]
TorchScript는 PyTorch 코드에서 직렬화와 최적화가 가능한 모델을 생성할때 쓰입니다. TorchScript로 만들어진 프로그램은 기존 Python 프로세스에서 저장한 뒤, 종속성이 없는 새로운 프로세스로 가져올 수 있습니다. PyTorch의 기본 설정인 `eager` 모드와 비교했을때, `jit` 모드는 연산자 결합과 같은 최적화 방법론을 통해 모델 추론에서 대부분 더 나은 성능을 제공합니다.

TorchScript에 대한 친절한 소개는 [PyTorch TorchScript 튜토리얼](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)을 참조하세요.

### JIT 모드와 함께하는 IPEX 그래프 최적화 [[ipex-graph-optimization-with-jitmode]]
Intel® Extension for PyTorch(IPEX)는 Transformers 계열 모델의 jit 모드에서 추가적인 최적화를 제공합니다. jit 모드와 더불어 Intel® Extension for PyTorch(IPEX)를 활용하시길 강력히 권장드립니다. Transformers 모델에서 자주 사용되는 일부 연산자 패턴은 이미 jit 모드 연산자 결합(operator fusion)의 형태로 Intel® Extension for PyTorch(IPEX)에서 지원되고 있습니다. Multi-head-attention, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm 결합 패턴 등이 이용 가능하며 활용했을 때 성능이 우수합니다. 연산자 결합의 이점은 사용자에게 고스란히 전달됩니다. 분석에 따르면, 질의 응답, 텍스트 분류 및 토큰 분류와 같은 가장 인기 있는 NLP 태스크 중 약 70%가 이러한 결합 패턴을 사용하여 Float32 정밀도와 BFloat16 혼합 정밀도 모두에서 성능상의 이점을 얻을 수 있습니다.

[IPEX 그래프 최적화](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)에 대한 자세한 정보를 확인하세요.

#### IPEX 설치: [[ipex-installation]]

IPEX 배포 주기는 PyTorch를 따라서 이루어집니다. 자세한 정보는 [IPEX 설치 방법](https://intel.github.io/intel-extension-for-pytorch/)을 확인하세요.

### JIT 모드 사용법 [[usage-of-jitmode]]
평가 또는 예측을 위해 Trainer에서 JIT 모드를 사용하려면 Trainer의 명령 인수에 `jit_mode_eval`을 추가해야 합니다.

<Tip warning={true}>

PyTorch의 버전이 1.14.0 이상이라면, jit 모드는 jit.trace에서 dict 입력이 지원되므로, 모든 모델의 예측과 평가가 개선될 수 있습니다.

PyTorch의 버전이 1.14.0 미만이라면, 질의 응답 모델과 같이 forward 매개변수의 순서가 jit.trace의 튜플 입력 순서와 일치하는 모델에 득이 될 수 있습니다. 텍스트 분류 모델과 같이 forward 매개변수 순서가 jit.trace의 튜플 입력 순서와 다른 경우, jit.trace가 실패하며 예외가 발생합니다. 이때 예외상황을 사용자에게 알리기 위해 Logging이 사용됩니다.

</Tip>

[Transformers 질의 응답](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)의 사용 사례 예시를 참조하세요.


- CPU에서 jit 모드를 사용한 추론:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--jit_mode_eval </b></pre> 

- CPU에서 IPEX와 함께 jit 모드를 사용한 추론:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--use_ipex \</b>
<b>--jit_mode_eval</b></pre> 
