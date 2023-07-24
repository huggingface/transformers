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

우리는 최근 텍스트, 이미지 및 오디오 모델에 대한 CPU에서 빠른 추론을 위해 `BetterTransformer`를 통합했습니다. 이 통합에 대해 더 자세한 내용이 담긴 문서는 [여기](https://huggingface.co/docs/optimum/bettertransformer/overview)에서 확인하세요.

## PyTorch JIT 모드 (TorchScript) [[pytorch-jitmode-torchscript]]
TorchScript는 PyTorch 코드에서 직렬화와 최적화 가능한 모델을 생성하는 방법입니다. TorchScript 프로그램은 Python 프로세스에서 저장한 뒤 Python 종속성이 없는 프로세스에서 로드할 수 있습니다. PyTorch의 기본 설정인 eager 모드와 비교했을때, jit 모드는 연산자 결합과 같은 최적화 방법론을 통해 모델 추론에 대해 일반적으로 더 나은 성능을 제공합니다.

TorchScript에 대한 친절한 소개는 [PyTorch TorchScript 튜토리얼](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)을 참조하세요.

### JIT 모드와 함께하는 IPEX 그래프 최적화 [[ipex-graph-optimization-with-jitmode]]
Intel® Extension for PyTorch(IPEX)는 Transformers 계열 모델의 jit 모드에서 추가적인 최적화를 제공합니다. jit 모드에서 Intel® Extension for PyTorch(IPEX)를 활용하는 것을 강력히 권장합니다. Transformers 모델에서 자주 사용되는 일부 연산자 패턴은 이미 jit 모드의 연산자 결합과 함께 Intel® Extension for PyTorch(IPEX)에서 지원되고 있습니다. Multi-head-attention 결합, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm 결합 등의 패턴이 이용 가능하며 성능이 우수합니다. 연산자 결합의 이점은 사용자에게 투명하게 전달됩니다. 분석에 따르면, 질의 응답, 텍스트 분류 및 토큰 분류와 같은 가장 인기 있는 NLP 작업 중 약 70%가 이러한 결합 패턴을 사용하여 Float32 정밀도와 BFloat16 혼합 정밀도 모두에서 성능상의 이점을 얻을 수 있습니다.

[IPEX 그래프 최적화](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)에 대한 자세한 정보를 확인하세요.

#### IPEX 설치: [[ipex-installation]]

IPEX 배포는 PyTorch를 따라 이루어집니다. [IPEX 설치 방법](https://intel.github.io/intel-extension-for-pytorch/)을 확인하세요.

### JIT 모드 사용법 [[usage-of-jitmode]]
평가 또는 예측을 위해 Trainer에서 JIT 모드를 사용하려면 사용자는 Trainer 명령 인수에 `jit_mode_eval`을 추가해야 합니다.

<Tip warning={true}>

PyTorch >= 1.14.0의 경우 jit 모드는 jit.trace에서 dict 입력이 지원되므로, 모든 모델의 예측 및 평가에 이점을 제공할 수 있습니다.

PyTorch < 1.14.0의 경우, 질의 응답 모델과 같이 forward 매개변수의 순서가 jit.trace의 튜플 입력 순서와 일치하는 모델에 이점을 제공할 수 있습니다. 텍스트 분류 모델과 같이 forward 매개변수 순서가 jit.trace의 튜플 입력 순서와 일치하지 않는 경우, jit.trace가 실패하며 예외가 발생합니다. 이를 사용자에게 알리기 위해 Logging이 사용됩니다.

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
