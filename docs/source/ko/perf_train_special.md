<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Apple 실리콘에서 Pytorch 학습 [[PyTorch training on Apple silicon]]

이전에는 Mac에서 모델을 학습할 때 CPU만 사용할 수 있었습니다. 그러나 이제 PyTorch v1.12의 출시로 Apple의 실리콘 GPU를 사용하여 훨씬 더 빠른 성능으로 모델을 학습할 수 있게 되었습니다. 이는 Pytorch에서 Apple의 Metal Performance Shaders (MPS)를 백엔드로 통합하면서 가능해졌습니다. [MPS 백엔드](https://pytorch.org/docs/stable/notes/mps.html)는 Pytorch 연산을 Metal 세이더로 구현하고 이 모듈들을 mps 장치에서 실행할 수 있도록 지원합니다.

<Tip warning={true}>

일부 Pytorch 연산들은 아직 MPS에서 지원되지 않아 오류가 발생할 수 있습니다. 이를 방지하려면 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1` 를 설정하여 CPU 커널을 대신 사용하도록 해야 합니다(이때 `UserWarning`이 여전히 표시될 수 있습니다). 

<br>

다른 오류가 발생할 경우 [PyTorch](https://github.com/pytorch/pytorch/issues) 리포지토리에 이슈를 등록해주세요. 현재 [`Trainer`]는 MPS 백엔드만 통합하고 있습니다.

</Tip>

`mps` 장치를 이용하면 다음과 같은 이점들을 얻을 수 있습니다:

* 로컬에서 더 큰 네트워크나 배치 크기로 학습 가능
* GPU의 통합 메모리 아키텍처로 인해 메모리에 직접 접근할 수 있어 데이터 로딩 지연 감소
* 클라우드 기반 GPU나 추가 GPU가 필요 없으므로 비용 절감 가능

Pytorch가 설치되어 있는지 확인하고 시작하세요. MPS 가속은 macOS 12.3 이상에서 지원됩니다.

```bash
pip install torch torchvision torchaudio
```

[`TrainingArguments`]는 `mps` 장치가 사용 가능한 경우 이를 기본적으로 사용하므로 장치를 따로 설정할 필요가 없습니다. 예를 들어, MPS 백엔드를 자동으로 활성화하여 [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) 스크립트를 아무 수정 없이 실행할 수 있습니다.

```diff
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
- --use_mps_device \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

`gloco`와 `nccl`과 같은 [분산 학습 백엔드](https://pytorch.org/docs/stable/distributed.html#backends)는 `mps` 장치에서 지원되지 않으므로, MPS 백엔드에서는 단일 GPU로만 학습이 가능합니다.

Mac에서 가속된 PyTorch 학습에 대한 더 자세한 내용은 [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) 블로그 게시물에서 확인할 수 있습니다.
