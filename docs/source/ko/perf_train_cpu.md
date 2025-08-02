<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU[[cpu]]

최신 CPU는 하드웨어에 내장된 최적화를 활용하고 fp16 또는 bf16 데이터 타입으로 학습을 수행함으로써 대규모 모델 학습을 효율적으로 처리할 수 있습니다.

이 가이드는 Intel CPU에서 혼합 정밀도(mixed precision)를 사용하여 대규모 모델을 학습하는 방법에 중점을 둡니다. PyTorch를 사용하는 CPU 백엔드 학습에서는 AMP가 활성화됩니다.

[`Trainer`]는 `--use_cpu` 및 `--bf16` 파라미터를 추가하여 CPU에서의 AMP 학습을 지원합니다. 아래 예시는 [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 스크립트를 활용한 사용법을 보여줍니다.

```bash
python run_qa.py \
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
 --bf16 \
 --use_cpu
```

 이러한 파라미터들은 아래와 같이 [TrainingArguments]에도 추가할 수 있습니다.

```py
training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_cpu=True,
)
```

## 리소스[[resources]]

Intel CPU에서의 학습에 대해 더 알아보고 싶다면 [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids) 블로그 게시물을 참고하세요.