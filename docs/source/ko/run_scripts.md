<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 스크립트로 실행하기[[train-with-a-script]]

🤗 Transformers 노트북과 함께 [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow), 또는 [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)를 사용해 특정 태스크에 대한 모델을 훈련하는 방법을 보여주는 예제 스크립트도 있습니다.

또한 [연구 프로젝트](https://github.com/huggingface/transformers-research-projects/) 및 [레거시 예제](https://github.com/huggingface/transformers/tree/main/examples/legacy)에서 대부분 커뮤니티에서 제공한 스크립트를 찾을 수 있습니다.
이러한 스크립트는 적극적으로 유지 관리되지 않으며 최신 버전의 라이브러리와 호환되지 않을 가능성이 높은 특정 버전의 🤗 Transformers를 필요로 합니다.

예제 스크립트가 모든 문제에서 바로 작동하는 것은 아니며, 해결하려는 문제에 맞게 스크립트를 변경해야 할 수도 있습니다.
이를 위해 대부분의 스크립트에는 데이터 전처리 방법이 나와있어 필요에 따라 수정할 수 있습니다.

예제 스크립트에 구현하고 싶은 기능이 있으면 pull request를 제출하기 전에 [포럼](https://discuss.huggingface.co/) 또는 [이슈](https://github.com/huggingface/transformers/issues)에서 논의해 주세요.
버그 수정은 환영하지만 가독성을 희생하면서까지 더 많은 기능을 추가하는 pull request는 병합(merge)하지 않을 가능성이 높습니다.

이 가이드에서는 [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) 및 [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization)에서 요약 훈련하는
 스크립트 예제를 실행하는 방법을 설명합니다.
특별한 설명이 없는 한 모든 예제는 두 프레임워크 모두에서 작동할 것으로 예상됩니다.

## 설정하기[[setup]]

최신 버전의 예제 스크립트를 성공적으로 실행하려면 새 가상 환경에서 **소스로부터 🤗 Transformers를 설치**해야 합니다:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

이전 버전의 예제 스크립트를 보려면 아래 토글을 클릭하세요:

<details>
  <summary>이전 버전의 🤗 Transformers 예제</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

그리고 다음과 같이 복제(clone)해온 🤗 Transformers 버전을 특정 버전(예: v3.5.1)으로 전환하세요:

```bash
git checkout tags/v3.5.1
```

올바른 라이브러리 버전을 설정한 후 원하는 예제 폴더로 이동하여 예제별로 라이브러리에 대한 요구 사항(requirements)을 설치합니다:

```bash
pip install -r requirements.txt
```

## 스크립트 실행하기[[run-a-script]]

예제 스크립트는 🤗 [Datasets](https://huggingface.co/docs/datasets/) 라이브러리에서 데이터 세트를 다운로드하고 전처리합니다.
그런 다음 스크립트는 요약 기능을 지원하는 아키텍처에서 [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)를 사용하여 데이터 세트를 미세 조정합니다.
다음 예는 [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) 데이터 세트에서 [T5-small](https://huggingface.co/google-t5/t5-small)을 미세 조정합니다.
T5 모델은 훈련 방식에 따라 추가 `source_prefix` 인수가 필요하며, 이 프롬프트는 요약 작업임을 T5에 알려줍니다.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## 혼합 정밀도(mixed precision)로 분산 훈련하기[[distributed-training-and-mixed-precision]]

[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 클래스는 분산 훈련과 혼합 정밀도(mixed precision)를 지원하므로 스크립트에서도 사용할 수 있습니다.
이 두 가지 기능을 모두 활성화하려면 다음 두 가지를 설정해야 합니다:

- `fp16` 인수를 추가해 혼합 정밀도(mixed precision)를 활성화합니다.
- `nproc_per_node` 인수를 추가해 사용할 GPU 개수를 설정합니다.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

TensorFlow 스크립트는 분산 훈련을 위해 [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)를 활용하며, 훈련 스크립트에 인수를 추가할 필요가 없습니다.
다중 GPU 환경이라면, TensorFlow 스크립트는 기본적으로 여러 개의 GPU를 사용합니다.

## TPU 위에서 스크립트 실행하기[[run-a-script-on-a-tpu]]

Tensor Processing Units (TPUs)는 성능을 가속화하기 위해 특별히 설계되었습니다.
PyTorch는 [XLA](https://www.tensorflow.org/xla) 딥러닝 컴파일러와 함께 TPU를 지원합니다(자세한 내용은 [여기](https://github.com/pytorch/xla/blob/master/README.md) 참조).
TPU를 사용하려면 `xla_spawn.py` 스크립트를 실행하고 `num_cores` 인수를 사용하여 사용하려는 TPU 코어 수를 설정합니다.

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## 🤗 Accelerate로 스크립트 실행하기[[run-a-script-with-accelerate]]

🤗 [Accelerate](https://huggingface.co/docs/accelerate)는 PyTorch 훈련 과정에 대한 완전한 가시성을 유지하면서 여러 유형의 설정(CPU 전용, 다중 GPU, TPU)에서 모델을 훈련할 수 있는 통합 방법을 제공하는 PyTorch 전용 라이브러리입니다.
🤗 Accelerate가 설치되어 있는지 확인하세요:

> 참고: Accelerate는 빠르게 개발 중이므로 스크립트를 실행하려면 accelerate를 설치해야 합니다.
```bash
pip install git+https://github.com/huggingface/accelerate
```

`run_summarization.py` 스크립트 대신 `run_summarization_no_trainer.py` 스크립트를 사용해야 합니다.
🤗 Accelerate 클래스가 지원되는 스크립트는 폴더에 `task_no_trainer.py` 파일이 있습니다.
다음 명령을 실행하여 구성 파일을 생성하고 저장합니다:
```bash
accelerate config
```

설정을 테스트하여 올바르게 구성되었는지 확인합니다:

```bash
accelerate test
```

이제 훈련을 시작할 준비가 되었습니다:

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## 사용자 정의 데이터 세트 사용하기[[use-a-custom-dataset]]

요약 스크립트는 사용자 지정 데이터 세트가 CSV 또는 JSON 파일인 경우 지원합니다.
사용자 지정 데이터 세트를 사용하는 경우에는 몇 가지 추가 인수를 지정해야 합니다:

- `train_file`과 `validation_file`은 훈련 및 검증 파일의 경로를 지정합니다.
- `text_column`은 요약할 입력 텍스트입니다.
- `summary_column`은 출력할 대상 텍스트입니다.

사용자 지정 데이터 세트를 사용하는 요약 스크립트는 다음과 같습니다:

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## 스크립트 테스트하기[[test-a-script]]

전체 데이터 세트를 대상으로 훈련을 완료하는데 꽤 오랜 시간이 걸리기 때문에, 작은 데이터 세트에서 모든 것이 예상대로 실행되는지 확인하는 것이 좋습니다.

다음 인수를 사용하여 데이터 세트를 최대 샘플 수로 잘라냅니다:
- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

모든 예제 스크립트가 `max_predict_samples` 인수를 지원하지는 않습니다.
스크립트가 이 인수를 지원하는지 확실하지 않은 경우 `-h` 인수를 추가하여 확인하세요:

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## 체크포인트(checkpoint)에서 훈련 이어서 하기[[resume-training-from-checkpoint]]

또 다른 유용한 옵션은 이전 체크포인트에서 훈련을 재개하는 것입니다.
이렇게 하면 훈련이 중단되더라도 처음부터 다시 시작하지 않고 중단한 부분부터 다시 시작할 수 있습니다.
체크포인트에서 훈련을 재개하는 방법에는 두 가지가 있습니다.

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

## 모델 공유하기[[share-your-model]]

모든 스크립트는 최종 모델을 [Model Hub](https://huggingface.co/models)에 업로드할 수 있습니다.
시작하기 전에 Hugging Face에 로그인했는지 확인하세요:
```bash
hf auth login
```

그런 다음 스크립트에 `push_to_hub` 인수를 추가합니다.
이 인수는 Hugging Face 사용자 이름과 `output_dir`에 지정된 폴더 이름으로 저장소를 생성합니다.

저장소에 특정 이름을 지정하려면 `push_to_hub_model_id` 인수를 사용하여 추가합니다.
저장소는 네임스페이스 아래에 자동으로 나열됩니다.
다음 예는 특정 저장소 이름으로 모델을 업로드하는 방법입니다:

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```
