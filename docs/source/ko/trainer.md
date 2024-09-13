<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer [[trainer]]

[`Trainer`]는 Transformers 라이브러리에 구현된 PyTorch 모델을 반복하여 훈련 및 평가 과정입니다. 훈련에 필요한 요소(모델, 토크나이저, 데이터셋, 평가 함수, 훈련 하이퍼파라미터 등)만 제공하면 [`Trainer`]가 필요한 나머지 작업을 처리합니다. 이를 통해 직접 훈련 루프를 작성하지 않고도 빠르게 훈련을 시작할 수 있습니다. 또한 [`Trainer`]는 강력한 맞춤 설정과 다양한 훈련 옵션을 제공하여 사용자 맞춤 훈련이 가능합니다.

<Tip>

Transformers는 [`Trainer`] 클래스 외에도 번역이나 요약과 같은 시퀀스-투-시퀀스 작업을 위한 [`Seq2SeqTrainer`] 클래스도 제공합니다. 또한 [TRL](https://hf.co/docs/trl) 라이브러리에는 [`Trainer`] 클래스를 감싸고 Llama-2 및 Mistral과 같은 언어 모델을 자동 회귀 기법으로 훈련하는 데 최적화된 [`~trl.SFTTrainer`] 클래스 입니다. [`~trl.SFTTrainer`]는 시퀀스 패킹, LoRA, 양자화 및 DeepSpeed와 같은 기능을 지원하여 크기 상관없이 모델 효율적으로 확장할 수 있습니다.

<br>

이들 다른 [`Trainer`] 유형 클래스에 대해 더 알고 싶다면 [API 참조](./main_classes/trainer)를 확인하여 언제 어떤 클래스가 적합할지 얼마든지 확인하세요. 일반적으로 [`Trainer`]는 가장 다재다능한 옵션으로, 다양한 작업에 적합합니다. [`Seq2SeqTrainer`]는 시퀀스-투-시퀀스 작업을 위해 설계되었고, [`~trl.SFTTrainer`]는 언어 모델 훈련을 위해 설계되었습니다.

</Tip>

시작하기 전에, 분산 환경에서 PyTorch 훈련과 실행을 할 수 있게 [Accelerate](https://hf.co/docs/accelerate) 라이브러리가 설치되었는지 확인하세요.

```bash
pip install accelerate

# 업그레이드
pip install accelerate --upgrade
```

이 가이드는 [`Trainer`] 클래스에 대한 개요를 제공합니다.

## 기본 사용법 [[basic-usage]]

[`Trainer`]는 기본적인 훈련 루프에 필요한 모든 코드를 포함하고 있습니다.

1. 손실을 계산하는 훈련 단계를 수행합니다.
2. [`~accelerate.Accelerator.backward`] 메소드로 그레이디언트를 계산합니다.
3. 그레이디언트를 기반으로 가중치를 업데이트합니다.
4. 정해진 에폭 수에 도달할 때까지 이 과정을 반복합니다.

[`Trainer`] 클래스는 PyTorch와 훈련 과정에 익숙하지 않거나 막 시작한 경우에도 훈련이 가능하도록 필요한 모든 코드를 추상화하였습니다. 또한 매번 훈련 루프를 손수 작성하지 않아도 되며, 훈련에 필요한 모델과 데이터셋 같은 필수 구성 요소만 제공하면, [Trainer] 클래스가 나머지를 처리합니다.

훈련 옵션이나 하이퍼파라미터를 지정하려면, [`TrainingArguments`] 클래스에서 확인 할 수 있습니다. 예를 들어, 모델을 저장할 디렉토리를 `output_dir`에 정의하고, 훈련 후에 Hub로 모델을 푸시하려면 `push_to_hub=True`로 설정합니다.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

`training_args`를 [`Trainer`]에 모델, 데이터셋, 데이터셋 전처리 도구(데이터 유형에 따라 토크나이저, 특징 추출기 또는 이미지 프로세서일 수 있음), 데이터 수집기 및 훈련 중 확인할 지표를 계산할 함수를 함께 전달하세요.

마지막으로, [`~Trainer.train`]를 호출하여 훈련을 시작하세요!

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 체크포인트 [[checkpoints]]

[`Trainer`] 클래스는 [`TrainingArguments`]의 `output_dir` 매개변수에 지정된 디렉토리에 모델 체크포인트를 저장합니다. 체크포인트는 `checkpoint-000` 하위 폴더에 저장되며, 여기서 끝의 숫자는 훈련 단계에 해당합니다. 체크포인트를 저장하면 나중에 훈련을 재개할 때 유용합니다.

```py
# 최신 체크포인트에서 재개
trainer.train(resume_from_checkpoint=True)

# 출력 디렉토리에 저장된 특정 체크포인트에서 재개
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

체크포인트를 Hub에 푸시하려면 [`TrainingArguments`]에서 `push_to_hub=True`로 설정하여 커밋하고 푸시할 수 있습니다. 체크포인트 저장 방법을 결정하는 다른 옵션은 [`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy) 매개변수에서 설정합니다:

* `hub_strategy="checkpoint"`는 최신 체크포인트를 "last-checkpoint"라는 하위 폴더에 푸시하여 훈련을 재개할 수 있습니다.
* `hub_strategy="all_checkpoints"`는 모든 체크포인트를 `output_dir`에 정의된 디렉토리에 푸시합니다(모델 리포지토리에서 폴더당 하나의 체크포인트를 볼 수 있습니다).

체크포인트에서 훈련을 재개할 때, [`Trainer`]는 체크포인트가 저장될 때와 동일한 Python, NumPy 및 PyTorch RNG 상태를 유지하려고 합니다. 하지만 PyTorch는 기본 설정으로 '일관된 결과를 보장하지 않음'으로 많이 되어있기 때문에, RNG 상태가 동일할 것이라고 보장할 수 없습니다. 따라서, 일관된 결과가 보장되도록 활성화 하려면, [랜덤성 제어](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) 가이드를 참고하여 훈련을 완전히 일관된 결과를 보장 받도록 만들기 위해 활성화할 수 있는 항목을 확인하세요. 다만, 특정 설정을 결정적으로 만들면 훈련이 느려질 수 있습니다.

## Trainer 맞춤 설정 [[customize-the-trainer]]

[`Trainer`] 클래스는 접근성과 용이성을 염두에 두고 설계되었지만, 더 다양한 기능을 원하는 사용자들을 위해 다양한 맞춤 설정 옵션을 제공합니다. [`Trainer`]의 많은 메소드는 서브클래스화 및 오버라이드하여 원하는 기능을 제공할 수 있으며, 이를 통해 전체 훈련 루프를 다시 작성할 필요 없이 원하는 기능을 추가할 수 있습니다. 이러한 메소드에는 다음이 포함됩니다:

* [`~Trainer.get_train_dataloader`]는 훈련 데이터로더를 생성합니다.
* [`~Trainer.get_eval_dataloader`]는 평가 데이터로더를 생성합니다.
* [`~Trainer.get_test_dataloader`]는 테스트 데이터로더를 생성합니다.
* [`~Trainer.log`]는 훈련을 모니터링하는 다양한 객체에 대한 정보를 로그로 남깁니다.
* [`~Trainer.create_optimizer_and_scheduler`]는 `__init__`에서 전달되지 않은 경우 옵티마이저와 학습률 스케줄러를 생성합니다. 이들은 각각 [`~Trainer.create_optimizer`] 및 [`~Trainer.create_scheduler`]로 별도로 맞춤 설정 할 수 있습니다.
* [`~Trainer.compute_loss`]는 훈련 입력 배치에 대한 손실을 계산합니다.
* [`~Trainer.training_step`]는 훈련 단계를 수행합니다.
* [`~Trainer.prediction_step`]는 예측 및 테스트 단계를 수행합니다.
* [`~Trainer.evaluate`]는 모델을 평가하고 평가 지표을 반환합니다.
* [`~Trainer.predict`]는 테스트 세트에 대한 예측(레이블이 있는 경우 지표 포함)을 수행합니다.

예를 들어, [`~Trainer.compute_loss`] 메소드를 맞춤 설정하여 가중 손실을 사용하려는 경우:

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self,

 model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # 순방향 전파
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # 서로 다른 가중치로 3개의 레이블에 대한 사용자 정의 손실을 계산
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### 콜백 [[callbacks]]

[`Trainer`]를 맞춤 설정하는 또 다른 방법은 [콜백](callbacks)을 사용하는 것입니다. 콜백은 훈련 루프에서 *변화를 주지 않습니다*. 훈련 루프의 상태를 검사한 후 상태에 따라 일부 작업(조기 종료, 결과 로그 등)을 실행합니다. 즉, 콜백은 사용자 정의 손실 함수와 같은 것을 구현하는 데 사용할 수 없으며, 이를 위해서는 [`~Trainer.compute_loss`] 메소드를 서브클래스화하고 오버라이드해야 합니다.

예를 들어, 훈련 루프에 10단계 후 조기 종료 콜백을 추가하려면 다음과 같이 합니다.

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
```

그런 다음, 이를 [`Trainer`]의 `callback` 매개변수에 전달합니다.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()],
)
```

## 로깅 [[logging]]

<Tip>

로깅 API에 대한 자세한 내용은 [로깅](./main_classes/logging) API 레퍼런스를 확인하세요.

</Tip>

[`Trainer`]는 기본적으로 `logging.INFO`로 설정되어 있어 오류, 경고 및 기타 기본 정보를 보고합니다. 분산 환경에서는 [`Trainer`] 복제본이 `logging.WARNING`으로 설정되어 오류와 경고만 보고합니다. [`TrainingArguments`]의 [`log_level`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level) 및 [`log_level_replica`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level_replica) 매개변수로 로그 레벨을 변경할 수 있습니다.

각 노드의 로그 레벨 설정을 구성하려면 [`log_on_each_node`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_on_each_node) 매개변수를 사용하여 각 노드에서 로그 레벨을 사용할지 아니면 주 노드에서만 사용할지 결정하세요.

<Tip>

[`Trainer`]는 [`Trainer.__init__`] 메소드에서 각 노드에 대해 로그 레벨을 별도로 설정하므로, 다른 Transformers 기능을 사용할 경우 [`Trainer`] 객체를 생성하기 전에 이를 미리 설정하는 것이 좋습니다.

</Tip>

예를 들어, 메인 코드와 모듈을 각 노드에 따라 동일한 로그 레벨을 사용하도록 설정하려면 다음과 같이 합니다.

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

각 노드에서 기록될 내용을 구성하기 위해 `log_level`과 `log_level_replica`를 다양한 조합으로 사용해보세요.

<hfoptions id="logging">
<hfoption id="single node">

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">

멀티 노드 환경에서는 `log_on_each_node 0` 매개변수를 추가합니다.

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# 오류만 보고하도록 설정
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

## NEFTune [[neftune]]

[NEFTune](https://hf.co/papers/2310.05914)은 훈련 중 임베딩 벡터에 노이즈를 추가하여 성능을 향상시킬 수 있는 기술입니다. [`Trainer`]에서 이를 활성화하려면 [`TrainingArguments`]의 `neftune_noise_alpha` 매개변수를 설정하여 노이즈의 양을 조절합니다.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```

NEFTune은 예상치 못한 동작을 피할 목적으로 처음 임베딩 레이어로 복원하기 위해 훈련 후 비활성화 됩니다.

## GaLore [[galore]]

Gradient Low-Rank Projection (GaLore)은 전체 매개변수를 학습하면서도 LoRA와 같은 일반적인 저계수 적응 방법보다 더 메모리 효율적인 저계수 학습 전략입니다.

먼저 GaLore 공식 리포지토리를 설치합니다:

```bash
pip install galore-torch
```

그런 다음 `optim`에 `["galore_adamw", "galore_adafactor", "galore_adamw_8bit"]` 중 하나와 함께 `optim_target_modules`를 추가합니다. 이는 적용하려는 대상 모듈 이름에 해당하는 문자열, 정규 표현식 또는 전체 경로의 목록일 수 있습니다. 아래는 end-to-end 예제 스크립트입니다(필요한 경우 `pip install trl datasets`를 실행):

```python
import torch
import datasets
import trl

from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw",
    optim_target_modules=["attn", "mlp"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```

GaLore가 지원하는 추가 매개변수를 전달하려면 `optim_args`를 설정합니다. 예를 들어:

```python
import torch
import datasets
import trl

from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw",
    optim_target_modules=["attn", "mlp"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```

해당 방법에 대한 자세한 내용은 [원본 리포지토리](https://github.com/jiaweizzhao/GaLore) 또는 [논문](https://arxiv.org/abs/2403.03507)을 참고하세요.

현재 GaLore 레이어로 간주되는 Linear 레이어만 훈련 할수 있으며, 저계수 분해를 사용하여 훈련되고 나머지 레이어는 기존 방식으로 최적화됩니다.

훈련 시작 전에 시간이 약간 걸릴 수 있습니다(NVIDIA A100에서 2B 모델의 경우 약 3분), 하지만 이후 훈련은 원활하게 진행됩니다.

다음과 같이 옵티마이저 이름에 `layerwise`를 추가하여 레이어별 최적화를 수행할 수도 있습니다:

```python
import torch
import datasets
import trl

from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw_layerwise",
    optim_target_modules=["attn", "mlp"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```

레이어별 최적화는 다소 실험적이며 DDP(분산 데이터 병렬)를 지원하지 않으므로, 단일 GPU에서만 훈련 스크립트를 실행할 수 있습니다. 자세한 내용은 [이 문서를](https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#train-7b-model-with-a-single-gpu-with-24gb-memory)을 참조하세요. gradient clipping, DeepSpeed 등 다른 기능은 기본적으로 지원되지 않을 수 있습니다. 이러한 문제가 발생하면 [GitHub에 이슈를 올려주세요](https://github.com/huggingface/transformers/issues).

## LOMO 옵티마이저 [[lomo-optimizer]]

LOMO 옵티마이저는 [제한된 자원으로 대형 언어 모델의 전체 매개변수 미세 조정](https://hf.co/papers/2306.09782)과 [적응형 학습률을 통한 저메모리 최적화(AdaLomo)](https://hf.co/papers/2310.10195)에서 도입되었습니다. 
이들은 모두 효율적인 전체 매개변수 미세 조정 방법으로 구성되어 있습니다. 이러한 옵티마이저들은 메모리 사용량을 줄이기 위해 그레이디언트 계산과 매개변수 업데이트를 하나의 단계로 융합합니다. LOMO에서 지원되는 옵티마이저는 `"lomo"`와 `"adalomo"`입니다. 먼저 pypi에서 `pip install lomo-optim`를 통해 `lomo`를 설치하거나, GitHub 소스에서 `pip install git+https://github.com/OpenLMLab/LOMO.git`로 설치하세요.

<Tip>

저자에 따르면, `grad_norm` 없이 `AdaLomo`를 사용하는 것이 더 나은 성능과 높은 처리량을 제공한다고 합니다.

</Tip>

다음은 IMDB 데이터셋에서 [google/gemma-2b](https://huggingface.co/google/gemma-2b)를 최대 정밀도로 미세 조정하는 간단한 스크립트입니다:

```python
import torch
import datasets
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import trl

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="adalomo",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="lomo-imdb",
)

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=1024,
)

trainer.train()
```

## Accelerate와 Trainer [[accelerate-and-trainer]]

[`Trainer`] 클래스는 [Accelerate](https://hf.co/docs/accelerate)로 구동되며, 이는 [FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 및 [DeepSpeed](https://www.deepspeed.ai/)와 같은 통합을 지원하는 분산 환경에서 PyTorch 모델을 쉽게 훈련할 수 있는 라이브러리입니다.

<Tip>

FSDP 샤딩 전략, CPU 오프로드 및 [`Trainer`]와 함께 사용할 수 있는 더 많은 기능을 알아보려면 [Fully Sharded Data Parallel](fsdp) 가이드를 확인하세요.

</Tip>

[`Trainer`]와 Accelerate를 사용하려면 [`accelerate.config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) 명령을 실행하여 훈련 환경을 설정하세요. 이 명령은 훈련 스크립트를 실행할 때 사용할 `config_file.yaml`을 생성합니다. 예를 들어, 다음 예시는 설정할 수 있는 일부 구성 예입니다.

<hfoptions id="config">
<hfoption id="DistributedDataParallel">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
distributed_type: MULTI_GPU                                                                                                    
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 # 노드에 따라 순위를 변경하세요
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="FSDP">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
deepspeed_config:                                                                                                              
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
</hfoptions>

[`accelerate_launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) 명령은 Accelerate와 [`Trainer`]를 사용하여 분산 시스템에서 훈련 스크립트를 실행하는 권장 방법이며, `config_file.yaml`에 지정된 매개변수를 사용합니다. 이 파일은 Accelerate 캐시 폴더에 저장되며 `accelerate_launch`를 실행할 때 자동으로 로드됩니다.

예를 들어, FSDP 구성을 사용하여 [run_glue.py](https://github.com/huggingface/transformers/blob/f4db565b695582891e43a5e042e5d318e28f20b8/examples/pytorch/text-classification/run_glue.py#L4) 훈련 스크립트를 실행하려면 다음과 같이 합니다:

```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

`config_file.yaml` 파일의 매개변수를 직접 지정할 수도 있습니다:

```bash
accelerate launch --num_processes=2 \
    --use_fsdp \
    --mixed_precision=bf16 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
    --fsdp_transformer_layer_cls_to_wrap="BertLayer" \
    --fsdp_sharding_strategy=1 \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

`accelerate_launch`와 사용자 정의 구성에 대해 더 알아보려면 [Accelerate 스크립트 실행](https://huggingface.co/docs/accelerate/basic_tutorials/launch) 튜토리얼을 확인하세요.