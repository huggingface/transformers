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

# 시각적 질의응답 (Visual Question Answering)

[[open-in-colab]]

시각적 질의응답(VQA)은 이미지를 기반으로 개방형 질문에 대응하는 작업입니다. 이 작업을 지원하는 모델의 입력은 대부분 이미지와 질문의 조합이며, 출력은 자연어로 된 답변입니다.

VQA의 주요 사용 사례는 다음과 같습니다:
* 시각 장애인을 위한 접근성 애플리케이션을 구축할 수 있습니다.
* 교육: 강의나 교과서에 나온 시각 자료에 대한 질문에 답할 수 있습니다. 또한 체험형 전시와 유적 등에서도 VQA를 활용할 수 있습니다.
* 고객 서비스 및 전자상거래: VQA는 사용자가 제품에 대해 질문할 수 있게 함으로써 사용자 경험을 향상시킬 수 있습니다.
* 이미지 검색: VQA 모델을 사용하여 원하는 특성을 가진 이미지를 검색할 수 있습니다. 예를 들어 사용자는 "강아지가 있어?"라고 물어봐서 주어진 이미지 묶음에서 강아지가 있는 모든 이미지를 받아볼 수 있습니다.

이 가이드에서 학습할 내용은 다음과 같습니다:

- VQA 모델 중 하나인 [ViLT](../../en/model_doc/vilt)를 [`Graphcore/vqa` 데이터셋](https://huggingface.co/datasets/Graphcore/vqa) 에서 미세조정하는 방법
- 미세조정된 ViLT 모델로 추론하는 방법
- BLIP-2 같은 생성 모델로 제로샷 VQA 추론을 실행하는 방법

## ViLT 미세 조정 [[finetuning-vilt]]

ViLT는 Vision Transformer (ViT) 내에 텍스트 임베딩을 포함하여 비전/자연어 사전훈련(VLP; Vision-and-Language Pretraining)을 위한 기본 디자인을 제공합니다.
ViLT 모델은 비전 트랜스포머(ViT)에 텍스트 임베딩을 넣어 비전/언어 사전훈련(VLP; Vision-and-Language Pre-training)을 위한 기본적인 디자인을 갖췄습니다. 이 모델은 여러 다운스트림 작업에 사용할 수 있습니다. VQA 태스크에서는 (`[CLS]` 토큰의 최종 은닉 상태 위에 선형 레이어인) 분류 헤더가 있으며 무작위로 초기화됩니다. 
따라서 여기에서 시각적 질의응답은 **분류 문제**로 취급됩니다.

최근의 BLIP, BLIP-2, InstructBLIP와 같은 모델들은 VQA를 생성형 작업으로 간주합니다. 가이드의 후반부에서는 이런 모델들을 사용하여 제로샷 VQA 추론을 하는 방법에 대해 설명하겠습니다.

시작하기 전 필요한 모든 라이브러리를 설치했는지 확인하세요.

```bash
pip install -q transformers datasets
```

커뮤니티에 모델을 공유하는 것을 권장 드립니다. Hugging Face 계정에 로그인하여 🤗 Hub에 업로드할 수 있습니다.
메시지가 나타나면 로그인할 토큰을 입력하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

모델 체크포인트를 전역 변수로 선언하세요.

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## 데이터 가져오기 [[load-the-data]]

이 가이드에서는 `Graphcore/vqa` 데이터세트의 작은 샘플을 사용합니다. 전체 데이터세트는 [🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa) 에서 확인할 수 있습니다.

[`Graphcore/vqa` 데이터세트](https://huggingface.co/datasets/Graphcore/vqa) 의 대안으로 공식 [VQA 데이터세트 페이지](https://visualqa.org/download.html) 에서 동일한 데이터를 수동으로 다운로드할 수 있습니다. 직접 공수한 데이터로 튜토리얼을 따르고 싶다면 [이미지 데이터세트 만들기](https://huggingface.co/docs/datasets/image_dataset#loading-script) 라는
🤗 Datasets 문서를 참조하세요.

검증 데이터의 첫 200개 항목을 불러와 데이터세트의 특성을 확인해 보겠습니다:

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```

예제를 하나 뽑아 데이터세트의 특성을 이해해 보겠습니다.

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

데이터세트에는 다음과 같은 특성이 포함되어 있습니다:
* `question`: 이미지에 대한 질문
* `image_id`: 질문과 관련된 이미지의 경로
* `label`: 데이터의 레이블 (annotations)

나머지 특성들은 필요하지 않기 때문에 삭제해도 됩니다:

```py 
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

보시다시피 `label` 특성은 같은 질문마다 답변이 여러 개 있을 수 있습니다. 모두 다른 데이터 라벨러들로부터 수집되었기 때문인데요. 질문의 답변은 주관적일 수 있습니다. 이 경우 질문은 "그는 어디를 보고 있나요?" 였지만, 어떤 사람들은 "아래"로 레이블을 달았고, 다른 사람들은 "테이블" 또는 "스케이트보드" 등으로 주석을 달았습니다.

아래의 이미지를 보고 어떤 답변을 선택할 것인지 생각해 보세요:

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>

질문과 답변의 모호성으로 인해 이러한 데이터세트는 여러 개의 답변이 가능하므로 다중 레이블 분류 문제로 처리됩니다. 게다가, 원핫(one-hot) 인코딩 벡터를 생성하기보다는 레이블에서 특정 답변이 나타나는 횟수를 기반으로 소프트 인코딩을 생성합니다.

위의 예시에서 "아래"라는 답변이 다른 답변보다 훨씬 더 자주 선택되었기 때문에 데이터세트에서 `weight`라고 불리는 점수로 1.0을 가지며, 나머지 답변들은 1.0 미만의 점수를 가집니다.

적절한 분류 헤더로 모델을 나중에 인스턴스화하기 위해 레이블을 정수로 매핑한 딕셔너리 하나, 반대로 정수를 레이블로 매핑한 딕셔너리 하나 총 2개의 딕셔너리를 생성하세요:

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()} 
```

이제 매핑이 완료되었으므로 문자열 답변을 해당 id로 교체하고, 데이터세트의 더 편리한 후처리를 위해 편평화 할 수 있습니다.

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## 데이터 전처리 [[preprocessing-data]]

다음 단계는 모델을 위해 이미지와 텍스트 데이터를 준비하기 위해 ViLT 프로세서를 가져오는 것입니다. 
[`ViltProcessor`]는 BERT 토크나이저와 ViLT 이미지 프로세서를 편리하게 하나의 프로세서로 묶습니다:

```py 
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

데이터를 전처리하려면 이미지와 질문을 [`ViltProcessor`]로 인코딩해야 합니다. 프로세서는 [`BertTokenizerFast`]로 텍스트를 토크나이즈하고 텍스트 데이터를 위해 `input_ids`, `attention_mask` 및 `token_type_ids`를 생성합니다.
이미지는 [`ViltImageProcessor`]로 이미지를 크기 조정하고 정규화하며, `pixel_values`와 `pixel_mask`를 생성합니다.

이런 전처리 단계는 모두 내부에서 이루어지므로, `processor`를 호출하기만 하면 됩니다. 하지만 아직 타겟 레이블이 완성되지 않았습니다. 타겟의 표현에서 각 요소는 가능한 답변(레이블)에 해당합니다. 정확한 답변의 요소는 해당 점수(weight)를 유지시키고 나머지 요소는 0으로 설정해야 합니다.

아래 함수가 위에서 설명한대로 이미지와 질문에 `processor`를 적용하고 레이블을 형식에 맞춥니다:

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']    

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()
    
...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score
      
...         targets.append(target)

...     encoding["labels"] = targets
    
...     return encoding
```

전체 데이터세트에 전처리 함수를 적용하려면 🤗 Datasets의 [`~datasets.map`] 함수를 사용하십시오. `batched=True`를 설정하여 데이터세트의 여러 요소를 한 번에 처리함으로써 `map`을 더 빠르게 할 수 있습니다. 이 시점에서 필요하지 않은 열은 제거하세요.

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

마지막 단계로, [`DefaultDataCollator`]를 사용하여 예제로 쓸 배치를 생성하세요:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## 모델 훈련 [[train-the-model]]

이제 모델을 훈련하기 위해 준비되었습니다! [`ViltForQuestionAnswering`]으로 ViLT를 가져올 차례입니다. 레이블의 수와 레이블 매핑을 지정하세요:

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

이 시점에서는 다음 세 단계만 남았습니다:

1. [`TrainingArguments`]에서 훈련 하이퍼파라미터를 정의하세요:

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. 모델, 데이터세트, 프로세서, 데이터 콜레이터와 함께 훈련 인수를 [`Trainer`]에 전달하세요:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     tokenizer=processor,
... )
```

3. [`~Trainer.train`]을 호출하여 모델을 미세 조정하세요:

```py
>>> trainer.train() 
```

훈련이 완료되면, [`~Trainer.push_to_hub`] 메소드를 사용하여 🤗 Hub에 모델을 공유하세요:

```py
>>> trainer.push_to_hub()
```

## 추론 [[inference]]

ViLT 모델을 미세 조정하고 🤗 Hub에 업로드했다면 추론에 사용할 수 있습니다. 미세 조정된 모델을 추론에 사용해보는 가장 간단한 방법은 [`Pipeline`]에서 사용하는 것입니다.

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")
```

이 가이드의 모델은 200개의 예제에서만 훈련되었으므로 그다지 많은 것을 기대할 수는 없습니다. 데이터세트의 첫 번째 예제를 사용하여 추론 결과를 설명해보겠습니다:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image, question, top_k=1)
"Where is he looking?"
[{'score': 0.5498199462890625, 'answer': 'down'}]
```

비록 확신은 별로 없지만, 모델은 실제로 무언가를 배웠습니다. 더 많은 예제와 더 긴 훈련 기간이 주어진다면 분명 더 나은 결과를 얻을 수 있을 것입니다!

원한다면 파이프라인의 결과를 수동으로 복제할 수도 있습니다:
1. 이미지와 질문을 가져와서 프로세서를 사용하여 모델에 준비합니다.
2. 전처리된 결과를 모델에 전달합니다.
3. 로짓에서 가장 가능성 있는 답변의 id를 가져와서 `id2label`에서 실제 답변을 찾습니다.

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## 제로샷 VQA [[zeroshot-vqa]]

이전 모델은 VQA를 분류 문제로 처리했습니다. BLIP, BLIP-2 및 InstructBLIP와 같은 최근의 모델은 VQA를 생성 작업으로 접근합니다. [BLIP-2](../../en/model_doc/blip-2)를 예로 들어 보겠습니다. 이 모델은 사전훈련된 비전 인코더와 LLM의 모든 조합을 사용할 수 있는 새로운 비전-자연어 사전 학습 패러다임을 도입했습니다. ([BLIP-2 블로그 포스트](https://huggingface.co/blog/blip-2)를 통해 더 자세히 알아볼 수 있어요)
이를 통해 시각적 질의응답을 포함한 여러 비전-자연어 작업에서 SOTA를 달성할 수 있었습니다.

이 모델을 어떻게 VQA에 사용할 수 있는지 설명해 보겠습니다. 먼저 모델을 가져와 보겠습니다. 여기서 GPU가 사용 가능한 경우 모델을 명시적으로 GPU로 전송할 것입니다. 이전에는 훈련할 때 쓰지 않은 이유는 [`Trainer`]가 이 부분을 자동으로 처리하기 때문입니다:

```py
>>> from transformers import AutoProcessor, Blip2ForConditionalGeneration
>>> import torch

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
```

모델은 이미지와 텍스트를 입력으로 받으므로, VQA 데이터세트의 첫 번째 예제에서와 동일한 이미지/질문 쌍을 사용해 보겠습니다:

```py 
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

BLIP-2를 시각적 질의응답 작업에 사용하려면 텍스트 프롬프트가 `Question: {} Answer:` 형식을 따라야 합니다.

```py
>>> prompt = f"Question: {question} Answer:" 
```

이제 모델의 프로세서로 이미지/프롬프트를 전처리하고, 처리된 입력을 모델을 통해 전달하고, 출력을 디코드해야 합니다:

```py
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"He is looking at the crowd" 
```

보시다시피 모델은 군중을 인식하고, 얼굴의 방향(아래쪽을 보고 있음)을 인식했지만, 군중이 스케이터 뒤에 있다는 사실을 놓쳤습니다. 그러나 사람이 직접 라벨링한 데이터셋을 얻을 수 없는 경우에, 이 접근법은 빠르게 유용한 결과를 생성할 수 있습니다.
