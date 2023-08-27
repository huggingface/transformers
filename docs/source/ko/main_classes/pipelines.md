<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 파이프라인 [[pipelines]]

파이프라인은 추론에 모델을 사용할 수 있는 훌륭하고 쉬운 방법입니다. 이러한 파이프라인은 라이브러리의 복잡한 코드를 대부분 추상화한 객체들로, 
명명된 개체 인식(Named Entity Recognition), 마스크드 언어 모델링(Masked Language Modeling), 
감정 분석(Sentiment Analysis), 특징 추출(Feature Extraction), 질문 응답(Question Answering)과 같은 여러 작업을 위한 간단한 API를 제공합니다. 
사용 예제는 [작업 요약](../task_summary)에서 확인할 수 있습니다.

파이프라인 추상화에는 두 가지 범주가 있습니다:

- 가장 강력한 객체인 [`pipeline`]은 다른 모든 파이프라인을 캡슐화합니다.
- 작업별 파이프라인은 [오디오](#audio), [컴퓨터 비전](#computer-vision), [자연어 처리](#natural-language-processing), [멀티모달](#multimodal) 작업에 사용할 수 있습니다.

## 파이프라인 추상화 [[the-pipeline-abstraction]]

*파이프라인* 추상화는 다른 모든 사용 가능한 파이프라인을 감싸고 있는 래퍼입니다. 
다른 파이프라인처럼 인스턴스화될 수 있지만, 추가적인 편의성을 제공할 수도 있습니다.

단일 항목에 대한 간단한 호출:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

[허브](https://huggingface.co)에서 특정 모델을 사용하려고 할 떄, 
허브에서 모델이 이미 그것을 정의하고 있다면 그 작업은 건너뛰어도 됩니다:

```python
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

여러 항목에 파이프라인을 호출하려면, *리스트*로 호출할 수 있습니다.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

전체 데이터 세트를 반복하려면 `데이터셋`을 직접 사용하는 것이 좋습니다. 
즉, 한 번에 전체 데이터 세트를 할당할 필요가 없고, 직접 일괄 처리할 필요도 없다는 것을 의미합니다.
이것은 GPU에서의 사용자 정의 루프만큼 빠르게 작동해야 합니다. 그렇지 않다면 주저하지 말고 이슈를 생성하세요.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset(*pt*만 해당)은 단순히 데이터셋 항목이 반환한 딕셔너리의 항목을 반환합니다. 
# 데이터셋의 *대상* 부분에는 관심이 없기 때문입니다. 문장 쌍의 경우 KeyPairDataset을 사용합니다.
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

사용하기 쉽도록 생성자도 가능합니다:


```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # 이것은 서버의 데이터 세트, 데이터베이스, 
        # 대기열 또는 HTTP 요청에서 올 수 있습니다.
        # 주의: 이 작업은 반복적이므로 `num_workers > 1` 변수를 이용하는 것으로 
        # 여러 스레드를 사용하여 데이터를 전처리할 수 없습니다. 
        # 메인 스레드가 대규모 추론을 실행하는 동안 전처리를 수행하는 하나의 스레드만 사용할 수 있습니다. 
        yield "This is a test"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

[[autodoc]] 파이프라인

## 파이프라인 일괄 처리 [[pipeline-batching]]

모든 파이프라인은 일괄 처리 작업이 가능합니다. 
일괄 처리는 파이프라인이 스트리밍 기능을 사용할 때마다(즉, 리스트나 `Dataset` 또는 `generator`를 전달할 때) 작동합니다.

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # 이전과 정확히 동일한 출력이지만 
    # 그 내용은 모델에 일괄적으로 전달됩니다.
```

<Tip warning={true}>

그러나 이것이 자동으로 성능 향상으로 이어지지는 않습니다. 
하드웨어, 데이터 및 실제 사용 중인 모델에 따라 10배 빨라지거나 5배 느려질 수도 있습니다.

속도 향상이 주로 나타나는 예:

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)
```

속도 저하가 주로 나타나는 예:

```python
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

이것은 다른 것들에 비해 아주 긴 문장입니다. 이 경우, **전체** 일괄 처리 작업의 길이가 400토큰이 되어야 하므로 
전체 일괄 처리 작업의 크기가 [64, 4]가 아닌 [64, 400]이 되어 속도가 매우 느려집니다. 더 나쁜 것은,
더 큰 일괄 처리 작업에서는 프로그램이 단순히 충돌한다는 것입니다.


```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

이 문제에 대한 좋은 (일반적인) 해결책은 없으며, 사용 사례에 따라 결과가 달라질 수 있습니다. 
대략적인 규칙:

경험에 따른 사용자를 위한 규칙은 다음과 같습니다:

- **하드웨어와 함께 부하에 대한 성능을 측정하세요. 측정하고, 측정하고, 계속 측정하세요. 
  실제 숫자만이 해답입니다.**
- 지연 시간 제약이 있는 경우(추론을 수행하는 실시간 상품), 일괄 처리하지 마세요.
- CPU를 사용하는 경우 일괄 처리하지 마세요.
- 처리량이 중요한 경우(많은 정적 데이터에 대해 모델을 실행하려는 경우) GPU를 사용하세요:

  - sequence_length("원래"의 데이터)의 크기에 대한 단서가 없는 경우, 기본적으로 일괄 처리하지 말고 측정한 후
    임시로 추가를 시도하고, 실패할 때 복구하기 위해 OOM 검사를 추가합니다.
    (그리고 시퀀스 길이를 제어하지 않으면 언젠가는 실패할 것입니다).
  - 시퀀스 길이가 매우 규칙적이라면, 일괄 처리가 아주 흥미롭게 보일 가능성이 큽니다. 
    OOM이 발생할 때까지 측정하고 푸시하세요.
  - GPU가 클수록 일괄 처리 더 흥미롭게 보일 가능성이 큽니다.
- 일괄처리를 활성화하자마자 OOM을 잘 처리할 수 있는지 확인하세요.
 
## 파이프라인 청크 일괄처리 [[pipeline-chunk-batching]]

`zero-shot-classification`과 `question-answering`은 단일 입력으로 여러 개의 포워드 패스를 생성할 수 있다는 점에서 약간 특별합니다. 
일반적인 상황에서는 이것이 `batch_size` 인수에서 문제가 발생할 것입니다.

이 문제를 피하기 위한 이 두 파이프라인은 조금 특별합니다. 바로 일반 `Pipeline` 대신 `ChunkPipeline`입니다. 
요약하면:


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

이제 다음과 같이 됩니다:


```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

파이프라인이 동일한 방식으로 사용되기 때문에 
코드에 매우 투명해야 합니다.

파이프라인이 일괄 처리 작업을 자동으로 처리할 수 있기 때문에 이것은 단순화된 보기입니다! 
즉, 실제로 얼마나 많은 포워드 패스를 트리거할지 신경 쓸 필요가 없으므로, 입력과 무관하게 `batch_size`를 최적화할 수 있습니다. 
이전 섹션의 주의 사항은 여전히 적용됩니다.

## 파이프라인 사용자 정의 코드 [[pipeline-custom-code]]

특정 파이프라인을 재정의하려면.

파이프라인의 목표는 사용하기 쉽고 대부분의 사용 사례를 지원하는 것이므로 이슈를 생성하는 것을 주저하지 마세요.
그 결과 '트랜스포머'가 당신의 사용 사례를 지원할 수 있습니다.


간단하게 시도하려면 다음을 수행할 수 있습니다:

- 선택한 파이프라인을 서브 클래스로 만들기

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

이것은 당신이 원하는 모든 사용자 정의 코드를 수행할 수 있게 해야 합니다.


## 파이프라인 구현하기 [[implementing-a-pipeline]]

[새로운 파이프라인 구현하기](../add_new_pipeline)

## 오디오 [[Audio]]

오디오 작업에 사용 가능한 파이프라인은 다음과 같습니다.

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### TextToAudioPipeline

[[autodoc]] TextToAudioPipeline
    - __call__
    - all


### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## Computer vision [[Computer-vision]]

컴퓨터 비전 작업에 사용 가능한 파이프라인은 다음과 같습니다.

### DepthEstimationPipeline
[[autodoc]] DepthEstimationPipeline
    - __call__
    - all

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline
    - __call__
    - all

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline
    - __call__
    - all

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline
    - __call__
    - all

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline
    - __call__
    - all

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline
    - __call__
    - all

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline
    - __call__
    - all

## Natural Language Processing

자연어 처리 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### ConversationalPipeline

[[autodoc]] Conversation

[[autodoc]] ConversationalPipeline
    - __call__
    - all

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### NerPipeline

[[autodoc]] NerPipeline

더 자세한 세부사항은 [`TokenClassificationPipeline`] 을 확인하세요.

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline
    - __call__
    - all

### SummarizationPipeline

[[autodoc]] SummarizationPipeline
    - __call__
    - all

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline
    - __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline
    - __call__
    - all

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline
    - __call__
    - all

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline
    - __call__
    - all

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline
    - __call__
    - all

### TranslationPipeline

[[autodoc]] TranslationPipeline
    - __call__
    - all

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline
    - __call__
    - all

## Multimodal

멀티모달 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline`

[[autodoc]] Pipeline
