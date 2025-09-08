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

파이프라인은 모델을 추론에 활용할 수 있는 훌륭하고 쉬운 방법입니다. 이 파이프라인은 라이브러리의 복잡한 코드를 대부분 추상화하여, 개체명 인식(Named Entity Recognition), 마스크드 언어 모델링(Masked Language Modeling), 감정 분석(Sentiment Analysis), 특성 추출(Feature Extraction), 질의응답(Question Answering) 등의 여러 작업에 특화된 간단한 API를 제공합니다. 사용 예시는 [작업 요약](../task_summary)을 참고하세요.

파이프라인 추상화는 다음 두 가지 범주로 나뉩니다.

- \[`파이프라인`]은 다른 모든 파이프라인을 캡슐화하는 가장 강력한 객체입니다.
- 작업별 파이프라인은 [오디오](#audio), [컴퓨터 비전](#computer-vision), [자연어 처리](#natural-language-processing), [멀티모달](#multimodal) 작업에 사용할 수 있습니다.

## 파이프라인 추상화 [[the-pipeline-abstraction]]

*파이프라인* 추상화는 사용 가능한 모든 파이프라인을 감싸는 래퍼입니다. 다른 파이프라인처럼 인스턴스화되며, 추가적인 편의 기능을 제공합니다.

단일 항목 호출 예시:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

[hub](https://huggingface.co)에서 특정 모델을 사용하려는 경우, 해당 모델이 이미 허브에 작업을 정의하고 있다면 작업명을 생략할 수 있습니다.

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

여러 항목을 처리하려면 *리스트*를 전달하세요.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"] )
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

전체 데이터셋을 순회하려면 `dataset`을 직접 사용하는 것이 좋습니다.
이렇게 하면 전체 데이터를 한 번에 메모리에 올릴 필요도 없고, 배치 처리를 따로 구현하지 않아도 됩니다.
이 방식은 GPU에서 사용자 정의 루프와 유사한 속도로 작동하며, 만약 그렇지 않을 경우 이슈를 등록해 주세요.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (*pt* 전용)는 데이터셋 항목의 딕셔너리에서 지정된 키만 반환합니다.
# 이 예제에서는 *target* 항목이 필요하지 않으므로 KeyDataset을 사용합니다. 문장 쌍 입력에는 KeyPairDataset을 사용하세요.
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

더 편리하게 사용하려면 제너레이터도 가능합니다.

```python
from transformers import pipeline

pipe = pipeline("text-classification")

def data():
    while True:
        # 데이터는 데이터셋, 데이터베이스, 큐 또는 HTTP 요청에서 올 수 있습니다.
        # 서버에서
        # 주의: 반복적이므로 `num_workers > 1` 변수를 사용할 수 없습니다.
        # 데이터를 전처리하기 위해 여러 스레드를 사용할 수 없습니다. 여전히
        # 메인 스레드가 대규모 추론을 수행하는 동안 하나의 스레드가 전처리를 수행할 수 있습니다.
        yield "This is a test"

for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

\[\[autodoc]] pipeline

## 파이프라인 배치 처리 [[pipeline-batching]]

모든 파이프라인은 배치 처리를 지원합니다. 리스트, `Dataset`, `Generator` 전달 시 스트리밍 기능을 사용할 때 작동합니다.

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # 이전과 동일한 출력이지만, 내용을 배치로 모델에 전달합니다.
```

<Tip warning={true}>

하지만 배치 처리가 항상 성능 향상을 보장하는 것은 아닙니다. 하드웨어, 데이터, 모델에 따라 속도가 10배로 빨라질수도, 5배 느려질 수 있습니다.

주로 속도 향상이 있는 예시:

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

주로 속도 저하가 있는 예시:

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

이는 다른 문장들에 비해 간헐적으로 매우 긴 문장이 포함된 경우입니다. 이 경우 **전체** 배치가 400토큰 길이로  
([64, 400]) 되어야 하므로, [64, 4] 대신 [64, 400]이 되어 크게 속도가 저하됩니다. 게다가, 더 큰 배치에서는 프로그램이 충돌할 수 있습니다.


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

일반적인 해결책은 없으며, 사용 사례에 따라 다를 수 있습니다.

사용자를 위한 경험상 지침:

- **하드웨어와 실제 워크로드로 성능을 측정하세요. 측정이 답입니다.**
- 실시간 추론(latency)이 중요하다면 배치 처리하지 마세요.
- CPU 사용 시에도 배치 처리하지 않는 것이 좋습니다.
- GPU에서 정적 데이터 처리(throughput)가 목적이라면

  - 입력 시퀀스 길이("실제" 데이터)를 잘 모르는 경우, 기본적으로 배치 처리하지 말고 성능을 측정하면서 임시로 배치를 적용해 보고, 실패 시 이를 복구할 수 있도록 OOM 검사 로직을 추가하세요. (시퀀스 길이를 제어하지 않으면 언젠가는 실패하게 됩니다.)
  - 시퀀스 길이가 일정하다면 배치 처리가 유리할 수 있습니다. 측정하며 OOM까지 시도해 보세요.
  - GPU 메모리가 클수록 배치 처리의 이점이 큽니다.
- 배치 처리 활성화 시 OOM을 핸들링할 수 있도록 대비하세요.

## 파이프라인 청크 배치 처리 [[pipeline-chunk-batching]]

`제로샷 분류` 및 `질의응답` 파이프라인은 단일 입력이 여러 포워드 패스를 유발할 수 있어 `배치 크기` 인자를 그대로 사용하면 문제가 발생할 수 있습니다.

이를 해결하기 위해 두 파이프라인은 `청크 파이프라인` 형태로 동작합니다. 요약하면

```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

이제 내부적으로는

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

파이프라인의 사용 방식이 동일하므로, 코드에는 거의 영향을 주지 않습니다.

파이프라인은 배치 처리를 자동으로 수행하기 때문에 입력이 몇 번의 포워드 패스를 발생시키는지 고려할 필요 없이, `배치 크기`는 입력과 무관하게 최적화할 수 있습니다.
다만 앞서 언급한 주의사항은 여전히 유효합니다.

## 파이프라인 FP16 추론 [[pipeline-fp16-inference]]

모델은 FP16 모드로 실행할 수 있으며, GPU에서 메모리를 절약하면서 처리 속도를 크게 향상시킬 수 있습니다. 대부분의 모델은 성능 저하 없이 FP16을 지원하며, 모델이 클수록 성능 저하 가능성은 더 낮아집니다.

FP16 추론을 활성화하려면 파이프라인 생성자에 `dtype=torch.float16` 또는 `dtype='float16'`을 전달하세요. 이 기능은 파이토치 백엔드를 사용하는 모델에서만 작동하며, 입력은 내부적으로 FP16 형식으로 변환됩니다.

## 파이프라인 사용자 정의 코드 [[pipeline-custom-code]]

특정 파이프라인을 오버라이드하려면, 먼저 해당 작업에 대한 이슈를 등록해 주세요. 파이프라인의 목표는 대부분의 사용 사례를 지원하는 것이므로, `transformers` 팀이 추가 지원을 고려할 수 있습니다.

간단히 시도하려면 파이프라인 클래스를 상속하세요.

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # 사용자 정의 후처리 코드 작성
        scores = scores * 100
        # 추가 코드 작성

my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# 또는 *pipeline* 함수를 사용할 경우:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

이를 통해 원하는 모든 커스텀 코드를 적용할 수 있습니다.

## 파이프라인 구현하기 [[implementing-a-pipeline]]

[새 파이프라인 구현](../add_new_pipeline)

## 오디오 [[audio]]

오디오 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### AudioClassificationPipeline [[transformers.AudioClassificationPipeline]]

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline [[transformers.AutomaticSpeechRecognitionPipeline]]

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### TextToAudioPipeline [[transformers.TextToAudioPipeline]]

[[autodoc]] TextToAudioPipeline
    - __call__
    - all


### ZeroShotAudioClassificationPipeline [[transformers.ZeroShotAudioClassificationPipeline]]

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## 컴퓨터 비전 [[computer-vision]]

컴퓨터 비전 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### DepthEstimationPipeline [[transformers.DepthEstimationPipeline]]
[[autodoc]] DepthEstimationPipeline
    - __call__
    - all

### ImageClassificationPipeline [[transformers.ImageClassificationPipeline]]

[[autodoc]] ImageClassificationPipeline
    - __call__
    - all

### ImageSegmentationPipeline [[transformers.ImageSegmentationPipeline]]

[[autodoc]] ImageSegmentationPipeline
    - __call__
    - all

### ImageToImagePipeline [[transformers.ImageToImagePipeline]]

[[autodoc]] ImageToImagePipeline
    - __call__
    - all

### ObjectDetectionPipeline [[transformers.ObjectDetectionPipeline]]

[[autodoc]] ObjectDetectionPipeline
    - __call__
    - all

### VideoClassificationPipeline [[transformers.VideoClassificationPipeline]]

[[autodoc]] VideoClassificationPipeline
    - __call__
    - all

### ZeroShotImageClassificationPipeline [[transformers.ZeroShotImageClassificationPipeline]]

[[autodoc]] ZeroShotImageClassificationPipeline
    - __call__
    - all

### ZeroShotObjectDetectionPipeline [[transformers.ZeroShotObjectDetectionPipeline]]

[[autodoc]] ZeroShotObjectDetectionPipeline
    - __call__
    - all

## 자연어 처리 [[natural-language-processing]]

자연어 처리 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### FillMaskPipeline [[transformers.FillMaskPipeline]]

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### QuestionAnsweringPipeline [[transformers.QuestionAnsweringPipeline]]

[[autodoc]] QuestionAnsweringPipeline
    - __call__
    - all

### SummarizationPipeline [[transformers.SummarizationPipeline]]

[[autodoc]] SummarizationPipeline
    - __call__
    - all

### TableQuestionAnsweringPipeline [[transformers.TableQuestionAnsweringPipeline]]

[[autodoc]] TableQuestionAnsweringPipeline
    - __call__

### TextClassificationPipeline [[transformers.TextClassificationPipeline]]

[[autodoc]] TextClassificationPipeline
    - __call__
    - all

### TextGenerationPipeline [[transformers.TextGenerationPipeline]]

[[autodoc]] TextGenerationPipeline
    - __call__
    - all

### Text2TextGenerationPipeline [[transformers.Text2TextGenerationPipeline]]

[[autodoc]] Text2TextGenerationPipeline
    - __call__
    - all

### TokenClassificationPipeline [[transformers.TokenClassificationPipeline]]

[[autodoc]] TokenClassificationPipeline
    - __call__
    - all

### TranslationPipeline [[transformers.TranslationPipeline]]

[[autodoc]] TranslationPipeline
    - __call__
    - all

### ZeroShotClassificationPipeline [[transformers.ZeroShotClassificationPipeline]]

[[autodoc]] ZeroShotClassificationPipeline
    - __call__
    - all

## 멀티모달 [[multimodal]]

멀티모달 작업에 사용할 수 있는 파이프라인은 다음과 같습니다.

### DocumentQuestionAnsweringPipeline [[transformers.DocumentQuestionAnsweringPipeline]]

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline [[transformers.FeatureExtractionPipeline]]

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageFeatureExtractionPipeline [[transformers.ImageFeatureExtractionPipeline]]

[[autodoc]] ImageFeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline [[transformers.ImageToTextPipeline]]

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### ImageTextToTextPipeline [[transformers.ImageTextToTextPipeline]]

[[autodoc]] ImageTextToTextPipeline
    - __call__
    - all

### MaskGenerationPipeline [[transformers.MaskGenerationPipeline]]

[[autodoc]] MaskGenerationPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline [[transformers.VisualQuestionAnsweringPipeline]]

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline` [[transformers.Pipeline]]

[[autodoc]] Pipeline
