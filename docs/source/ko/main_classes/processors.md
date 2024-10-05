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

# 프로세서 [[processors]]

프로세서는 트랜스포머 라이브러리에서 두 가지 의미를 가집니다:
- [Wav2Vec2](../model_doc/wav2vec2) (음성 및 텍스트) 또는 [CLIP](../model_doc/clip)(텍스트 및 비전)과 같은 멀티 모달 모델의 입력을 전처리하는 객체
- 이전 버전의 라이브러리에서 GLUE 또는 SQUAD의 데이터를 전처리하기 위해 사용되었던 더 이상 지원되지 않는 객체

## 멀티 모달 프로세서 [[transformers.ProcessorMixin]]

멀티 모달 모델은 텍스트, 비전(이미지), 오디오와 같은 여러 모달리티의 데이터를 인코딩 또는 디코딩하는 객체를 필요로 합니다. 이러한 작업은 프로세서라는 객체에 의해 처리되며, 텍스트 모달리티의 토크나이저, 비전(이미지)의 이미지 프로세서, 오디오의 특징 추출기와 같은 두 개 이상의 프로세싱 객체를 그룹화합니다.

이 프로세서들은 저장 및 불러오기 기능을 구현하는 아래의 기본 클래스를 상속합니다.

[[autodoc]] ProcessorMixin

## 더 이상 지원되지 않는 프로세서 [[transformers.DataProcessor]]

모든 프로세서는 [`~data.processors.utils.DataProcessor`]의 아키텍처를 따릅니다. 
이 프로세서는 [`~data.processors.utils.InputExample`] 목록을 리턴합니다. 
이러한 [`~data.processors.utils.InputExample`]은 모델에 입력될 수 있도록 [`~data.processors.utils.InputFeatures`]로 변환될 수 있습니다.


[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE [[transformers.glue_convert_examples_to_features]]

[일반 언어 이해 평가 (GLUE)](https://gluebenchmark.com/) 는 기존의 다양한 자연어 이해(NLU) 과제를 통해 모델 성능을 평가하는 벤치마크입니다. 이는 논문 [GLUE: 멀티태스크 벤치마크와 자연어 이해를 위한 분석 플랫폼](https://openreview.net/pdf?id=rJ4km2R5t7)와 함께 발표되었습니다.

라이브러리 GLUE는 10개의 프로세서를 제공해 다음과 같은 과제들을 수행합니다 : MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB, QQP, QNLI, RTE, WNLI.

제공하는 프로세서들은 다음과 같습니다:

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]


또한, 아래의 메서드를 사용하여 데이터 파일에서 값을 불러와 [`~data.processors.utils.InputExample`] 목록으로 변환할 수 있습니다.

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI [[xnli]]
[교차 언어적 NLI 코퍼스 (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) 는 교차 언어 텍스트 표현의 품질을 평가하는 벤치마크입니다. XNLI는 [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/)를 기반으로 크라우드소싱으로 구축된 데이터셋으로, 15개의 다른 언어(영어-리소스가 많은 언어들- 및 스와힐리어 -리소스가 충분하지 않은 언어들- 포함)에 대한 텍스트 추론 주석이 달린 텍스트 쌍으로 구성되어 있습니다.

이 데이터셋은 논문 [XNLI: 교차 언어적 문장 표현 평가](https://arxiv.org/abs/1809.05053)와 함께 발표되었습니다.

이 라이브러리는 XNLI 데이터를 불러오는 프로세서를 제공합니다:

[`~data.processors.utils.XnliProcessor`]
참고로, 테스트셋에 대한 골드 레이블(정답 레이블)이 제공되므로, 테스트셋에서 평가가 수행됩니다.

프로세서 사용 예시는 [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) 스크립트에서 확인할 수 있습니다.



## SQuAD [[squad]]
[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//)은 질의 응답 작업에서 모델의 성능을 평가하는 벤치마크입니다. 두 가지 버전 v1.1과 v2.0이 있습니다. 첫 번째 버전(v1.1)은 논문 [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)과 함께 발표되었습니다. 두 번째 버전(v2.0)은 논문 [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)와 함께 발표되었습니다.

이 라이브러리는 각각의 버전에 해당하는 프로세서를 제공합니다:

### 프로세서 [[transformers.data.processors.squad.SquadProcessor]]

제공하는 프로세서들은 다음과 같습니다:

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]
이들은 모두 추상 클래스 [`~data.processors.utils.SquadProcessor`]를 상속받습니다.

[[autodoc]] data.processors.squad.SquadProcessor - 전체

추가로, SQuAD 예제를 [`~data.processors.utils.SquadFeatures`]로 변환하여 모델 입력으로 사용할 수 있는 메서드도 제공됩니다.


[[autodoc]] data.processors.squad.squad_convert_examples_to_features

이 프로세서들과 앞서 언급한 메서드는 데이터 파일뿐만 아니라 *tensorflow_datasets* 패키지와 함께 사용할 수 있습니다. 예시는 아래에 나와 있습니다.


### 사용 예시 [[example-usage]]

다음은 프로세서와 변환 메서드를 데이터 파일과 함께 사용하는 예시입니다:

```python
# V2 프로세서 로드
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# V1 프로세서 로드
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

 *tensorflow_datasets* 를 사용하는 것은 데이터 파일을 사용하는 것 처럼 쉽습니다 :

```python
# tensorflow_datasets은 Squad V1만 다룹니다.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```
이 프로세서의 또다른 사용 사례는 [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) 스크립트에서 확인 할 수 있습니다.
