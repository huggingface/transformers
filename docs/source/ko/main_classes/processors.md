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

# 프로세서[[processors]]

Transformers 라이브러리에서 프로세서는 두 가지 다른 의미를 가질 수 있습니다:
- [Wav2Vec2](../model_doc/wav2vec2) (음성과 텍스트) 또는 [CLIP](../model_doc/clip) (텍스트와 비전)과 같은 멀티모달 모델의 입력을 전처리하는 객체
- GLUE 또는 SQUAD 데이터를 전처리하기 위해 라이브러리의 이전 버전에서 사용되었던 deprecated 객체

## 멀티모달 프로세서[[multi-modal-processors]]

모든 멀티모달 모델은 여러 양식(텍스트, 비전, 오디오 중)을 그룹화하는 데이터를 인코딩하거나 디코딩하는 객체가 필요합니다. 이는 토크나이저(텍스트 양식용), 이미지 프로세서(비전용), 피처 추출기(오디오용)와 같은 두 개 이상의 처리 객체를 함께 그룹화하는 프로세서라고 불리는 객체에 의해 처리됩니다.

이러한 프로세서는 저장 및 로딩 기능을 구현하는 다음 기본 클래스를 상속받습니다:

[[autodoc]] ProcessorMixin

## Deprecated 프로세서[[deprecated-processors]]

모든 프로세서는 [`~data.processors.utils.DataProcessor`]와 같은 동일한 아키텍처를 따릅니다. 프로세서는 [`~data.processors.utils.InputExample`]의 목록을 반환합니다. 이 [`~data.processors.utils.InputExample`]들은 모델에 입력하기 위해 [`~data.processors.utils.InputFeatures`]로 변환될 수 있습니다.

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/)는 다양한 기존 NLU 작업에서 모델의 성능을 평가하는 벤치마크입니다. [GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7) 논문과 함께 발표되었습니다.

이 라이브러리는 MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB, QQP, QNLI, RTE, WNLI 작업에 대해 총 10개의 프로세서를 호스팅합니다.

이러한 프로세서들은 다음과 같습니다:

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

또한, 다음 메서드를 사용하여 데이터 파일에서 값을 로드하고 [`~data.processors.utils.InputExample`] 목록으로 변환할 수 있습니다.

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI

[The Cross-Lingual NLI Corpus (XNLI)](https://www.nyu.edu/projects/bowman/xnli/)는 교차언어 텍스트 표현의 품질을 평가하는 벤치마크입니다. XNLI는 [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/)를 기반으로 한 크라우드소싱 데이터셋입니다: 텍스트 쌍은 15개의 다른 언어(영어와 같은 고자원 언어와 스와힐리어와 같은 저자원 언어 모두 포함)에 대해 텍스트 함의 어노테이션으로 라벨링됩니다.

[XNLI: Evaluating Cross-lingual Sentence Representations](https://huggingface.co/papers/1809.05053) 논문과 함께 발표되었습니다.

이 라이브러리는 XNLI 데이터를 로드하는 프로세서를 호스팅합니다:

- [`~data.processors.utils.XnliProcessor`]

테스트 세트에서 골드 라벨을 사용할 수 있으므로 테스트 세트에서 평가가 수행된다는 점에 유의하세요.

이러한 프로세서를 사용하는 예시는 [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) 스크립트에 제공되어 있습니다.


## SQuAD

[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//)는 질문 답변에서 모델의 성능을 평가하는 벤치마크입니다. v1.1과 v2.0 두 가지 버전이 사용 가능합니다. 첫 번째 버전(v1.1)은 [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://huggingface.co/papers/1606.05250) 논문과 함께 발표되었습니다. 두 번째 버전(v2.0)은 [Know What You Don't Know: Unanswerable Questions for SQuAD](https://huggingface.co/papers/1806.03822) 논문과 함께 발표되었습니다.

이 라이브러리는 두 버전 각각에 대한 프로세서를 호스팅합니다:

### 프로세서[[processors]]

이러한 프로세서들은 다음과 같습니다:

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

둘 다 추상 클래스 [`~data.processors.utils.SquadProcessor`]를 상속받습니다.

[[autodoc]] data.processors.squad.SquadProcessor
- all

또한, 다음 메서드를 사용하여 SQuAD 예시를 모델 입력으로 사용할 수 있는 [`~data.processors.utils.SquadFeatures`]로 변환할 수 있습니다.

[[autodoc]] data.processors.squad.squad_convert_examples_to_features


이러한 프로세서들과 앞서 언급한 메서드는 데이터가 포함된 파일뿐만 아니라 *tensorflow_datasets* 패키지와도 함께 사용할 수 있습니다. 예시는 아래에 제공됩니다.


### 사용 예시[[example-usage]]

다음은 데이터 파일을 사용하여 프로세서와 변환 메서드를 사용하는 예시입니다:

```python
# V2 프로세서 로딩
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# V1 프로세서 로딩
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

*tensorflow_datasets* 사용은 데이터 파일 사용만큼 쉽습니다:

```python
# tensorflow_datasets는 Squad V1만 처리합니다.
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

이러한 프로세서를 사용하는 또 다른 예시는 [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) 스크립트에 제공되어 있습니다.