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

# RAG(검색 증강 생성) [[rag]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=rag">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-rag-blueviolet">
</a>
</div>

## 개요 [[overview]]

검색 증강 생성(Retrieval-augmented generation, "RAG") 모델은 사전 훈련된 밀집 검색(DPR)과 시퀀스-투-시퀀스 모델의 장점을 결합합니다. RAG 모델은 문서를 검색하고, 이를 시퀀스-투-시퀀스 모델에 전달한 다음, 주변화(marginalization)를 통해 출력을 생성합니다. 검색기와 시퀀스-투-시퀀스 모듈은 사전 훈련된 모델로 초기화되며, 함께 미세 조정되어 검색과 생성 모두 다운스트림 작업(모델을 특정 태스크에 적용하는 것)에 적응할 수 있게 합니다.

이 모델은 Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela의 논문 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)를 기반으로 합니다.

논문의 초록은 다음과 같습니다.

*대규모 사전 훈련 언어 모델들은 그들의 매개변수에 사실적 지식을 저장하고 있으며, 다운스트림 NLP 작업에 대해 미세 조정될 때 최첨단 결과를 달성합니다. 그러나 지식에 접근하고 정확하게 조작하는 능력은 여전히 제한적이며, 따라서 지식 집약적 작업에서 그들의 성능은 작업별 아키텍처에 비해 뒤떨어집니다. 또한, 그들의 결정에 대한 근거를 제공하고 세계 지식을 업데이트하는 것은 여전히 열린 연구 문제로 남아 있습니다. 명시적 비매개변수 메모리에 대한 미분 가능한 접근 메커니즘을 가진 사전 훈련 모델은 이 문제를 극복할 수 있지만, 지금까지는 추출적 다운스트림 작업에 대해서만 연구되었습니다. 우리는 언어 생성을 위해 사전 훈련된 매개변수 및 비매개변수 메모리를 결합하는 모델인 검색 증강 생성(RAG)에 대한 일반적인 목적의 미세 조정 방법을 탐구합니다. 우리는 매개변수 메모리가 사전 훈련된 시퀀스-투-시퀀스 모델이고 비매개변수 메모리가 사전 훈련된 신경 검색기로 접근되는 위키피디아의 밀집 벡터 인덱스인 RAG 모델을 소개합니다. 우리는 생성된 전체 시퀀스에 걸쳐 동일한 검색된 구절을 조건으로 하는 RAG 공식과 토큰별로 다른 구절을 사용할 수 있는 RAG 공식을 비교합니다. 우리는 광범위한 지식 집약적 NLP 작업에 대해 모델을 미세 조정하고 평가하며, 매개변수 시퀀스-투-시퀀스 모델과 작업별 검색-추출 아키텍처를 능가하여 세 가지 개방형 도메인 QA 작업에서 최첨단 성능을 달성합니다. 언어 생성 작업의 경우, RAG 모델이 최첨단 매개변수 전용 시퀀스-투-시퀀스 기준선보다 더 구체적이고, 다양하며, 사실적인 언어를 생성한다는 것을 발견했습니다.*

이 모델은 [ola13](https://huggingface.co/ola13)에 의해 기여되었습니다.

## 사용 팁 [[usage-tips]]

검색 증강 생성(Retrieval-augmented generation, "RAG") 모델은 사전 훈련된 밀집 검색(DPR)과 시퀀스-투-시퀀스 모델의 강점을 결합합니다. RAG 모델은 문서를 검색하고, 이를 시퀀스-투-시퀀스 모델에 전달한 다음, 주변화(marginalization)를 통해 출력을 생성합니다. 검색기와 시퀀스-투-시퀀스 모듈은 사전 훈련된 모델로 초기화되며, 함께 미세 조정됩니다. 이를 통해 검색과 생성 모두 다운스트림 작업에 적응할 수 있게 됩니다.

## RagConfig [[transformers.RagConfig]]

[[autodoc]] RagConfig

## RagTokenizer [[transformers.RagTokenizer]]

[[autodoc]] RagTokenizer

## Rag specific outputs [[transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput]]

[[autodoc]] models.rag.modeling_rag.RetrievAugLMMarginOutput

[[autodoc]] models.rag.modeling_rag.RetrievAugLMOutput

## RagRetriever [[transformers.RagRetriever]]

[[autodoc]] RagRetriever

<frameworkcontent>
<pt>

## RagModel [[transformers.RagModel]]

[[autodoc]] RagModel
    - forward

## RagSequenceForGeneration [[transformers.RagSequenceForGeneration]]

[[autodoc]] RagSequenceForGeneration
    - forward
    - generate

## RagTokenForGeneration [[transformers.RagTokenForGeneration]]

[[autodoc]] RagTokenForGeneration
    - forward
    - generate

</pt>
<tf>

## TFRagModel [[transformers.TFRagModel]]

[[autodoc]] TFRagModel
    - call

## TFRagSequenceForGeneration [[transformers.TFRagSequenceForGeneration]]

[[autodoc]] TFRagSequenceForGeneration
    - call
    - generate

## TFRagTokenForGeneration [[transformers.TFRagTokenForGeneration]]

[[autodoc]] TFRagTokenForGeneration
    - call
    - generate

</tf>
</frameworkcontent>
