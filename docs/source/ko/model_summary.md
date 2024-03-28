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

# Transformer 모델군[[the-transformer-model-family]]

2017년에 소개된 [기본 Transformer](https://arxiv.org/abs/1706.03762) 모델은 자연어 처리(NLP) 작업을 넘어 새롭고 흥미로운 모델들에 영감을 주었습니다. [단백질 접힘 구조 예측](https://huggingface.co/blog/deep-learning-with-proteins), [치타의 달리기 훈련](https://huggingface.co/blog/train-decision-transformers), [시계열 예측](https://huggingface.co/blog/time-series-transformers) 등을 위한 다양한 모델이 생겨났습니다. Transformer의 변형이 너무 많아서, 큰 그림을 놓치기 쉽습니다. 하지만 여기 있는 모든 모델의 공통점은 기본 Trasnformer 아키텍처를 기반으로 한다는 점입니다. 일부 모델은 인코더 또는 디코더만 사용하고, 다른 모델들은 인코더와 디코더를 모두 사용하기도 합니다. 이렇게 Transformer 모델군 내 상위 레벨에서의 차이점을 분류하고 검토하면 유용한 분류 체계를 얻을 수 있으며, 이전에 접해보지 못한 Transformer 모델들 또한 이해하는 데 도움이 될 것입니다. 

기본 Transformer 모델에 익숙하지 않거나 복습이 필요한 경우, Hugging Face 강의의 [트랜스포머는 어떻게 동작하나요?](https://huggingface.co/course/chapter1/4?fw=pt) 챕터를 확인하세요. 

<div align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/H39Z_720T5s" title="YouTube video player"
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
    picture-in-picture" allowfullscreen></iframe>
</div>

## 컴퓨터 비전[[computer-vision]]

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FacQBpeFBVvrDUlzFlkejoz%2FModelscape-timeline%3Fnode-id%3D0%253A1%26t%3Dm0zJ7m2BQ9oe0WtO-1" allowfullscreen></iframe> 

### 합성곱 네트워크[[convolutional-network]]

[Vision Transformer](https://arxiv.org/abs/2010.11929)가 확장성과 효율성을 입증하기 전까지 오랫동안 합성곱 네트워크(CNN)가 컴퓨터 비전 작업의 지배적인 패러다임이었습니다. 그럼에도 불구하고, 이동 불변성(translation invariance)과 같은 CNN의 우수한 부분이 도드라지기 때문에 몇몇 (특히 특정 과업에서의) Transformer 모델은 아키텍처에 합성곱을 통합하기도 했습니다. [ConvNeXt](model_doc/convnext)는 이런 관례를 뒤집어 CNN을 현대화하기 위해 Transformer의 디자인을 차용합니다. 예를 들면 ConvNeXt는 겹치지 않는 슬라이딩 창(sliding window)을 사용하여 이미지를 패치화하고, 더 큰 커널로 전역 수용 필드(global receptive field)를 확장시킵니다. ConvNeXt는 또한 메모리 효율을 높이고 성능을 향상시키기 위해 여러 레이어 설계를 선택하기 때문에 Transformer와 견줄만합니다!

### 인코더[[cv-encoder]]

[Vision Transformer(ViT)](model_doc/vit)는 합성곱 없는 컴퓨터 비전 작업의 막을 열었습니다. ViT는 표준 Transformer 인코더를 사용하지만, 가장 큰 혁신은 이미지를 처리하는 방식이었습니다. 문장을 토큰으로 분할하는 것처럼 이미지를 고정된 크기의 패치로 분할하고, 이를 사용하여 임베딩을 생성합니다. ViT는 Transformer의 효율적인 아키텍처를 활용하여 훈련에 더 적은 자원을 사용하면서도 당시 CNN에 비견하는 결과를 입증했습니다. 그리고 ViT를 뒤이어 분할(segmentation)과 같은 고밀도 비전 작업과 탐지 작업도 다룰 수 있는 다른 비전 모델이 등장했습니다.

이러한 모델 중 하나가 [Swin](model_doc/swin) Transformer입니다. 이 모델은 작은 크기의 패치에서 계층적 특징 맵(CNN 👀과 같지만 ViT와는 다름)을 만들고 더 깊은 레이어의 인접 패치와 병합합니다. 어텐션(Attention)은 지역 윈도우 내에서만 계산되며, 모델이 더 잘 학습할 수 있도록 어텐션 레이어 간에 윈도우를 이동하며 연결을 생성합니다. Swin Transformer는 계층적 특징 맵을 생성할 수 있으므로, 분할(segmentation)과 탐지와 같은 고밀도 예측 작업에 적합합니다. [SegFormer](model_doc/segformer) 역시 Transformer 인코더를 사용하여 계층적 특징 맵을 구축하지만, 상단에 간단한 다층 퍼셉트론(MLP) 디코더를 추가하여 모든 특징 맵을 결합하고 예측을 수행합니다. 

BeIT와 ViTMAE와 같은 다른 비전 모델은 BERT의 사전훈련 목표(objective)에서 영감을 얻었습니다. [BeIT](model_doc/beit)는 *마스크드 이미지 모델링(MIM)*으로 사전훈련되며, 이미지 패치는 임의로 마스킹되고 이미지도 시각적 토큰으로 토큰화됩니다. BeIT는 마스킹된 패치에 해당하는 시각적 토큰을 예측하도록 학습됩니다. [ViTMAE](model_doc/vitmae)도 비슷한 사전훈련 목표가 있지만, 시각적 토큰 대신 픽셀을 예측해야 한다는 점이 다릅니다. 특이한 점은 이미지 패치의 75%가 마스킹되어 있다는 것입니다! 디코더는 마스킹된 토큰과 인코딩된 패치에서 픽셀을 재구성합니다. 사전훈련이 끝나면 디코더는 폐기되고 인코더는 다운스트림 작업에 사용할 준비가 됩니다.

### 디코더[[cv-decoder]]

대부분의 비전 모델은 인코더에 의존하여 이미지 표현을 학습하기 때문에 디코더 전용 비전 모델은 드뭅니다. 하지만 이미지 생성 등의 사례의 경우, GPT-2와 같은 텍스트 생성 모델에서 보았듯이 디코더가 가장 적합합니다. [ImageGPT](model_doc/imagegpt)는 GPT-2와 동일한 아키텍처를 사용하지만, 시퀀스의 다음 토큰을 예측하는 대신 이미지의 다음 픽셀을 예측합니다. ImageGPT는 이미지 생성 뿐만 아니라 이미지 분류를 위해 미세 조정할 수도 있습니다. 

### 인코더-디코더[[cv-encoder-decoder]]

비전 모델은 일반적으로 인코더(백본으로도 알려짐)를 사용하여 중요한 이미지 특징을 추출한 후, 이를 Transformer 디코더로 전달합니다. [DETR](model_doc/detr)에 사전훈련된 백본이 있지만, 객체 탐지를 위해 완전한 Transformer 인코더-디코더 아키텍처도 사용합니다. 인코더는 이미지 표현을 학습하고 이를 디코더에서 객체 쿼리(각 객체 쿼리는 이미지의 영역 또는 객체에 중점을 두고 학습된 임베딩)와 결합합니다. DETR은 각 객체 쿼리에 대한 바운딩 박스 좌표와 클래스 레이블을 예측합니다.

## 자연어처리[[natural-language-processing]]

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FUhbQAZDlpYW5XEpdFy6GoG%2Fnlp-model-timeline%3Fnode-id%3D0%253A1%26t%3D4mZMr4r1vDEYGJ50-1" allowfullscreen></iframe>

### 인코더[[nlp-encoder]]

[BERT](model_doc/bert)는 인코더 전용 Transformer로, 다른 토큰을 보고 소위 "부정 행위"를 저지르는 걸 막기 위해 입력에서 특정 토큰을 임의로 마스킹합니다. 사전훈련의 목표는 컨텍스트를 기반으로 마스킹된 토큰을 예측하는 것입니다. 이를 통해 BERT는 왼쪽과 오른쪽 컨텍스트를 충분히 활용하여 입력에 대해 더 깊고 풍부한 표현을 학습할 수 있습니다. 그러나 BERT의 사전훈련 전략에는 여전히 개선의 여지가 남아 있었습니다. [RoBERTa](model_doc/roberta)는 더 긴 시간 동안 더 큰 배치에 대한 훈련을 포함하고, 전처리 중에 한 번만 마스킹하는 것이 아니라 각 에폭에서 토큰을 임의로 마스킹하고, 다음 문장 예측 목표를 제거하는 새로운 사전훈련 방식을 도입함으로써 이를 개선했습니다. 

성능 개선을 위한 전략으로 모델 크기를 키우는 것이 지배적입니다. 하지만 큰 모델을 훈련하려면 계산 비용이 많이 듭니다. 계산 비용을 줄이는 한 가지 방법은 [DistilBERT](model_doc/distilbert)와 같이 작은 모델을 사용하는 것입니다. DistilBERT는 압축 기법인 [지식 증류(knowledge distillation)](https://arxiv.org/abs/1503.02531)를 사용하여, 거의 모든 언어 이해 능력을 유지하면서 더 작은 버전의 BERT를 만듭니다. 

그러나 대부분의 Transformer 모델에 더 많은 매개변수를 사용하는 경향이 이어졌고, 이에 따라 훈련 효율성을 개선하는 것에 중점을 둔 새로운 모델이 등장했습니다. [ALBERT](model_doc/albert)는 두 가지 방법으로 매개변수 수를 줄여 메모리 사용량을 줄였습니다. 바로 큰 어휘를 두 개의 작은 행렬로 분리하는 것과 레이어가 매개변수를 공유하도록 하는 것입니다. [DeBERTa](model_doc/deberta)는 단어와 그 위치를 두 개의 벡터로 개별적으로 인코딩하는 분리된(disentangled) 어텐션 메커니즘을 추가했습니다. 어텐션은 단어와 위치 임베딩을 포함하는 단일 벡터 대신 이 별도의 벡터에서 계산됩니다. [Longformer](model_doc/longformer)는 특히 시퀀스 길이가 긴 문서를 처리할 때, 어텐션을 더 효율적으로 만드는 것에 중점을 두었습니다. 지역(local) 윈도우 어텐션(각 토큰 주변의 고정된 윈도우 크기에서만 계산되는 어텐션)과 전역(global) 어텐션(분류를 위해 `[CLS]`와 같은 특정 작업 토큰에만 해당)의 조합을 사용하여 전체(full) 어텐션 행렬 대신 희소(sparse) 어텐션 행렬을 생성합니다. 

### 디코더[[nlp-decoder]]

[GPT-2](model_doc/gpt2)는 시퀀스에서 다음 단어를 예측하는 디코더 전용 Transformer입니다. 토큰을 오른쪽으로 마스킹하여 모델이 이전 토큰을 보고 "부정 행위"를 하지 못하도록 합니다. GPT-2는 방대한 텍스트에 대해 사전훈련하여 텍스트가 일부만 정확하거나 사실인 경우에도 상당히 능숙하게 텍스트를 생성할 수 있게 되었습니다. 하지만 GPT-2는 BERT가 사전훈련에서 갖는 양방향 컨텍스트가 부족하기 때문에 특정 작업에 적합하지 않았습니다. [XLNET](model_doc/xlnet)은 양방향 훈련이 가능한 permutation language modeling objective(PLM)를 사용하여 BERT와 GPT-2의 사전훈련 목표에 대한 장점을 함께 가지고 있습니다.

GPT-2 이후, 언어 모델은 더욱 거대해졌고 현재는 *대규모 언어 모델(LLM)*로 알려져 있습니다. 충분히 큰 데이터 세트로 사전훈련된 LLM은 퓨샷(few-shot) 또는 제로샷(zero-shot) 학습을 수행합니다. [GPT-J](model_doc/gptj)는 6B 크기의 매개변수가 있고 400B 크기의 토큰으로 훈련된 LLM입니다. GPT-J에 이어 디코더 전용 모델군인 [OPT](model_doc/opt)가 등장했으며, 이 중 가장 큰 모델은 175B 크기이고 180B 크기의 토큰으로 훈련되었습니다. [BLOOM](model_doc/bloom)은 비슷한 시기에 출시되었으며, 이 중 가장 큰 모델은 176B 크기의 매개변수가 있고 46개의 언어와 13개의 프로그래밍 언어로 된 366B 크기의 토큰으로 훈련되었습니다. 

### 인코더-디코더[[nlp-encoder-decoder]]

[BART](model_doc/bart)는 기본 Transformer 아키텍처를 유지하지만, 일부 텍스트 스팬(span)이 단일 `마스크` 토큰으로 대체되는 *text infilling* 변형으로 사전훈련 목표를 수정합니다. 디코더는 변형되지 않은 토큰(향후 토큰은 마스킹됨)을 예측하고 인코더의 은닉 상태를 사용하여 이 작업을 돕습니다. [Pegasus](model_doc/pegasus)는 BART와 유사하지만, Pegasus는 텍스트 스팬 대신 전체 문장을 마스킹합니다. Pegasus는 마스크드 언어 모델링 외에도 gap sentence generation(GSG)로 사전훈련됩니다. GSG는 문서에 중요한 문장 전체를 마스킹하여 `마스크` 토큰으로 대체하는 것을 목표로 합니다. 디코더는 남은 문장에서 출력을 생성해야 합니다. [T5](model_doc/t5)는 특정 접두사를 사용하여 모든 NLP 작업을 텍스트 투 텍스트 문제로 변환하는 더 특수한 모델입니다. 예를 들어, 접두사 `Summarize:`은 요약 작업을 나타냅니다. T5는 지도(GLUE 및 SuperGLUE) 훈련과 자기지도 훈련(토큰의 15%를 임의로 샘플링하여 제거)으로 사전훈련됩니다.

## 오디오[[audio]]

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2Fvrchl8jDV9YwNVPWu2W0kK%2Fspeech-and-audio-model-timeline%3Fnode-id%3D0%253A1%26t%3DmM4H8pPMuK23rClL-1" allowfullscreen></iframe>

### 인코더[[audio-encoder]]

[Wav2Vec2](model_doc/wav2vec2)는 Transformer 인코더를 사용하여 원본 오디오 파형(raw audio waveform)에서 직접 음성 표현을 학습합니다. 허위 음성 표현 세트에서 실제 음성 표현을 판별하는 대조 작업으로 사전훈련됩니다. [HuBERT](model_doc/hubert)는 Wav2Vec2와 유사하지만 훈련 과정이 다릅니다. 타겟 레이블이 유사한 오디오 세그먼트가 클러스터에 할당되어 은닉 단위(unit)가 되는 군집화(clustering) 단계에서 생성됩니다. 은닉 단위는 예측을 위한 임베딩에 매핑됩니다.

### 인코더-디코더[[audio-encoder-decoder]]

[Speech2Text](model_doc/speech_to_text)는 자동 음성 인식(ASR) 및 음성 번역을 위해 고안된 음성 모델입니다. 이 모델은 오디오 파형에서 추출한 log mel-filter bank 특징을 채택하고 자기회귀 방식으로 사전훈련하여, 전사본 또는 번역을 만듭니다. [Whisper](model_doc/whisper)은 ASR 모델이지만, 다른 많은 음성 모델과 달리 제로샷 성능을 위해 대량의 ✨ 레이블이 지정된 ✨ 오디오 전사 데이터에 대해 사전훈련됩니다. 데이터 세트의 큰 묶음에는 영어가 아닌 언어도 포함되어 있어서 자원이 적은 언어에도 Whisper를 사용할 수 있습니다. 구조적으로, Whisper는 Speech2Text와 유사합니다. 오디오 신호는 인코더에 의해 인코딩된 log-mel spectrogram으로 변환됩니다. 디코더는 인코더의 은닉 상태와 이전 토큰으로부터 자기회귀 방식으로 전사를 생성합니다.

## 멀티모달[[multimodal]]

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FcX125FQHXJS2gxeICiY93p%2Fmultimodal%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### 인코더[[mm-encoder]]

[VisualBERT](model_doc/visual_bert)는 BERT 이후에 출시된 비전 언어 작업을 위한 멀티모달 모델입니다. 이 모델은 BERT와 사전훈련된 객체 탐지 시스템을 결합하여 이미지 특징을 시각 임베딩으로 추출하고, 텍스트 임베딩과 함께 BERT로 전달합니다. VisualBERT는 마스킹되지 않은 텍스트와 시각 임베딩을 기반으로 마스킹된 텍스트를 예측하고, 텍스트가 이미지와 일치하는지 예측해야 합니다. ViT가 이미지 임베딩을 구하는 방식이 더 쉬웠기 때문에, ViT가 출시된 후 [ViLT](model_doc/vilt)는 아키텍처에 ViT를 채택했습니다. 이미지 임베딩은 텍스트 임베딩과 함께 처리됩니다. 여기에서, ViLT는 이미지 텍스트 매칭, 마스크드 언어 모델링, 전체 단어 마스킹을 통해 사전훈련됩니다.

[CLIP](model_doc/clip)은 다른 접근 방식을 사용하여 (`이미지`, `텍스트`)의 쌍 예측을 수행합니다. (`이미지`, `텍스트`) 쌍에서의 이미지와 텍스트 임베딩 간의 유사도를 최대화하기 위해 4억 개의 (`이미지`, `텍스트`) 쌍 데이터 세트에 대해 이미지 인코더(ViT)와 텍스트 인코더(Transformer)를 함께 훈련합니다. 사전훈련 후, 자연어를 사용하여 이미지가 주어진 텍스트를 예측하거나 그 반대로 예측하도록 CLIP에 지시할 수 있습니다. [OWL-ViT](model_doc/owlvit)는 CLIP을 제로샷 객체 탐지를 위한 백본(backbone)으로 사용하여 CLIP 상에 구축됩니다. 사전훈련 후, 객체 탐지 헤드가 추가되어 (`클래스`, `바운딩 박스`) 쌍에 대한 집합(set) 예측을 수행합니다.

### 인코더-디코더[[mm-encoder-decoder]]

광학 문자 인식(OCR)은 이미지를 이해하고 텍스트를 생성하기 위해 다양한 구성 요소를 필요로 하는 전통적인 텍스트 인식 작업입니다. [TrOCR](model_doc/trocr)은 종단간(end-to-end) Transformer를 사용하여 이 프로세스를 간소화합니다. 인코더는 이미지 이해를 위한 ViT 방식의 모델이며 이미지를 고정된 크기의 패치로 처리합니다. 디코더는 인코더의 은닉 상태를 받아서 자기회귀 방식으로 텍스트를 생성합니다. [Donut](model_doc/donut)은 OCR 기반 접근 방식에 의존하지 않는 더 일반적인 시각 문서 이해 모델입니다. 이 모델은 Swin Transformer를 인코더로, 다국어 BART를 디코더로 사용합니다. Donut은 이미지와 텍스트 주석을 기반으로 다음 단어를 예측하여 텍스트를 읽도록 사전훈련됩니다. 디코더는 프롬프트가 주어지면 토큰 시퀀스를 생성합니다. 프롬프트는 각 다운스트림 작업에 대한 특수 토큰으로 표현됩니다. 예를 들어, 문서 파싱(parsing)에는 인코더의 은닉 상태와 결합되어 문서를 정형 출력 형식(JSON)으로 파싱하는 특수 `파싱` 토큰이 있습니다.

## 강화 학습[[reinforcement-learning]]

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FiB3Y6RvWYki7ZuKO6tNgZq%2Freinforcement-learning%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### 디코더[[rl-decoder]]

Decision 및 Trajectory Transformer는 상태(state), 행동(action), 보상(reward)을 시퀀스 모델링 문제로 표현합니다. [Decision Transformer](model_doc/decision_transformer)는 기대 보상(returns-to-go), 과거 상태 및 행동을 기반으로 미래의 원하는 수익(return)으로 이어지는 일련의 행동을 생성합니다. 마지막 *K* 시간 스텝(timestep)에 대해, 세 가지 모달리티는 각각 토큰 임베딩으로 변환되고 GPT와 같은 모델에 의해 처리되어 미래의 액션 토큰을 예측합니다. [Trajectory Transformer](model_doc/trajectory_transformer)도 상태, 행동, 보상을 토큰화하여 GPT 아키텍처로 처리합니다. 보상 조건에 중점을 둔 Decision Transformer와 달리 Trajectory Transformer는 빔 서치(beam search)로 미래 행동을 생성합니다.