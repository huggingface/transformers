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

# OpenAI GPT [[openai-gpt]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=openai-gpt">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-openai--gpt-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/openai-gpt">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요 [[overview]]

OpenAI GPT 모델은 Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever가 작성한 [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 논문에서 제안되었습니다. 이는 Toronto Book Corpus와 같은 장기 의존성을 가진 대규모 말뭉치를 사용하여 언어 모델링으로 사전 학습된 인과적(단방향) 트랜스포머입니다.

논문의 초록은 다음과 같습니다:

*자연어 이해는 텍스트 함의, 질문 응답, 의미 유사성 평가, 문서 분류와 같은 다양한 작업을 포함합니다. 비록 대규모의 레이블이 없는 텍스트 말뭉치가 풍부하기는 하지만, 이러한 특정 작업에 대한 학습을 위한 레이블된 데이터는 부족하여 판별적으로 학습된 모델이 적절하게 성능을 발휘하기 어렵습니다. 우리는 다양한 레이블이 없는 텍스트 말뭉치에 대한 언어 모델의 생성적 사전 학습을 수행하고, 각 특정 과제에 대한 판별적 미세 조정을 수행함으로써 이러한 과제에서 큰 성과를 달성할 수 있음을 보여줍니다. 이전 접근 방식과 달리, 우리는 모델 아키텍처에 최소한의 변화를 요구하면서 효과적인 전이를 달성하기 위해 미세 조정 중에 과제 인식 입력 변환(task-aware input transformation)을 사용합니다. 우리는 자연어 이해를 위한 다양한 벤치마크에서 우리의 접근 방식의 효과를 입증합니다. 우리의 general task-agnostic 모델은 각 과제에 특별히 설계된 아키텍처를 사용하는 판별적으로 학습된 모델보다 뛰어나며, 연구된 12개 과제 중 9개 부문에서 최첨단 성능(state of the art)을 크게 향상시킵니다.*

[Write With Transformer](https://transformer.huggingface.co/doc/gpt)는 Hugging Face가 만든 웹 애플리케이션으로, 여러 모델의 생성 능력을 보여주며 그 중에는 GPT도 포함되어 있습니다.

이 모델은 [thomwolf](https://huggingface.co/thomwolf)에 의해 기여되었으며, 원본 코드는 [여기](https://github.com/openai/finetune-transformer-lm)에서 확인할 수 있습니다.

## 사용 팁 [[usage-tips]]

- GPT는 절대 위치 임베딩을 사용하는 모델이므로 입력을 일반적으로 왼쪽보다는 오른쪽에 패딩하는 것이 권장됩니다.
- GPT는 인과 언어 모델링(Causal Language Modeling, CLM) 목표로 학습되었기 때문에 시퀀스에서 다음 토큰을 예측하는 데 강력한 성능을 보여줍니다. 이를 활용하면 *run_generation.py* 예제 스크립트에서 볼 수 있듯이 GPT-2는 구문적으로 일관된 텍스트를 생성할 수 있습니다.

참고:

*OpenAI GPT* 논문의 원래 토큰화 과정을 재현하려면 `ftfy`와 `SpaCy`를 설치해야 합니다:

```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

`ftfy`와 `SpaCy`를 설치하지 않으면 [`OpenAIGPTTokenizer`]는 기본적으로 BERT의 `BasicTokenizer`를 사용한 후 Byte-Pair Encoding을 통해 토큰화합니다(대부분의 사용에 문제가 없으니 걱정하지 마세요).

## 리소스 [[resources]]

OpenAI GPT를 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티(🌎 표시) 리소스 목록입니다. 여기에 리소스를 추가하고 싶다면, Pull Request를 열어주시면 검토하겠습니다! 리소스는 기존 리소스를 복제하지 않고 새로운 것을 보여주는 것이 좋습니다.

<PipelineTag pipeline="text-classification"/>

- [SetFit을 사용하여 텍스트 분류에서 OpenAI GPT-3을 능가하는 방법](https://www.philschmid.de/getting-started-setfit) 블로그 게시물.
- 추가 자료: [텍스트 분류 과제 가이드](../tasks/sequence_classification)

<PipelineTag pipeline="text-generation"/>

- [Hugging Face와 함께 비영어 GPT-2 모델을 미세 조정하는 방법](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface) 블로그.
- GPT-2와 함께 [Transformers를 사용한 언어 생성의 다양한 디코딩 방법](https://huggingface.co/blog/how-to-generate)에 대한 블로그.
- [Scratch에서 CodeParrot 🦜을 훈련하는 방법](https://huggingface.co/blog/codeparrot), 대규모 GPT-2 모델에 대한 블로그.
- GPT-2와 함께 [TensorFlow 및 XLA를 사용한 더 빠른 텍스트 생성](https://huggingface.co/blog/tf-xla-generate)에 대한 블로그.
- [Megatron-LM으로 언어 모델을 훈련하는 방법](https://huggingface.co/blog/megatron-training)에 대한 블로그.
- [좋아하는 아티스트의 스타일로 가사를 생성하도록 GPT2를 미세 조정하는 방법](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb)에 대한 노트북. 🌎
- [좋아하는 트위터 사용자의 스타일로 트윗을 생성하도록 GPT2를 미세 조정하는 방법](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb)에 대한 노트북. 🌎
- 🤗 Hugging Face 코스의 [인과 언어 모델링](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) 장.
- [`OpenAIGPTLMHeadModel`]은 [인과 언어 모델링 예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [텍스트 생성 예제 스크립트](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에 의해 지원됩니다.
- [`TFOpenAIGPTLMHeadModel`]은 [인과 언어 모델링 예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에 의해 지원됩니다.
- 추가 자료: [인과 언어 모델링 과제 가이드](../tasks/language_modeling)

<PipelineTag pipeline="token-classification"/>

- [Byte-Pair Encoding 토큰화](https://huggingface.co/course/en/chapter6/5)에 대한 강의 자료.

## OpenAIGPTConfig [[transformers.OpenAIGPTConfig]]

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer [[transformers.OpenAIGPTTokenizer]]

[[autodoc]] OpenAIGPTTokenizer
    - save_vocabulary

## OpenAIGPTTokenizerFast [[transformers.OpenAIGPTTokenizerFast]]

[[autodoc]] OpenAIGPTTokenizerFast

## OpenAI specific outputs [[transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput]]

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

[[autodoc]] models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput

<frameworkcontent>
<pt>

## OpenAIGPTModel [[transformers.OpenAIGPTModel]]

[[autodoc]] OpenAIGPTModel
    - forward

## OpenAIGPTLMHeadModel [[transformers.OpenAIGPTLMHeadModel]]

[[autodoc]] OpenAIGPTLMHeadModel
    - forward

## OpenAIGPTDoubleHeadsModel [[transformers.OpenAIGPTDoubleHeadsModel]]

[[autodoc]] OpenAIGPTDoubleHeadsModel
    - forward

## OpenAIGPTForSequenceClassification [[transformers.OpenAIGPTForSequenceClassification]]

[[autodoc]] OpenAIGPTForSequenceClassification
    - forward

</pt>
<tf>

## TFOpenAIGPTModel [[transformers.TFOpenAIGPTModel]]

[[autodoc]] TFOpenAIGPTModel
    - call

## TFOpenAIGPTLMHeadModel [[transformers.TFOpenAIGPTLMHeadModel]]

[[autodoc]] TFOpenAIGPTLMHeadModel
    - call

## TFOpenAIGPTDoubleHeadsModel [[transformers.TFOpenAIGPTDoubleHeadsModel]]

[[autodoc]] TFOpenAIGPTDoubleHeadsModel
    - call

## TFOpenAIGPTForSequenceClassification [[transformers.TFOpenAIGPTForSequenceClassification]]

[[autodoc]] TFOpenAIGPTForSequenceClassification
    - call

</tf>
</frameworkcontent>
