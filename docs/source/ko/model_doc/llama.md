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

# LLaMA [[llama]]

## 개요 [[overview]]

LLaMA 모델은 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample에 의해 제안된 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 에서 소개되었습니다. 이 모델은 7B에서 65B 파라미터까지 다양한 크기의 기초 언어 모델의 모음입니다.

논문의 초록은 다음과 같습니다:

*"우리는 7B에서 65B 파라미터까지 다양한 크기의 기초 언어 모델 모음인 LLaMA를 소개합니다. 우리는 이 모델을 수조 토큰에서 훈련시키고, 공개적으로 사용 가능한 데이터셋만 사용하여 최첨단 모델을 훈련시킬 수 있음을 보여줍니다. 특히 LLaMA-13B는 대부분의 벤치마크에서 GPT-3 (175B)을 앞서가며, LLaMA-65B는 Chinchilla-70B와 PaLM-540B와 경쟁합니다. 우리는 모든 모델을 연구 커뮤니티에 공개합니다."*

팁:

- LLaMA 모델의 가중치는 [이 양식](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)을 작성하여 얻을 수 있습니다.
- 가중치를 다운로드한 후, 이를 Hugging Face Transformers 형식으로 변환해야합니다. 변환 스크립트는 다음과 같이 호출할 수 있습니다 (예제):

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- 변환 후, 모델과 토크나이저는 다음과 같이 로드할 수 있습니다:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

스크립트 실행에는 충분한 CPU RAM이 필요하며, 65B 모델의 경우 130GB의 RAM이 필요합니다.

- LLaMA 토크나이저는 [sentencepiece](https://github.com/google/sentencepiece)를 기반으로 하는 BPE 모델입니다. sentencepiece의 한 가지 독특한 점은 시퀀스를 디코딩할 때 첫 번째 토큰이 단어의 시작일 때 (예: "Banana"), 토크나이저가 문자열 앞에 접두사 공백을 추가하지 않는다는 것입니다.

이 모델은 [zphang](https://huggingface.co/zphang)에 의해 제공되었으며, [BlackSamorez](https://huggingface.co/BlackSamorez)의 기여도 포함되었습니다. Hugging Face에서의 구현 코드는 GPT-NeoX를 기반으로 하며 [여기](https://github.com/EleutherAI/gpt-neox)에서 찾을 수 있습니다. 원본 저자의 코드는 [여기](https://github.com/facebookresearch/llama)에서 찾을 수 있습니다.


원래 LLaMA 모델을 기반으로, Meta AI에서 몇 가지 후속 작업을 출시했습니다:

- **Llama2**: Llama2는 Llama의 개선된 버전으로 일부 구조적 수정 (그룹화된 쿼리 어텐션)을 포함하고 있으며 2조 토큰에 대해 사전 훈련되었습니다. Llama2에 대한 자세한 내용은 [여기](llama2)에서 찾을 수 있습니다.

## 자원 [[resources]]

LLaMA를 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티 (🌎로 표시됨) 자원 목록입니다. 새로운 리소스를 제출하려면 이 리소스가 이미 있는 리소스를 중복하는 대신 무언가 새로운 것을 보여주는 것이 좋습니다.

<PipelineTag pipeline="text-classification"/>

- 텍스트 분류 작업을 위해 LLaMA 모델을 조정하는 방법에 대한 [노트북](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=f04ba4d2). 🌎

<PipelineTag pipeline="question-answering"/>

- [StackLLaMA: RLHF를 사용하여 LLaMA를 훈련하는 손잡이 가이드](https://huggingface.co/blog/stackllama#stackllama-a-hands-on-guide-to-train-llama-with-rlhf), Stack Exchange에서 질문에 답하는 방법에 대한 블로그 포스트. 🌎

⚗️ 최적화
- 제한된 메모리를 가진 GPU에서 xturing 라이브러리를 사용하여 LLaMA 모델을 어떻게 미세 조정하는지에 대한 [노트북](https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing). 🌎

⚡️ 추론
- 🤗 PEFT 라이브러리에서 PeftModel을 사용하여 LLaMA 모델을 어떻게 실행하는지에 대한 [노트북](https://colab.research.google.com/github/DominguesM/alpaca-lora-ptbr-7b/blob/main/notebooks/02%20-%20Evaluate.ipynb). 🌎
- LangChain을 사용하여 PEFT 어댑터 LLaMA 모델을 어떻게 로드하는지에 대한 [노트북](https://colab.research.google.com/drive/1l2GiSSPbajVyp2Nk3CFT4t3uH6-5TiBe?usp=sharing). 🌎

🚀 배포
- 직관적인 UI를 통해 🤗 PEFT 라이브러리를 사용하여 LoRA 방법을 통해 LLaMA 모델을 어떻게 미세 조정하는지에 대한 [노트북](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb#scrollTo=3PM_DilAZD8T). 🌎
- Amazon SageMaker에서 텍스트 생성을 위해 Open-LLaMA 모델을 배포하는 방법에 대한 [노트북](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-open-llama.ipynb). 🌎

## LlamaConfig [[llamaconfig]]

[[autodoc]] LlamaConfig


## LlamaTokenizer [[llamatokenizer]]

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast [[llamatokenizerfast]]

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## LlamaModel [[llamamodel]]

[[autodoc]] LlamaModel
    - forward


## LlamaForCausalLM [[llamaforcausallm]]

[[autodoc]] LlamaForCausalLM
    - forward

## LlamaForSequenceClassification [[llamaforsequenceclassification]]

[[autodoc]] LlamaForSequenceClassification
    - forward
