<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Llama2 [[llama2]]

## 개요 [[overview]]

Llama2 모델은 Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Ya1smine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom의 논문 [LLaMA: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)에서 제안되었습니다. 채팅 어플리케이션에 맞게 미세 조정된 체크포인트를 포함된 7B에서 70B 범위의 매개변수를 가진 기초 언어 모델 모음입니다!

논문의 초록은 다음과 같습니다:

*이 연구에서 우리는 70억에서 700억 파라미터의 범위에서 사전 훈련 및 미세 조정된 대규모 언어 모델(LLMs)의 모음인 Llama 2를 개발 및 공개합니다. Llama 2-Chat라고 불리는 미세 조정된 LLMs은 대화 사용 사례에 최적화되었습니다. 우리의 모델은 테스트한 대부분의 벤치마크에서 오픈 소스 채팅 모델보다 성능이 뛰어나며, 유용성과 안전성에 대한 인적 평가를 바탕으로 비공개 소스 모델을 대체할 수 있는 적절한 대안이 될 수 있습니다. 우리는 Llama 2-Chat의 미세 조정 및 안전성 향상의 접근 방식에 대한 자세한 설명을 제공하여 커뮤니티가 우리의 작업을 기반으로 LLMs의 책임있는 개발에 기여할 수 있도록 합니다.*

[여기](https://huggingface.co/models?search=llama2)에서 모든 Llama2 모델을 확인할 수 있습니다.

<Tip warning={true}>

`Llama2` 모델은 `bfloat16`을 사용하여 훈련되었지만, 원래 추론은 `float16`을 사용합니다. 허브에 업로드된 체크포인트는 `dtype = 'float16'`을 사용하며, 이는 `AutoModel` API에 의해 체크포인트를 `torch.float32`에서 `torch.float16`으로 캐스팅하는 데 사용됩니다. 

온라인 가중치의 `dtype`은 `model = AutoModelForCausalLM.from_pretrained("path", dtype = "auto")`를 사용하여 모델을 초기화할 때 `dtype="auto"`를 사용하지 않는 한 대부분 관련이 없습니다. 그 이유는 모델이 먼저 다운로드될 것이고 (온라인 체크포인트의 `dtype`을 사용하여) 그다음에 기본 `dtype`인 `torch`로 캐스팅하고(`torch.float32`가 됨), 마지막으로 구성(configuration)에서 제공된 `dtype`이 있는 경우 이를 사용하기 때문입니다.

모델을 `float16`에서 훈련하는 것은 권장되지 않으며 `nan`을 생성하는 것으로 알려져 있습니다. 따라서 모델은 `bfloat16`에서 훈련되어야 합니다.

</Tip>

🍯 팁:

- Llama2 모델의 가중치는 [이 양식](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)을 작성하여 얻을 수 있습니다.
- 아키텍처는 처음 버전의 Llama와 매우 유사하며, [이 논문](https://huggingface.co/papers/2305.13245)의 내용에 따라 Grouped Query Attention (GQA)이 추가되었습니다.
- `config.pretraining_tp`를 1과 다른 값으로 설정하면 더 정확하지만 느린 선형 레이어 계산이 활성화되어 원본 로짓과 더 잘 일치하게 됩니다.
- 원래 모델은 `pad_id = -1`을 사용하는데, 이는 패딩 토큰이 없음을 의미합니다. 동일한 로직을 사용할 수 없으므로 `tokenizer.add_special_tokens({"pad_token":"<pad>"})`를 사용하여 패딩 토큰을 추가하고 이에 따라 토큰 임베딩 크기를 조정해야 합니다. 또한 `model.config.pad_token_id`를 설정해야 합니다. 모델의 `embed_tokens` 레이어는 `self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)`로 초기화되어, 패딩 토큰 인코딩이 0을 출력하도록 합니다. 따라서 초기화 시에 전달하는 것을 권장합니다.
- 양식을 작성하고 모델 체크포인트 접근 권한을 얻은 후에는 이미 변환된 체크포인트를 사용할 수 있습니다. 그렇지 않고 자신의 모델을 직접 변환하려는 경우, [변환 스크립트](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)를 자유롭게 사용하세요. 스크립트는 다음과 같은 예시의 명령어로 호출할 수 있습니다:

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- 변환 후 모델과 토크나이저는 다음과 같이 로드할 수 있습니다:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

스크립트를 실행하려면 모델을 float16 정밀도로 전부 호스트할 수 있을 만큼 충분한 CPU RAM이 필요합니다 (가장 큰 버전이 여러 체크포인트로 제공되더라도 각 체크포인트는 모델 가중치의 일부만을 포함하므로 모두 RAM에 로드해야 합니다). 75B 모델의 경우, 총 145GB의 RAM이 필요합니다.

- LLaMA 토크나이저는 [sentencepiece](https://github.com/google/sentencepiece)를 기반으로 한 BPE 모델입니다. sentencepiece의 특징 중 하나는 시퀀스를 디코딩할 때 첫 번째 토큰이 단어의 시작이면 (예: "Banana") 토크나이저는 문자열 앞에 접두사 공간을 추가하지 않는 것입니다.

이 모델은 [Arthur Zucker](https://huggingface.co/ArthurZ)가 [Lysandre Debut](https://huggingface.co/lysandre)의 도움을 받아 제공하였습니다. Hugging Face에서의 구현 코드는 [여기](https://github.com/EleutherAI/gpt-neox)의 GPT-NeoX 를 기반으로 합니다. 저자의 원래 코드는 [여기](https://github.com/facebookresearch/llama)에서 찾을 수 있습니다.

## 리소스 [[resources]]

LLaMA2를 시작하는 데 도움이 될 Hugging Face의 공식 및 커뮤니티(🌎로 표시) 리소스 목록입니다. 여기에 새로운 리소스를 추가하기 위해서 Pull Request를 열어 주시면 검토하겠습니다! 리소스는 기존 리소스와 중복되지 않는 새로운 것을 보여주는 것이 이상적입니다.

- [Llama 2 is here - get it on Hugging Face](https://huggingface.co/blog/llama2), Llama 2에 관한 블로그 포스트와 🤗 Transformers 및 🤗 PEFT와 함께 사용하는 방법에 대한 내용입니다.
- [LLaMA 2 - Every Resource you need](https://www.philschmid.de/llama-2), LLaMA 2에 대해 알아보고 빠르게 시작하는 데 필요한 관련 리소스의 모음입니다.

<PipelineTag pipeline="text-generation"/>

- Google Colab에서 QLoRA와 4-bit 정밀도를 사용하여 Llama 2를 미세 조정하는 방법에 대한 [노트북](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing)입니다. 🌎
- "Llama-v2-7b-guanaco" 모델을 4-bit QLoRA로 미세 조정하고 PDF에서 Q&A 데이터셋을 생성하는 방법에 대한 [노트북](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing)입니다. 🌎

⚗️ 최적화
- [Llama 2를 DPO로 미세 조정하기](https://huggingface.co/blog/dpo-trl), TRL 라이브러리의 DPO 방법을 사용하여 특정 데이터셋에서 Llama 2를 미세 조정하는 방법을 안내하는 가이드입니다.
- [확장 가이드: Llama 2 명령어 조정](https://www.philschmid.de/instruction-tune-llama-2), 입력에서 명령어를 생성하도록 Llama 2를 훈련시키는 방법을 안내하는 가이드로, 명령어를 따르는 모델에서 명령어를 주는 모델로 변환합니다.
- 개인 컴퓨터에서 QLoRA와 TRL을 사용하여 Llama 2 모델을 미세 조정하는 방법에 대한 [노트북](https://colab.research.google.com/drive/1SYpgFpcmtIUzdE7pxqknrM4ArCASfkFQ?usp=sharing)입니다. 🌎

⚡️ 추론
- AutoGPTQ 라이브러리의 GPTQ를 사용하여 Llama 2 모델을 양자화하는 방법에 대한 [노트북](https://colab.research.google.com/drive/1TC56ArKerXUpbgRy5vM3woRsbTEVNq7h?usp=sharing)입니다. 🌎
- 로컬 컴퓨터나 Google Colab에서 4-bit 양자화로 Llama 2 채팅 모델을 실행하는 방법에 대한 [노트북](https://colab.research.google.com/drive/1X1z9Q6domMKl2CnEM0QGHNwidLfR4dW2?usp=sharing)입니다. 🌎

🚀 배포
- [Amazon SageMaker에서 LLaMA 2 (7-70B) 미세 조정하기](https://www.philschmid.de/sagemaker-llama2-qlora), Amazon SageMaker에서 QLoRA 미세 조정 및 배포에 이르기까지의 완전한 가이드입니다.
- [Amazon SageMaker에서 Llama 2 7B/13B/70B 배포하기](https://www.philschmid.de/sagemaker-llama-llm), 안전하고 확장 가능한 배포를 위해 Hugging Face의 LLM DLC 컨테이너를 사용하는 방법에 대한 가이드입니다.


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
