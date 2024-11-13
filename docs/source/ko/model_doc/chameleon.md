<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chameleon [[chameleon]]

## 개요 [[overview]]

Chameleon 모델은 META AI Chameleon 팀의 논문 [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818v1)에서 제안되었습니다. Chameleon은 벡터 양자화를 사용하여 이미지를 토큰화함으로써 멀티모달 출력을 생성할 수 있는 비전-언어 모델입니다. 이 모델은 교차된 형식을 포함한 이미지와 텍스트를 입력으로 받으며, 텍스트 응답을 생성합니다. 이미지 생성 모듈은 아직 공개되지 않았습니다.

논문의 초록은 다음과 같습니다:

*우리는 이미지와 텍스트를 임의의 순서로 이해하고 생성할 수 있는 early-fusion 토큰 기반의 혼합 모달(mixed-modal) 모델의 일종인 Chameleon을 소개합니다. 우리는 초기부터 안정적인 훈련 접근법, 정렬 방법, 그리고 early-fusion, 토큰 기반, 혼합 모달 설정에 맞춘 아키텍처 매개변수를 제시합니다. 이 모델들은 시각적 질문 응답, 이미지 캡션 생성, 텍스트 생성, 이미지 생성, 장문 혼합 모달 생성 등 포괄적인 작업 범위에서 평가되었습니다. Chameleon은 단일 모델에서 이미지 캡션 생성 작업에서의 최첨단 성능을 포함한 광범위하고 일반적으로 적용 가능한 능력을 보여주며, 텍스트 전용 작업에서 Llama-2를 능가하면서 Mixtral 8x7B와 Gemini-Pro와 같은 모델들 사이에서도 경쟁력을 갖추고 있습니다. 그리고 상당한 성능의 이미지 생성도 수행합니다. 또한 프롬프트나 출력에 이미지와 텍스트의 혼합 시퀀스가 포함된 새로운 장문 혼합 모달 생성 평가에서, 인간의 판단에 따르면 Gemini Pro와 GPT-4V를 포함한 훨씬 더 큰 모델의 성능과 동등하거나 이를 능가합니다. Chameleon은 완전한 멀티모달 문서의 통합 모델링에서 중요한 발전을 보여줍니다.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/chameleon_arch.png"
alt="drawing" width="600"/>

<small>Chameleon은 이미지를 이산적인 토큰으로 변환하기 위해 벡터 양자화 모듈을 통합합니다. 이는 자기회귀 transformer를 사용한 이미지 생성을 가능하게 합니다. <a href="https://arxiv.org/abs/2405.09818v1">원본 논문</a>에서 가져왔습니다.</small>

이 모델은 [joaogante](https://huggingface.co/joaogante)와 [RaushanTurganbay](https://huggingface.co/RaushanTurganbay)가 기여했습니다. 원본 코드는 [여기](https://github.com/facebookresearch/chameleon)에서 찾을 수 있습니다.

## 사용 팁 [[usage-tips]]

- 더 정확한 결과를 위해, 배치 생성 시 `padding_side="left"`를 사용하는 것을 권장합니다. 생성하기 전에 `processor.tokenizer.padding_side = "left"`로 설정하십시오.

- Chameleon은 안전성 정렬을 위해 튜닝되었음을 유의하십시오. 모델이 응답을 거부하는 경우, 열린 질문보다는 더 구체적으로 질문을 해보세요.

- Chameleon은 채팅 형식으로 생성하므로, 생성된 텍스트는 항상 "assistant's turn"으로 표시됩니다. 프로세서를 호출할 때 `return_for_text_completion=True`를 전달하여 텍스트 완성 생성을 활성화할 수 있습니다.

> [!NOTE]
> Transformers에서의 Chameleon 구현은 이미지 임베딩을 병합할 위치를 나타내기 위해 특별한 이미지 토큰을 사용합니다. 특별한 이미지 토큰을 위해 새로운 토큰을 추가하지 않고 예약된 토큰 중 하나인 `<reserved08707>`를 사용했습니다. 올바른 생성을 위해 프롬프트에서 이미지가 임베딩될 위치에 `<image>`를 추가해야 합니다.

## 사용 예제 [[usage-example]]

### 단일 이미지 추론 [[single-image-inference]]

Chameleon은 게이티드(gated) 모델이므로 Hugging Face Hub에 대한 액세스 권한이 있고 토큰으로 로그인했는지 확인하세요. 다음은 모델을 로드하고 반정밀도(`torch.bfloat16`)로 추론하는 방법입니다:

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# 이미지와 텍스트 프롬프트 준비
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = "이 이미지에서 무엇을 보나요?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# 프롬프트를 자기회귀적으로 완성
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 다중 이미지 추론 [[multi-image-inference]]

Chameleon은 여러 이미지를 입력으로 받아들이며, 이미지들은 동일한 프롬프트에 속하거나 다른 프롬프트에 속할 수 있습니다(배치 추론에서). 다음은 그 방법입니다:

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# 세 가지 다른 이미지 가져오기
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# 배치된 프롬프트 준비: 첫 번째는 다중 이미지 프롬프트이고 두 번째는 단일 이미지 프롬프트입니다
prompts = [
    "이 이미지들은 무엇이 공통점인가요?<image><image>",
    "<image>이 이미지에 무엇이 나타나 있나요?"
]

# 이미지들을 텍스트 프롬프트에서 사용되어야 하는 순서대로 입력할 수 있습니다
# 각 "<image>" 토큰은 하나의 이미지를 사용하며, 다음 "<image>" 토큰은 다음 이미지를 사용합니다
inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

# 생성
generate_ids = model.generate(**inputs, max_new_tokens=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## 모델 최적화 [[model-optimization]]

### Bitsandbytes를 사용한 양자화 [[quantization-using-bitsandbytes]]

모델은 8비트 또는 4비트로 로드할 수 있으며, 이는 원본 모델의 성능을 유지하면서 메모리 요구 사항을 크게 줄여줍니다. 먼저 bitsandbytes를 설치하고(`pip install bitsandbytes`), 라이브러리가 지원하는 GPU/가속기를 사용 중인지 확인하십시오.

<Tip>

bitsandbytes는 CUDA 이외의 여러 백엔드를 지원하도록 리팩터링되고 있습니다. 현재 ROCm(AMD GPU) 및 Intel CPU 구현이 성숙 단계이며, Intel XPU는 진행 중이고 Apple Silicon 지원은 Q4/Q1에 예상됩니다. 설치 지침 및 최신 백엔드 업데이트는 [이 링크](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)를 방문하세요.

전체 공개 전에 버그를 식별하는 데 도움이 되는 피드백을 환영합니다! 자세한 내용과 피드백은 [이 문서](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends)를 확인하세요.

</Tip>

위의 코드 스니펫을 다음과 같이 변경하면 됩니다:

```python
from transformers import ChameleonForConditionalGeneration, BitsAndBytesConfig

# 모델 양자화 방식 지정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", quantization_config=quantization_config, device_map="cuda")
```

### Flash-Attention 2와 SDPA를 사용하여 생성 속도 향상 [[use-flash-attention-2-and-sdpa-to-further-speed-up-generation]]

이 모델은 최적화를 위해 Flash-Attention 2와 PyTorch의 [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)를 모두 지원합니다. SDPA는 모델을 로드할 때 기본 옵션입니다. Flash Attention 2로 전환하려면 먼저 flash-attn을 설치해야 합니다. 해당 패키지 설치에 대해서는 [원본 리포지토리](https://github.com/Dao-AILab/flash-attention)를 참고하십시오. 위의 코드 스니펫을 다음과 같이 변경하면 됩니다:

```python
from transformers import ChameleonForConditionalGeneration

model_id = "facebook/chameleon-7b"
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
).to(0)
```

## ChameleonConfig [[transformers.ChameleonConfig]]

[[autodoc]] ChameleonConfig

## ChameleonVQVAEConfig [[transformers.ChameleonVQVAEConfig]]

[[autodoc]] ChameleonVQVAEConfig

## ChameleonProcessor [[transformers.ChameleonProcessor]]

[[autodoc]] ChameleonProcessor

## ChameleonImageProcessor [[transformers.ChameleonImageProcessor]]

[[autodoc]] ChameleonImageProcessor
    - preprocess

## ChameleonVQVAE [[transformers.ChameleonVQVAE]]

[[autodoc]] ChameleonVQVAE
    - forward

## ChameleonModel [[transformers.ChameleonModel]]

[[autodoc]] ChameleonModel
    - forward

## ChameleonForConditionalGeneration [[transformers.ChameleonForConditionalGeneration]]

[[autodoc]] ChameleonForConditionalGeneration
    - forward
