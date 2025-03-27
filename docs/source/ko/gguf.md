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

# GGUF와 Transformers의 상호작용 [[gguf-and-interaction-with-transformers]]

GGUF 파일 형식은 [GGML](https://github.com/ggerganov/ggml)과 그에 의존하는 다른 라이브러리, 예를 들어 매우 인기 있는 [llama.cpp](https://github.com/ggerganov/llama.cpp)이나 [whisper.cpp](https://github.com/ggerganov/whisper.cpp)에서 추론을 위한 모델을 저장하는데 사용됩니다.

이 파일 형식은 [Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)에서 지원되며, 파일 내의 텐서와 메타데이터를 신속하게 검사할 수 있는 기능을 제공합니다.

이 형식은 "단일 파일 형식(single-file-format)"으로 설계되었으며, 하나의 파일에 설정 속성, 토크나이저 어휘, 기타 속성뿐만 아니라 모델에서 로드되는 모든 텐서가 포함됩니다. 이 파일들은 파일의 양자화 유형에 따라 다른 형식으로 제공됩니다. 다양한 양자화 유형에 대한 간략한 설명은 [여기](https://huggingface.co/docs/hub/en/gguf#quantization-types)에서 확인할 수 있습니다.

## Transformers 내 지원 [[support-within-transformers]]

`transformers` 내에서 `gguf` 파일을 로드할 수 있는 기능을 추가하여 GGUF 모델의 추가 학습/미세 조정을 제공한 후 `ggml` 생태계에서 다시 사용할 수 있도록 `gguf` 파일로 변환하는 기능을 제공합니다. 모델을 로드할 때 먼저 FP32로 역양자화한 후, PyTorch에서 사용할 수 있도록 가중치를 로드합니다.

> [!NOTE]
> 지원은 아직 초기 단계에 있으며, 다양한 양자화 유형과 모델 아키텍처에 대해 이를 강화하기 위한 기여를 환영합니다.

현재 지원되는 모델 아키텍처와 양자화 유형은 다음과 같습니다:

### 지원되는 양자화 유형 [[supported-quantization-types]]

초기에 지원되는 양자화 유형은 Hub에서 공유된 인기 있는 양자화 파일에 따라 결정되었습니다.

- F32
- F16
- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q5_K
- Q6_K
- IQ1_S
- IQ1_M
- IQ2_XXS
- IQ2_XS
- IQ2_S
- IQ3_XXS
- IQ3_S
- IQ4_XS
- IQ4_NL

> [!NOTE]
> GGUF 역양자화를 지원하려면 `gguf>=0.10.0` 설치가 필요합니다.

### 지원되는 모델 아키텍처 [[supported-model-architectures]]

현재 지원되는 모델 아키텍처는 Hub에서 매우 인기가 많은 아키텍처들로 제한되어 있습니다:

- LLaMa
- Mistral
- Qwen2
- Qwen2Moe
- Phi3
- Bloom

## 사용 예시 [[example-usage]]

`transformers`에서 `gguf` 파일을 로드하려면 `from_pretrained` 메소드에 `gguf_file` 인수를 지정해야 합니다. 동일한 파일에서 토크나이저와 모델을 로드하는 방법은 다음과 같습니다: 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
```

이제 PyTorch 생태계에서 모델의 양자화되지 않은 전체 버전에 접근할 수 있으며, 다른 여러 도구들과 결합하여 사용할 수 있습니다.

`gguf` 파일로 다시 변환하려면 llama.cpp의 [`convert-hf-to-gguf.py`](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py)를 사용하는 것을 권장합니다.

위의 스크립트를 완료하여 모델을 저장하고 다시 `gguf`로 내보내는 방법은 다음과 같습니다:

```python
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```
