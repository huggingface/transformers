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

# LLM 추론 최적화 [[llm-inference-optimization]]

대규모 언어 모델(LLM)은 채팅 및 코드 완성 모델과 같은 텍스트 생성 응용 프로그램을 한 단계 끌어올리며, 높은 수준의 이해력과 유창함을 보여주는 텍스트를 생성합니다. 그러나 LLM을 강력하게 만드는 요소인 그들의 크기는 동시에 추론 과정에서 도전 과제가 되기도 합니다.

기본적인 추론은 느립니다, 왜냐하면 LLM이 다음 토큰을 생성하기 위해 반복적으로 호출되어야 하기 때문입니다. 생성이 진행됨에 따라 입력 시퀀스가 길어져 처리 시간이 점점 길어집니다. 또한, LLM은 수십억 개의 매개변수를 가지고 있어 모든 가중치를 메모리에 저장하고 처리하는 데 어려움이 있습니다.

이 가이드는 LLM 추론을 가속하기 위해 Transformers에서 사용할 수 있는 최적화 기술을 사용하는 방법을 보여줍니다.

> [!TIP]
> Hugging Face는 LLM을 추론에 최적화하여 배포하고 서비스하는 데 전념하는 라이브러리인 [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference)을 제공합니다. 이 라이브러리는 처리량 증가를 위한 지속적인 배칭과 다중 GPU 추론을 위한 텐서 병렬화와 같은 Transformers에 포함되지 않은 더 많은 최적화 기능을 포함합니다.

## 정적 kv-cache와 torch.compile [[static-kv-cache-and-torchcompile]]

디코딩 중에 LLM은 각 입력 토큰에 대한 key-value(kv) 값을 계산합니다. LLM은 자기회귀(autoregressive)이기 때문에 생성된 출력이 현재 입력의 일부가 되어 매번 동일한 kv 값을 계산합니다. 이는 매번 동일한 kv 값을 다시 계산하기 때문에 효율적이지 않습니다.

이를 최적화하기 위해, 이전 키(key)와 값(value)을 재계산하지 않고 저장하는 kv-cache를 사용할 수 있습니다. 그러나 kv-cache는 각 생성 단계에서 증가하며 동적이기 때문에 PyTorch 코드를 빠르고 최적화된 커널로 통합하는 강력한 최적화 도구인 [torch.compile](./perf_torch_compile)을 사용하는 데 제약이 있습니다.

*정적 kv-cache*는 최댓값을 미리 할당하여 이 문제를 해결하여 torch.compile과 결합할 수 있게 합니다. 이를 통해 최대 4배의 속도 향상이 가능합니다.

> [!WARNING]
현재 [Llama](./model_doc/llama2) 및 몇 가지 다른 모델만 정적 kv-cache와 torch.compile을 지원합니다. 실시간 모델 호환성 목록은 [이 이슈](https://github.com/huggingface/transformers/issues/28981)를 확인하십시오.

이 예제에서는 [Gemma](https://hf.co/google/gemma-2b) 모델을 로드해 보겠습니다.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", device_map="auto"
)
```

두 가지 방법으로 모델을 정적 kv-cache를 사용하도록 설정할 수 있습니다. A100에서 7B 모델의 경우 두 방법 모두 순전파에서 4배 속도 향상이 가능합니다. 속도 향상은 모델 크기(더 큰 모델의 경우 속도 향상이 적음)와 하드웨어에 따라 다를 수 있습니다. [~GenerationMixin.generate] 메서드를 사용하는 경우, 속도는 최대 3배 빨라집니다. 순전파(여전히 4배 속도 향상)는 전체 [~GenerationMixin.generate] 코드의 일부일 뿐입니다.

<hfoptions id="static-kv">
<hfoption id="generation_config">

모델의 `generation_config` 속성에 접근하여 `cache_implementation`을 “static”으로 설정하십시오.

```py
model.generation_config.cache_implementation = "static"
```

정적 kv-cache와 함께 순전파를 컴파일하기 위해 torch.compile을 호출합니다.

```py
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = compiled_model.generate(**input_ids)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

내부적으로, 'generate'는 동일한 캐시 객체를 재사용하려고 시도하여 각 호출에서 다시 컴파일할 필요가 없도록 합니다. 그러나 배치 크기 또는 최대 출력 길이가 호출 간에 증가하면 캐시를 다시 초기화해야 하므로 새로 컴파일해야 합니다.

</hfoption>
<hfoption id="Static Cache">

[`StaticCache`] 객체를 `past_key_values` 인수로 모델의 순전파에 전달하여 정적 kv-cache로 사용할 수 있습니다. 이 전략을 사용하여 현재 토큰과 위치, 이전에 생성된 토큰의 캐시 위치를 기준으로 다음 토큰을 디코딩하는 함수를 작성할 수 있습니다. 또한 [`StaticCache`] 객체를 [`~GenerationMixin.generate`]에 전달하여 동적 캐시처럼 호출 간에 사용할 수 있습니다.

```py
from transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache, logging
from transformers.testing_utils import CaptureLogger
import torch

prompts = [
    "Simply put, the theory of relativity states that ",
    "My favorite all time favorite condiment is ketchup.",
]

NUM_TOKENS_TO_GENERATE = 40
torch_device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="sequential")
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token
```

`StaticCache` 메서드를 사용하여 정적 kv-cache와 torch.compile을 활성화하려면 몇 가지 중요한 작업을 수행해야 합니다:

1. 추론에 모델을 사용하기 전에 [`StaticCache`] 인스턴스를 초기화합니다. 여기서 최대 배치 크기와 시퀀스 길이와 같은 매개변수를 설정할 수 있습니다.

2. 정적 kv-cache와 함께 순전파를 컴파일하기 위해 모델에 torch.compile을 호출합니다.

3. [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) 컨텍스트 관리자에서 `enable_math=True`를 설정하여 네이티브 PyTorch C++ 구현된 스케일된 점곱 어텐션(scaled dot product attention)을 활성화하여 추론 속도를 더욱 높입니다.

```py
batch_size, seq_length = inputs["input_ids"].shape
with torch.no_grad():
    past_key_values = StaticCache(
        config=model.config, max_batch_size=2, max_cache_len=4096, device=torch_device, dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device=torch_device)
    generated_ids = torch.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch_device).to(torch.int)

    logits = model(
        **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
    )[0]
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]

    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    cache_position = torch.tensor([seq_length + 1], device=torch_device)
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.clone(), None, cache_position, past_key_values)
            generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
text
['Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.',
 'My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p']
```

> [!TIP]
> [`StaticCache`] 객체를 새 프롬프트에 재사용하려면 `.reset()` 메서드로 내용을 초기화하십시오.

</hfoption>
</hfoptions>

## 추정 디코딩 [[speculative-decoding]]

> [!TIP]
> 보다 심층적인 설명을 원한다면, [Assisted Generation: a new direction toward low-latency text generation](https://hf.co/blog/assisted-generation) 블로그 게시물을 확인하십시오!

자기 회귀의 또 다른 문제는 각 입력 토큰에 대해 순전파 중에 모델 가중치를 매번 로드해야 한다는 점입니다. 이는 수십억 개의 매개변수를 가진 LLM에는 느리고 번거롭습니다. 추정 디코딩(speculative decoding)은 더 작고 빠른 보조 모델을 사용하여 후보 토큰을 생성하고, 이를 큰 LLM이 단일 순전파에서 검증하여 이 속도 저하를 완화합니다. 검증된 토큰이 정확하다면, LLM은 본래 자체적으로 생성하는 것처럼 토큰을 얻을 수 있습니다. 전방 패스가 동일한 출력을 보장하기 때문에 정확도 저하가 없습니다.

가장 큰 속도 향상을 얻기 위해, 보조 모델은 빠르게 토큰을 생성할 수 있도록 LLM보다 훨씬 작아야 합니다. 보조 모델과 LLM 모델은 토큰을 다시 인코딩하고 디코딩하지 않도록 동일한 토크나이저를 공유해야 합니다.

> [!WARNING]
> 추정 디코딩은 탐욕 검색과 샘플링 디코딩 전략에서만 지원되며, 배치 입력을 지원하지 않습니다.

보조 모델을 로드하고 이를 [`~GenerationMixin.generate`] 메서드에 전달하여 추정 디코딩을 활성화하십시오.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Einstein's theory of relativity states that the speed of light is constant.    "]
```

</hfoption>
<hfoption id="sampling">

추정 샘플링 디코딩(speculative sampling decoding)을 위해, 보조 모델 외에도 [`~GenerationMixin.generate`] 메서드에 `do_sample` 및 `temperature` 매개변수를 추가하십시오.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["Einstein's theory of relativity states that motion in the universe is not a straight line.\n"]
```

</hfoption>
</hfoptions>

### 프롬프트 조회 디코딩 [[prompt-lookup-decoding]]

프롬프트 조회 디코딩은 탐욕 검색과 샘플링과도 호환되는 추정 디코딩의 변형입니다. 프롬프트 조회는 요약과 같은 입력 기반 작업에 특히 잘 작동합니다. 여기서는 프롬프트와 출력 간에 종종 겹치는 단어가 있습니다. 이러한 겹치는 n-그램이 LLM 후보 토큰으로 사용됩니다.

프롬프트 조회 디코딩을 활성화하려면 `prompt_lookup_num_tokens` 매개변수에 겹치는 토큰 수를 지정하십시오. 그런 다음 이 매개변수를 [`~GenerationMixin.generate`] 메서드에 전달할 수 있습니다.

<hfoptions id="pld">
<hfoption id="greedy decoding">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The second law of thermodynamics states that entropy increases with temperature.      ']
```

</hfoption>
<hfoption id="sampling">

샘플링과 함께 프롬프트 조회 디코딩을 사용하려면, [`~GenerationMixin.generate`] 메서드에 `do_sample` 및 `temperature` 매개변수를 추가하십시오.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["The second law of thermodynamics states that energy cannot be created nor destroyed. It's not a"]
```

</hfoption>
</hfoptions>

## 어텐션 최적화 [[attention-optimizations]]

트랜스포머 모델의 알려진 문제는 셀프 어텐션 메커니즘이 입력 토큰 수와 함께 계산 및 메모리가 제곱으로 증가한다는 것입니다. 이 제한은 훨씬 더 긴 시퀀스를 처리하는 LLM에서는 더욱 커집니다. 이를 해결하기 위해 FlashAttention2 또는 PyTorch의 스케일된 점곱 어텐션을 사용해 보십시오. 이들은 더 메모리 효율적인 어텐션 구현으로 추론을 가속화할 수 있습니다.

### FlashAttention-2 [[flashattention-2]]

FlashAttention과 [FlashAttention-2](./perf_infer_gpu_one#flashattention-2)는 어텐션 계산을 더 작은 청크로 나누고 중간 읽기/쓰기 작업을 줄여 추론 속도를 높입니다. FlashAttention-2는 원래 FlashAttention 알고리즘을 개선하여 시퀀스 길이 차원에서도 병렬 처리를 수행하고 하드웨어에서 작업을 더 잘 분할하여 동기화 및 통신 오버헤드를 줄입니다.

FlashAttention-2를 사용하려면 [`~PreTrainedModel.from_pretrained`] 메서드에서 `attn_implementation="flash_attention_2"`를 설정하십시오.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### PyTorch 스케일된 점곱 어텐션(scaled dot product attention) [[pytorch-scaled-dot-product-attention]]

스케일된 점곱 어텐션(SDPA)는 PyTorch 2.0에서 자동으로 활성화되며, FlashAttention, xFormers, PyTorch의 C++ 구현을 지원합니다. SDPA는 CUDA 백엔드를 사용하는 경우 가장 성능이 좋은 어텐션 알고리즘을 선택합니다. 다른 백엔드에서는 SDPA가 PyTorch C++ 구현으로 기본 설정됩니다.

> [!TIP]
> SDPA는 최신 PyTorch 버전이 설치되어 있으면 FlashAttention-2도 지원합니다.

세 가지 어텐션 알고리즘 중 하나를 명시적으로 활성화하거나 비활성화하려면 [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) 컨텍스트 관리자를 사용하십시오. 예를 들어 FlashAttention을 활성화하려면 `enable_flash=True`로 설정하십시오.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.bfloat16,
)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)
```

## 양자화 [[quantization]]

양자화는 LLM 가중치를 더 낮은 정밀도로 저장하여 크기를 줄입니다. 이는 메모리 사용량을 줄이며 GPU 메모리에 제약이 있는 경우 추론을 위해 LLM을 로드하는 것을 더 용이하게 합니다. GPU가 충분하다면, 모델을 양자화할 필요는 없습니다. 추가적인 양자화 및 양자화 해제 단계로 인해 약간의 지연이 발생할 수 있기 때문입니다(AWQ 및 융합 AWQ 모듈 제외).

> [!TIP]
> 다양한 양자화 라이브러리(자세한 내용은 [Quantization](./quantization) 가이드를 참조하십시오)가 있습니다. 여기에는 Quanto, AQLM, AWQ 및 AutoGPTQ가 포함됩니다. 사용 사례에 가장 잘 맞는 라이브러리를 사용해 보십시오. 또한 AutoGPTQ와 bitsandbytes를 비교하는 [Overview of natively supported quantization schemes in 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) 블로그 게시물을 읽어보는 것을 추천합니다.

아래의 모델 메모리 계산기를 사용하여 모델을 로드하는 데 필요한 메모리를 추정하고 비교해 보십시오. 예를 들어 [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)를 로드하는 데 필요한 메모리를 추정해 보십시오.

<iframe
	src="https://hf-accelerate-model-memory-usage.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

Mistral-7B-v0.1을 반정밀도로 로드하려면 [`~transformers.AutoModelForCausalLM.from_pretrained`] 메서드에서 `torch_dtype` 매개변수를 `torch.bfloat16`으로 설정하십시오. 이 경우 13.74GB의 메모리가 필요합니다.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto",
)
```

추론을 위해 양자화된 모델(8비트 또는 4비트)을 로드하려면 [bitsandbytes](https://hf.co/docs/bitsandbytes)를 사용하고 `load_in_4bit` 또는 `load_in_8bit` 매개변수를 `True`로 설정하십시오. 모델을 8비트로 로드하는 데는 6.87GB의 메모리만 필요합니다.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```
