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
*이 모델은 2023년 8월 24일에 공개되었으며, 2023년 8월 25일에 Hugging Face Transformers에 추가되었습니다.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        ">
    </div>
</div>

# CodeLlama[[codellama]]

[Code Llama](https://huggingface.co/papers/2308.12950)는 코딩 작업에 특화된 대규모 언어 모델 계열로,  [Llama 2](./llama2)를 기반으로 개발되었습니다. 일반적인 코드, Python 특화, 명령어(지시) 기반 변형 등 다양한 버전으로 제공되며, 모두 7B, 13B, 34B, 70B 매개변수 크기로 사용할 수 있습니다. Code Llama 모델은 코드를 생성하고 설명하며, 코드의 누락된 부분을 채울 수도 있습니다. 이를 인필링(infilling)이라고 합니다. 16K 토큰 길이로 훈련되었지만, 최대 100K 토큰까지 안정적으로 생성하며 긴 컨텍스트도 처리할 수 있습니다.

[Code Llama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933) 컬렉션에서 모든 원본 Code Llama 체크포인트를 찾을 수 있습니다.

> [!TIP]
> 다양한 코딩 작업에 Code Llama를 적용하는 더 많은 예시를 보려면 오른쪽 사이드바의 Code Llama 모델을 클릭하세요.

아래 예시는 [`Pipeline`], [`AutoModel`], 그리고 명령줄에서 코드를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map=0
)

# 기본 코드 생성
result = pipe("# Function to calculate the factorial of a number\ndef factorial(n):", max_new_tokens=256)
print(result[0]['generated_text'])

# 인필링
infill_result = pipe("def remove_non_ascii(s: str) -> str:\n    \"\"\" <FILL_ME>\n    return result", max_new_tokens=200)
print(infill_result[0]['generated_text'])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# 기본 코드 생성
prompt = "# Function to calculate the factorial of a number\ndef factorial(n):"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    **input_ids,
    max_new_tokens=256,
    cache_implementation="static"
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 인필링
infill_prompt = "def remove_non_ascii(s: str) -> str:\n    \"\"\" <FILL_ME>\n    return result"
input_ids = tokenizer(infill_prompt, return_tensors="pt").to(model.device)

filled_output = model.generate(**input_ids, max_new_tokens=200)
filled_text = tokenizer.decode(filled_output[0], skip_special_tokens=True)
print(filled_text)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "# Function to calculate the factorial of a number\ndef factorial(n):" | transformers run --task text-generation --model meta-llama/CodeLlama-7b-hf --device 0
```

</hfoption>
</hfoptions>

양자화는 가중치를 더 낮은 정밀도로 표현하여 대규모 모델의 메모리 부담을 줄입니다. 더 많은 사용 가능한 양자화 백엔드는 [양자화](../quantization/overview) 개요를 참조하세요.

아래 예시는 [bitsandbytes](../quantization/bitsandbytes)를 사용하여 가중치를 4비트로만 양자화합니다.

```py
# bitsandbytes를 설치합니다.
import torch
from transformers import AutoModelForCausalLM, CodeLlamaTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-34b-hf")
model = AutoModelForCausalLM.from_pretrained(
   "meta-llama/CodeLlama-34b-hf",
   torch_dtype=torch.bfloat16,
   device_map="auto",
   quantization_config=bnb_config
)

prompt = "# Write a Python function to check if a string is a palindrome\ndef is_palindrome(s):"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(**input_ids, max_new_tokens=200, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139)를 사용하면 모델이 어떤 토큰에 주의를 기울일 수 있고 기울일 수 없는지를 더 잘 이해할 수 있습니다.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("meta-llama/CodeLlama-7b-hf")
visualizer("""def func(a, b):
  return a + b""")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/codellama-attn-mask.png"/>
</div>

## 참고사항[[notes]]

- 인필링 기능은 7B 및 13B 기반 모델에서만 사용할 수 있으며, Python, Instruct, 34B 또는 70B 모델에서는 사용할 수 없습니다.
- 코드를 채워 넣고 싶은 부분에 `<FILL_ME>` 토큰을 사용하세요. 토크나이저는 이 토큰을 분할하여 [원본 훈련 패턴](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402) 을 따르는 입력 문자열로 변환합니다. 이는 직접 패턴을 준비하는 것보다 더 안정적입니다.
    ```py
    from transformers import LlamaForCausalLM, CodeLlamaTokenizer

    tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
    PROMPT = '''def remove_non_ascii(s: str) -> str:
        """ <FILL_ME>
        return result
    '''
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)

    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print(PROMPT.replace("<FILL_ME>", filling))
    ```
- 추가 훈련이나 미세 조정에는 `bfloat16`을 사용하고 추론에는 `float16`을 사용하세요.
- `BOS` 문자는 접두사나 접미사를 인코딩할 때 인필링 작업에 사용되지 않으며, 각 프롬프트의 맨 앞에서만 사용됩니다.
- 토크나이저는 [SentencePiece](https://github.com/google/sentencepiece)를 기반으로 하는 byte-pair 인코딩 모델입니다. 디코딩 과정에서 첫 번째 토큰이 단어의 시작인 경우(예를 들어 "Banana"), 토크나이저는 문자열에 접두사 공백을 추가하지 않습니다.

## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary
