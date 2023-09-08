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

# CodeLlama [[codellama]]

## 개요 [[overview]]

Code Llama 모델은 Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve에 의해 [Code Llama: 코드를 위한 오픈 기반 모델](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)에서 제안되었습니다.

해당 논문의 초록은 다음과 같습니다:

*우리는 Code Llama를 발표합니다. 이는 Llama 2를 기반으로 한 코드 전용 대형 언어 모델로, 오픈 모델 중 최첨단 성능, 내용 채우기 기능, 큰 입력 컨텍스트 지원 및 프로그래밍 작업에 대한 제로샷 명령어 따르기 능력을 제공합니다. 우리는 다양한 응용 프로그램을 포괄하기 위해 기반 모델(Code Llama), Python 전문화 모델(Code Llama - Python), 그리고 명령어 따르기 모델(Code Llama - Instruct)로, 각각 7B, 13B 및 34B의 파라미터를 제공합니다. 모든 모델은 16k 토큰의 시퀀스에서 훈련되었으며 최대 100k 토큰의 입력에서 향상된 성능을 보입니다. 7B와 13B Code Llama 및 Code Llama - Instruct 버전은 주변 컨텐츠를 기반으로 내용을 채우는 것을 지원합니다. Code Llama는 HumanEval과 MBPP에서 각각 최대 53% 및 55%의 점수로 여러 코드 벤치마크에서 오픈 모델 중 최첨단 성능을 달성합니다. 특히, Code Llama - Python 7B는 HumanEval 및 MBPP에서 Llama 2 70B를 능가하며, 우리의 모든 모델은 MultiPL-E에서 다른 모든 공개적으로 사용 가능한 모델을 능가합니다. 우리는 연구 및 상업적 사용 모두를 허용하는 유연한 라이센스로 Code Llama를 공개합니다.*

모든 Code Llama 모델들을 [여기](https://huggingface.co/models?search=code_llama)에서 확인하고, 공식적으로 발표된 것들은 [codellama org](https://huggingface.co/codellama)에서 확인하세요.

<Tip warning={true}>

`Llama2` 패밀리 모델은 Code Llama의 기반이 되었으며, `bfloat16`을 사용하여 훈련되었지만 원래의 추론은 `float16`을 사용합니다. 다양한 정밀도를 살펴봅시다:

* `float32`: 모델 초기화시 PyTorch의 관례는 모델 가중치가 어떤 `dtype`으로 저장되었는지와 상관없이 모델을 `float32`로 로드하는 것입니다. `transformers`도 PyTorch와 일관성을 유지하기 위해 이 관례를 따릅니다. 이것이 기본값으로 선택됩니다. 저장 가중치 타입으로 체크포인트를 로드하도록 `AutoModel` API에 지시하려면 `torch_dtype="auto"`를 지정해야 합니다. 예: `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`.
* `bfloat16`: Code Llama는 이 정밀도로 훈련되었으므로, 추가 훈련이나 미세조정을 위해 이를 사용하는 것이 좋습니다.
* `float16`: 우리는 이 정밀도를 사용하여 추론을 실행하는 것이 좋습니다. 왜냐하면 이것은 보통 `bfloat16`보다 빠르고, 평가 지표는 `bfloat16`에 대해 눈에 띄게 저하되지 않기 때문입니다. `bfloat16`을 사용하여 추론을 실행할 수도 있습니다. 미세조정 후에는 `float16`과 `bfloat16` 모두로 추론 결과를 확인하는 것이 좋습니다.

위에서 언급했듯이, 저장 가중치의 `dtype`은 모델을 초기화할 때 `torch_dtype="auto"`를 사용하지 않는 한 대부분 중요하지 않습니다. 이유는 모델이 먼저 다운로드될 것이기 때문입니다. (온라인 체크포인트의 `dtype`을 사용하여) 그리고 `torch`의 기본 `dtype`으로 형변환됩니다(`torch.float32`가 됩니다). 지정된 `torch_dtype`이 있으면 대신 사용됩니다.

</Tip>

팁:

- 이 모델들은 `Llama2` 모델과 동일한 아키텍처를 가지고 있습니다.
- 내용 채우기 작업은 박스 밖에서 지원됩니다. 입력을 채우고자 하는 곳에서 `tokenizer.fill_token`을 사용해야 합니다.
- 모델 변환 스크립트는 `Llama2` 패밀리와 동일합니다.

다음은 사용 예시입니다.
```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```
스크립트를 실행하려면 전체 모델을 float16 정밀도로 호스팅하기에 충분한 CPU RAM이 필요합니다 (가장 큰 버전들이 여러 체크포인트에 나누어져 있더라도 각각은 모델의 각 가중치의 일부를 포함하므로 RAM에 모두 로드해야 합니다).

- 변환 후, 모델과 토크나이저는 다음을 통해 로드할 수 있습니다:

```python
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```

채워진 부분만 원한다면:
```python
>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128, return_type = 1)
```

내부적으로, 토크나이저는 [원래의 훈련 패턴](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402)을 따르는 형식화된 입력 문자열을 생성하기 위해 [`<FILL_ME>`로 자동 분할](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token)합니다. 이것은 패턴을 직접 준비하는 것보다 더욱 안정적입니다: 토큰 붙이기와 같은 매우 디버깅하기 어려운 함정을 피할 수 있습니다. 이 모델 또는 다른 모델에 필요한 CPU 및 GPU 메모리 양을 확인하려면 [이 계산기](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)를 사용하여 그 값을 결정할 수 있습니다.

- LLaMA 토크나이저는 [sentencepiece](https://github.com/google/sentencepiece)를 기반으로 한 BPE 모델입니다. sentencepiece의 특징 중 하나는 시퀀스를 디코딩할 때 첫 번째 토큰이 단어의 시작이면 (예: "Banana"), 토크나이저는 문자열 앞에 접두사 공간을 추가하지 않는 것입니다.

이 모델은 [ArthurZucker](https://huggingface.co/ArthurZ)에 의해 기여되었습니다. 원저자의 원래 코드는 [여기](https://github.com/facebookresearch/llama)에서 찾을 수 있습니다.


## CodeLlamaTokenizer [[codellamatokenizer]]

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast [[codellamatokenizerfast]]

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary
