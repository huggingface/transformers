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

# bitsandbytes [[bitsandbytes]]

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)는 모델을 8비트 및 4비트로 양자화하는 가장 쉬운 방법입니다. 8비트 양자화는 fp16의 이상치와 int8의 비이상치를 곱한 후, 비이상치 값을 fp16으로 다시 변환하고, 이들을 합산하여 fp16으로 가중치를 반환합니다. 이렇게 하면 이상치 값이 모델 성능에 미치는 저하 효과를 줄일 수 있습니다. 4비트 양자화는 모델을 더욱 압축하며, [QLoRA](https://hf.co/papers/2305.14314)와 함께 사용하여 양자화된 대규모 언어 모델을 미세 조정하는 데 흔히 사용됩니다.

bitsandbytes를 사용하려면 다음 라이브러리가 설치되어 있어야 합니다:

<hfoptions id="bnb">
<hfoption id="8-bit">

```bash
pip install transformers accelerate bitsandbytes>0.37.0
```

</hfoption>
<hfoption id="4-bit">

```bash
pip install bitsandbytes>=0.39.0
pip install --upgrade accelerate transformers
```

</hfoption>
</hfoptions>

이제 `BitsAndBytesConfig`를 [`~PreTrainedModel.from_pretrained`] 메소드에 전달하여 모델을 양자화할 수 있습니다. 이는 Accelerate 가져오기를 지원하고 `torch.nn.Linear` 레이어가 포함된 모든 모델에서 작동합니다.

<hfoptions id="bnb">
<hfoption id="8-bit">

모델을 8비트로 양자화하면 메모리 사용량이 절반으로 줄어들며, 대규모 모델의 경우 사용 가능한 GPU를 효율적으로 활용하려면 `device_map="auto"`를 설정하세요. 

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7", 
    quantization_config=quantization_config
)
```

기본적으로 `torch.nn.LayerNorm`과 같은 다른 모듈은 `torch.float16`으로 변환됩니다. 원한다면 `dtype` 매개변수로 이들 모듈의 데이터 유형을 변경할 수 있습니다:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    quantization_config=quantization_config, 
    dtype=torch.float32
)
model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

모델이 8비트로 양자화되면 최신 버전의 Transformers와 bitsandbytes를 사용하지 않는 한 양자화된 가중치를 Hub에 푸시할 수 없습니다. 최신 버전을 사용하는 경우, [`~PreTrainedModel.push_to_hub`] 메소드를 사용하여 8비트 모델을 Hub에 푸시할 수 있습니다. 양자화 config.json 파일이 먼저 푸시되고, 그 다음 양자화된 모델 가중치가 푸시됩니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", 
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

</hfoption>
<hfoption id="4-bit">

모델을 4비트로 양자화하면 메모리 사용량이 4배 줄어들며, 대규모 모델의 경우 사용 가능한 GPU를 효율적으로 활용하려면 `device_map="auto"`를 설정하세요:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    quantization_config=quantization_config
)
```

기본적으로 `torch.nn.LayerNorm`과 같은 다른 모듈은 `torch.float16`으로 변환됩니다. 원한다면 `dtype` 매개변수로 이들 모듈의 데이터 유형을 변경할 수 있습니다:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    quantization_config=quantization_config, 
    dtype=torch.float32
)
model_4bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

`bitsandbytes>=0.41.3`을 사용하는 경우 4비트 모델을 직렬화하고 Hugging Face Hub에 푸시할 수 있습니다. 모델을 4비트 정밀도로 가져온 후 `model.push_to_hub()`를 호출하면 됩니다. 또한 `model.save_pretrained()` 명령어로 로컬에 직렬화된 4비트 모델을 저장할 수도 있습니다.

</hfoption>
</hfoptions>

<Tip warning={true}>

8비트 및 4비트 가중치로 훈련하는 것은 *추가* 매개변수에 대해서만 지원됩니다.

</Tip>

메모리 사용량을 확인하려면 `get_memory_footprint`를 사용하세요:

```py
print(model.get_memory_footprint())
```

양자화된 모델은 [`~PreTrainedModel.from_pretrained`] 메소드를 사용하여 `load_in_8bit` 또는 `load_in_4bit` 매개변수를 지정하지 않고도 가져올 수 있습니다:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

## 8비트 (LLM.int8() 알고리즘)[[8-bit-(llm.int8()-algorithm)]]

<Tip>

8비트 양자화에 대한 자세한 내용을 알고 싶다면 이 [블로그 포스트](https://huggingface.co/blog/hf-bitsandbytes-integration)를 참조하세요!

</Tip>

이 섹션에서는 오프로딩, 이상치 임곗값, 모듈 변환 건너뛰기 및 미세 조정과 같은 8비트 모델의 특정 기능을 살펴봅니다.

### 오프로딩 [[offloading]]

8비트 모델은 CPU와 GPU 간에 가중치를 오프로드하여 매우 큰 모델을 메모리에 장착할 수 있습니다. CPU로 전송된 가중치는 실제로 **float32**로 저장되며 8비트로 변환되지 않습니다. 예를 들어, [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) 모델의 오프로드를 활성화하려면 [`BitsAndBytesConfig`]를 생성하는 것부터 시작하세요:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

CPU에 전달할 `lm_head`를 제외한 모든 것을 GPU에 적재할 수 있도록 사용자 정의 디바이스 맵을 설계합니다:

```py
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

이제 사용자 정의 `device_map`과 `quantization_config`을 사용하여 모델을 가져옵니다:

```py
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### 이상치 임곗값[[outlier-threshold]]

"이상치"는 특정 임곗값을 초과하는 은닉 상태 값을 의미하며, 이러한 값은 fp16으로 계산됩니다. 값은 일반적으로 정규 분포 ([-3.5, 3.5])를 따르지만, 대규모 모델의 경우 이 분포는 매우 다를 수 있습니다 ([-60, 6] 또는 [6, 60]). 8비트 양자화는 ~5 정도의 값에서 잘 작동하지만, 그 이상에서는 상당한 성능 저하가 발생합니다. 좋은 기본 임곗값 값은 6이지만, 더 불안정한 모델 (소형 모델 또는 미세 조정)에는 더 낮은 임곗값이 필요할 수 있습니다.

모델에 가장 적합한 임곗값을 찾으려면 [`BitsAndBytesConfig`]에서 `llm_int8_threshold` 매개변수를 실험해보는 것이 좋습니다:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### 모듈 변환 건너뛰기[[skip-module-conversion]]

[Jukebox](model_doc/jukebox)와 같은 일부 모델은 모든 모듈을 8비트로 양자화할 필요가 없으며, 이는 실제로 불안정성을 유발할 수 있습니다. Jukebox의 경우, [`BitsAndBytesConfig`]의 `llm_int8_skip_modules` 매개변수를 사용하여 여러 `lm_head` 모듈을 건너뛰어야 합니다:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)
```

### 미세 조정[[finetuning]]

[PEFT](https://github.com/huggingface/peft) 라이브러리를 사용하면 [flan-t5-large](https://huggingface.co/google/flan-t5-large) 및 [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)와 같은 대규모 모델을 8비트 양자화로 미세 조정할 수 있습니다. 훈련 시 `device_map` 매개변수를 전달할 필요가 없으며, 모델을 자동으로 GPU에 가져옵니다. 그러나 원하는 경우 `device_map` 매개변수로 장치 맵을 사용자 정의할 수 있습니다 (`device_map="auto"`는 추론에만 사용해야 합니다).

## 4비트 (QLoRA 알고리즘)[[4-bit-(qlora-algorithm)]]

<Tip>

이 [노트북](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf)에서 4비트 양자화를 시도해보고 자세한 내용은 이 [블로그 게시물](https://huggingface.co/blog/4bit-transformers-bitsandbytes)에서 확인하세요.

</Tip>

이 섹션에서는 계산 데이터 유형 변경, Normal Float 4 (NF4) 데이터 유형 사용, 중첩 양자화 사용과 같은 4비트 모델의 특정 기능 일부를 탐구합니다.


### 데이터 유형 계산[[compute-data-type]]

계산 속도를 높이기 위해 [`BitsAndBytesConfig`]에서 `bnb_4bit_compute_dtype` 매개변수를 사용하여 데이터 유형을 float32(기본값)에서 bf16으로 변경할 수 있습니다:

```py
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

### Normal Float 4 (NF4)[[normal-float-4-(nf4)]]

NF4는 [QLoRA](https://hf.co/papers/2305.14314) 논문에서 소개된 4비트 데이터 유형으로, 정규 분포에서 초기화된 가중치에 적합합니다. 4비트 기반 모델을 훈련할 때 NF4를 사용해야 합니다. 이는 [`BitsAndBytesConfig`]에서 `bnb_4bit_quant_type` 매개변수로 설정할 수 있습니다:

```py
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

추론의 경우, `bnb_4bit_quant_type`은 성능에 큰 영향을 미치지 않습니다. 그러나 모델 가중치와 일관성을 유지하기 위해 `bnb_4bit_compute_dtype` 및 `dtype` 값을 사용해야 합니다.

### 중첩 양자화[[nested-quantization]]

중첩 양자화는 추가적인 성능 손실 없이 추가적인 메모리를 절약할 수 있는 기술입니다. 이 기능은 이미 양자화된 가중치의 2차 양자화를 수행하여 매개변수당 추가로 0.4비트를 절약합니다. 예를 들어, 중첩 양자화를 통해 16GB NVIDIA T4 GPU에서 시퀀스 길이 1024, 배치 크기 1, 그레이디언트 누적 4단계를 사용하여 [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) 모델을 미세 조정할 수 있습니다.

```py
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b", quantization_config=double_quant_config)
```

## `bitsandbytes` 모델의 비양자화[[dequantizing-`bitsandbytes`-models]]
양자화된 후에는 모델을 원래의 정밀도로 비양자화할 수 있지만, 이는 모델의 품질이 약간 저하될 수 있습니다. 비양자화된 모델에 맞출 수 있는 충분한 GPU RAM이 있는지 확인하세요.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id, BitsAndBytesConfig(load_in_4bit=True))
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.dequantize()

text = tokenizer("Hello my name is", return_tensors="pt").to(0)

out = model.generate(**text)
print(tokenizer.decode(out[0]))
```
