<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 생성 기능 [[generation-features]]

[`~GenerationMixin.generate`] API는 이를 기반으로 애플리케이션을 만들 때 유용한 몇 가지 기능을 지원합니다.

이 가이드에서는 이러한 기능을 사용하는 방법을 보여드립니다.

## 스트리밍 [[streaming]]

스트리밍은 텍스트가 생성되는 즉시 반환하기 시작하므로, 생성된 응답 전체가 한꺼번에 나올 때까지 기다릴 필요가 없습니다. 이는 체감 지연 시간을 줄이고, 사용자가 생성 과정을 지켜볼 수 있게 하기 때문에, 사용자가 직접 사용하는 애플리케이션에서 중요합니다.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual-dark_360.gif"/>
</div>

> [!TIP]
> 스트리밍에 대한 자세한 내용은 [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming) 문서에서 확인하세요.

토크나이저로 [`TextStreamer`]의 인스턴스를 생성합니다. [`~GenerationMixin.generate`]의 `streamer` 매개변수에 [`TextStreamer`]를 전달하면 출력을 한 단어씩 스트리밍할 수 있습니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
```

`streamer` 매개변수는 [`~TextStreamer.put`]과 [`~TextStreamer.end`] 메서드를 가진 모든 클래스와 호환됩니다. [`~TextStreamer.put`]은 새 토큰을 전달하고, [`~TextStreamer.end`]는 생성이 끝났음을 알립니다. 이 두 메서드만 포함하면 직접 스트리머 클래스를 만들 수 있으며, Transformers의 기본 스트리머 클래스를 사용할 수도 있습니다.

## 워터마킹 [[watermarking]]

워터마킹은 텍스트가 생성된 것인지 감지하는 데 유용합니다. Transformers의 [워터마킹 전략](https://hf.co/papers/2306.04634)은 토큰의 일부를 무작위로 초록색으로 "칠합니다". 초록색 토큰은 로짓에 작은 편향이 더해지므로 생성될 확률이 더 높아집니다. 생성된 텍스트에 포함된 초록색 토큰의 비율을 사람이 쓴 텍스트에서 일반적으로 나타나는 초록색 토큰의 양과 비교하여 생성된 텍스트를 감지할 수 있습니다.

워터마킹은 Transformers의 모든 생성형 모델에서 지원되며, 워터마크가 적용된 텍스트를 감지하는 데 별도의 분류 모델이 필요하지 않습니다.

로짓에 더할 편향 값과 워터마킹 알고리즘을 지정하여 [`WatermarkingConfig`]를 만듭니다. 아래 예시에서는 초록색 토큰 선택이 현재 토큰에만 의존하는 `"selfhash"` 알고리즘을 사용합니다. [`WatermarkingConfig`]를 [`~GenerationMixin.generate`]에 전달합니다.

> [!TIP]
> [`WatermarkDetector`] 클래스는 생성된 텍스트에서 초록색 토큰의 비율을 감지합니다. 따라서 프롬프트 텍스트가 생성된 텍스트보다 훨씬 길다면 프롬프트를 제거하는 것이 좋습니다. 패딩도 [`WatermarkDetector`]에 영향을 줄 수 있습니다.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

inputs = tokenizer(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
out = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
```

[`WatermarkDetector`]의 인스턴스를 생성하고 모델 출력을 전달하여 텍스트가 기계로 생성된 것인지 감지합니다. [`WatermarkDetector`]는 생성 시 사용한 것과 동일한 [`WatermarkingConfig`]를 사용해야 합니다.

```py
detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
detection_out = detector(out, return_dict=True)
detection_out.prediction
array([True, True])
```
