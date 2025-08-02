
<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Gemma 3[[gemma-3]]

[Gemma 3](https://goo.gle/Gemma3Report)는 사전 학습 및 지침 튜닝된 변형이 있는 멀티모달 모델로, 1B, 13B, 27B 파라미터로 제공됩니다. 아키텍처는 이전 Gemma 버전과 대부분 동일합니다. 주요 차이점은 모든 전역 셀프 어텐션 레이어에 대해 5개의 로컬 슬라이딩 윈도우 셀프 어텐션 레이어를 번갈아 사용하고, 128K 토큰의 더 긴 컨텍스트 길이를 지원하며, 고해상도 이미지나 정사각형이 아닌 종횡비의 이미지에서 정보가 사라지는 것을 방지하기 위해 고해상도 이미지를 "패닝 및 스캐닝"할 수 있는 [SigLip](./siglip) 인코더를 사용한다는 것입니다.

지침 튜닝된 변형은 지식 증류 및 강화 학습으로 후속 학습되었습니다.

모든 원본 Gemma 3 체크포인트는 [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) 릴리스에서 찾을 수 있습니다.

> [!TIP]
> Gemma를 다양한 비전 및 언어 작업에 적용하는 방법에 대한 더 많은 예시는 오른쪽 사이드바에서 Gemma 3 모델을 클릭하세요.

아래 예시는 [`Pipeline`] 또는 [`AutoModel`] 클래스를 사용하여 이미지를 기반으로 텍스트를 생성하는 방법을 보여줍니다.

<는 가중치를 더 낮은 정밀도로 표현하여 대규모 모델의 메모리 부담을 줄입니다. 사용 가능한 더 많은 양자화 백엔드는 [양자화](../quantization/overview) 개요를 참조하세요.

아래 예시는 [torchao](../quantization/torchao)를 사용하여 가중치만 int4로 양자화합니다.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-27b-it",
    padding_side="left"
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "당신은 도움이 되는 어시스턴트입니다."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "이 이미지에 무엇이 보이나요?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

모델이 어떤 토큰에 어텐션할 수 있고 어떤 토큰에 어텐션할 수 없는지 더 잘 이해하려면 [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139)를 사용하세요.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/gemma-3-4b-it")
visualizer("<img>이 이미지에 무엇이 보이나요?")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gemma-3-attn-mask.png"/>
</div>

## 노트[[notes]]

- 이미지-텍스트 및 이미지 전용 입력에는 [`Gemma3ForConditionalGeneration`]을 사용하세요.
- Gemma 3는 여러 입력 이미지를 지원하지만, 프로세서에 전달하기 전에 이미지가 올바르게 배치되었는지 확인하세요. 각 배치는 하나 이상의 이미지 리스트여야 합니다.

    ```py
    url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    messages =[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "당신은 도움이 되는 어시스턴트입니다."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url_cow},
                {"type": "image", "url": url_cat},
                {"type": "text", "text": "어떤 이미지가 더 귀여운가요?"},
            ]
        },
    ]
    ```
- 프로세서에 전달되는 텍스트에는 이미지가 삽입되어야 하는 위치마다 `<image>` 토큰이 있어야 합니다.
- 프로세서에는 채팅 메시지를 모델 입력으로 변환하는 자체 [`~ProcessorMixin.apply_chat_template`] 메서드가 있습니다.
- 기본적으로 이미지는 잘리지 않으며 기본 이미지만 모델로 전달됩니다. 고해상도 이미지나 정사각형이 아닌 종횡비의 이미지에서는 비전 인코더가 896x896의 고정 해상도를 사용하기 때문에 아티팩트가 발생할 수 있습니다. 이러한 아티팩트를 방지하고 추론 중 성능을 향상시키려면, `do_pan_and_scan=True`를 설정하여 이미지를 여러 개의 작은 패치로 자르고 기본 이미지 임베딩과 연결하세요. 더 빠른 추론을 위해 패닝 및 스캐닝을 비활성화할 수 있습니다.

    ```diff
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    +   do_pan_and_scan=True,
        ).to("cuda")
    ```
- 텍스트 전용 모드로 학습된 Gemma-3 1B 체크포인트의 경우, 대신 [`AutoModelForCausalLM`]를 사용하세요.

    ```py
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-pt",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

    output = model.generate(**input_ids, cache_implementation="static")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    ```

## Gemma3ImageProcessor

[[autodoc]] Gemma3ImageProcessor

## Gemma3ImageProcessorFast

[[autodoc]] Gemma3ImageProcessorFast

## Gemma3Processor

[[autodoc]] Gemma3Processor

## Gemma3TextConfig

[[autodoc]] Gemma3TextConfig

## Gemma3Config

[[autodoc]] Gemma3Config

## Gemma3TextModel

[[autodoc]] Gemma3TextModel
    - forward

## Gemma3Model

[[autodoc]] Gemma3Model

## Gemma3ForCausalLM

[[autodoc]] Gemma3ForCausalLM
    - forward

## Gemma3ForConditionalGeneration

[[autodoc]] Gemma3ForConditionalGeneration
    - forward

## Gemma3ForSequenceClassification

[[autodoc]] Gemma3ForSequenceClassification
    - forward
