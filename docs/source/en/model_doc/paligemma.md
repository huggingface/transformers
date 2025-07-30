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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# PaliGemma

[PaliGemma](https://huggingface.co/papers/2407.07726) is a family of vision-language models (VLMs), combining [SigLIP](./siglip) with the [Gemma](./gemma) 2B model. PaliGemma is available in 3B, 10B, and 28B parameters. The main purpose of PaliGemma is to provide an adaptable base VLM that is easy to transfer to other tasks. The SigLIP vision encoder is a "shape optimized" contrastively pretrained [ViT](./vit) that converts an image into a sequence of tokens and prepended to an optional prompt. The Gemma 2B model is used as the decoder. PaliGemma uses full attention on all image and text tokens to maximize its capacity.

[PaliGemma 2](https://huggingface.co/papers/2412.03555) improves on the first model by using Gemma 2 (2B, 9B, and 27B parameter variants) as the decoder. These are available as **pt** or **mix** variants. The **pt** checkpoints are intended for further fine-tuning and the **mix** checkpoints are ready for use out of the box.

You can find all the original PaliGemma checkpoints under the [PaliGemma](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda), [PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48), and [PaliGemma 2 Mix](https://huggingface.co/collections/google/paligemma-2-mix-67ac6a251aaf3ee73679dcc4) collections.

> [!TIP]
> Click on the PaliGemma models in the right sidebar for more examples of how to apply PaliGemma to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/paligemma2-3b-mix-224",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="What is in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/paligemma2-3b-mix-224",
)

prompt = "What is in this image?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import torch
import requests
from PIL import Image
from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-28b-mix-224",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "google/paligemma2-28b-mix-224",
)

prompt = "What is in this image?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/paligemma2-3b-mix-224")
visualizer("<img> What is in this image?")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/paligemma2-attn-mask.png"/>
</div>

## Notes

- PaliGemma is not a conversational model and works best when fine-tuned for specific downstream tasks such as image captioning, visual question answering (VQA), object detection, and document understanding.
- [`PaliGemmaProcessor`] can prepare images, text, and optional labels for the model. Pass the `suffix` parameter to the processor to create labels for the model during fine-tuning.

    ```py
    prompt = "What is in this image?"
    answer = "a pallas cat"
    inputs = processor(images=image, text=prompt, suffix=answer, return_tensors="pt")
    ```
- PaliGemma can support multiple input images if it is fine-tuned to accept multiple images. For example, the [NLVR2](https://huggingface.co/google/paligemma-3b-ft-nlvr2-448) checkpoint supports multiple images. Pass the images as a list to the processor.

    ```py
    import torch
    import requests
    from PIL import Image
    from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration

    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-nlvr2-448")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-nlvr2-448")

    prompt = "Are these two images the same?"
    cat_image = Image.open(
        requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw
    )
    cow_image = Image.open(
        requests.get(
            "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4=", stream=True
        ).raw
    )

    inputs = processor(images=[[cat_image, cow_image]], text=prompt, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=20, cache_implementation="static")
    print(processor.decode(output[0], skip_special_tokens=True))
    ```

## PaliGemmaConfig

[[autodoc]] PaliGemmaConfig

## PaliGemmaProcessor

[[autodoc]] PaliGemmaProcessor

## PaliGemmaModel

[[autodoc]] PaliGemmaModel

## PaliGemmaForConditionalGeneration

[[autodoc]] PaliGemmaForConditionalGeneration
    - forward
