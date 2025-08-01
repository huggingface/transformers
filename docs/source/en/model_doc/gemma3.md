
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

# Gemma 3

[Gemma 3](https://goo.gle/Gemma3Report) is a multimodal model with pretrained and instruction-tuned variants, available in 1B, 13B, and 27B parameters. The architecture is mostly the same as the previous Gemma versions. The key differences are alternating 5 local sliding window self-attention layers for every global self-attention layer, support for a longer context length of 128K tokens, and a [SigLip](./siglip) encoder that can "pan & scan" high-resolution images to prevent information from disappearing in high resolution images or images with non-square aspect ratios.

The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning.

You can find all the original Gemma 3 checkpoints under the [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) release.

> [!TIP]
> Click on the Gemma 3 models in the right sidebar for more examples of how to apply Gemma to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-3-4b-pt",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="left"
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
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

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model google/gemma-3-1b-pt --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-27b-it",
    dtype=torch.bfloat16,
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
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
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

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/gemma-3-4b-it")
visualizer("<img>What is shown in this image?")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gemma-3-attn-mask.png"/>
</div>

## Notes

- Use [`Gemma3ForConditionalGeneration`] for image-and-text and image-only inputs.
- Gemma 3 supports multiple input images, but make sure the images are correctly batched before passing them to the processor. Each batch should be a list of one or more images.

    ```py
    url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    messages =[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url_cow},
                {"type": "image", "url": url_cat},
                {"type": "text", "text": "Which image is cuter?"},
            ]
        },
    ]
    ```
- Text passed to the processor should have a `<start_of_image>` token wherever an image should be inserted.
- The processor has its own [`~ProcessorMixin.apply_chat_template`] method to convert chat messages to model inputs.
- By default, images aren't cropped and only the base image is forwarded to the model. In high resolution images or images with non-square aspect ratios, artifacts can result because the vision encoder uses a fixed resolution of 896x896. To prevent these artifacts and improve performance during inference, set `do_pan_and_scan=True` to crop the image into multiple smaller patches and concatenate them with the base image embedding. You can disable pan and scan for faster inference.

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
- For Gemma-3 1B checkpoint trained in text-only mode, use [`AutoModelForCausalLM`] instead.

    ```py
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-pt",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
        dtype=torch.bfloat16,
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
