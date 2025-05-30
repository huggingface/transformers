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

# Aria

[Aria](https://huggingface.co/papers/2410.05993) is Rhymes AI’s from-scratch, open-source multimodal model—natively integrating text, images, video and code without forcing visuals through text-only bottlenecks.

Its Mixture-of-Experts design activates just 3.9 billion parameters per visual token (3.5 billion per text token), cutting compute while preserving each modality’s unique spatial and hierarchical structure. This efficient, native architecture outperforms other models of similar size on multimodal, language and coding benchmarks.

You can find all the original Aria checkpoints under the [Aria](https://huggingface.co/rhymes-ai/Aria) collection.

> [!TIP]
> Click on the Aria models in the right sidebar for more examples of how to apply Aria to different multimodal tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
from PIL import Image
import requests

# Initialize pipeline for multimodal tasks
aria_pipe = pipeline("image-to-text", model="rhymes-ai/Aria")

# Process image and text query
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

response = aria_pipe(
    image,
    prompt="what is the image?",
    max_new_tokens=500,
    temperature=0.9
)

print(response[0]['generated_text'])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

# Load model with precision control
model = AutoModelForCausalLM.from_pretrained(
    "rhymes-ai/Aria",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("rhymes-ai/Aria", trust_remote_code=True)

# Prepare multimodal input
image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", stream=True).raw)
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"text": "Describe this image", "type": "text"}
    ]
}]

# Process and generate
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs = {k: v.to(model.device).to(model.dtype) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.9,
        pad_token_id=processor.tokenizer.eos_token_id
    )

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

</hfoption>
</hfoptions>

## Quantization

The original Aria model uses a custom grouped-gemm implementation, which is not compatible with standard quantization tools. To make quantization feasible, the community provides the [rhymes-ai/Aria-sequential_mlp](https://huggingface.co/rhymes-ai/Aria-sequential_mlp) fork, which replaces grouped-gemm with standard `nn.Linear` layers, enabling successful 4-bit and 8-bit quantization using libraries like bitsandbytes and TorchAO. We suggest you to use quantization-friendly fork or other pre-quantized weights(listed below) to run Aria efficiently on GPUs with limited memory, as the original model architecture does not natively support standard quantization workflows.

**Pre-quantized versions:**

- [FP8 dynamic quantization (llm-compressor)](https://huggingface.co/leon-se/Aria-sequential_mlp-FP8-dynamic) – ~30GB VRAM.
- [4-bit NF4 quantization (bitsandbytes)](https://huggingface.co/leon-se/Aria-sequential_mlp-bnb_nf4) – ~15.5GB VRAM.
- [int8 quantization (TorchAO, official)](https://huggingface.co/rhymes-ai/Aria-torchao-int8wo) – ~30GB VRAM.

## Notes

- Context window: 64,000 tokens for long multimodal documents.


## AriaImageProcessor

[[autodoc]] AriaImageProcessor

## AriaProcessor

[[autodoc]] AriaProcessor

## AriaTextConfig

[[autodoc]] AriaTextConfig

## AriaConfig

[[autodoc]] AriaConfig

## AriaTextModel

[[autodoc]] AriaTextModel

## AriaModel

[[autodoc]] AriaModel

## AriaTextForCausalLM

[[autodoc]] AriaTextForCausalLM

## AriaForConditionalGeneration

[[autodoc]] AriaForConditionalGeneration
    - forward
