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

*This model was released on 2025-05-06 and added to Hugging Face Transformers on 2025-10-07.*

# FastVLM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<!-- <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white"> -->
</div>

## Overview

FastVLM is an open-source vision-language model featuring a novel hybrid vision encoder, FastViTHD. Leveraging reparameterizable convolutional layers, scaled input resolution, and a reduced number of visual tokens, FastVLM delivers high accuracy with exceptional efficiency. Its optimized architecture enables deployment even on edge devices, achieving ultra-low TTFT (time to first token) without sacrificing performance.

The model was proposed in [FastVLM: Efficient Vision Encoding for Vision Language Models](https://huggingface.co/papers/2412.13303) by Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel and Hadi Pouransari.

The abstract from the paper is the following:

*Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and  minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, token count, and LLM size, we introduce FastVLM—a model that achieves an optimized trade-off between resolution, latency, and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2× improvement in time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the highest resolution (1152×1152), FastVLM achieves better performance on key benchmarks like SeedBench, MMMU and DocVQA, using the same 0.5B LLM, but with 85× faster TTFT and a vision encoder that is 3.4× smaller.*

This model was contributed by [Kamila](https://github.com/kamila-chay).
The original code can be found [here](https://github.com/apple/ml-fastvlm).

## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

- Note the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results.

### Formatting Prompts with Chat Templates  

Each **checkpoint** is trained with a specific prompt format, depending on the underlying large language model backbone. To ensure correct formatting, use the processor’s `apply_chat_template` method.  

**Important:**  
- You must construct a conversation history — passing a plain string won't work.  
- Each message should be a dictionary with `"role"` and `"content"` keys.  
- The `"content"` should be a list of dictionaries for different modalities like `"text"` and `"image"`.  


Here’s an example of how to structure your input. 
We will use a conversation history of text and image. Each content field has to be a list of dicts, as follows:


```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
            ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in more details."},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your images
print(text_prompt)
>>> "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat’s shown in this image?<|im_end|>\n<|im_start|>assistant\n\nThis image shows a red stop sign.<|im_end|>\n<|im_start|>user\n\nDescribe the image in more details.<|im_end|>\n<|im_start|>assistant\n"
```

🚀 **Bonus:** If you're using `transformers>=4.49.0`, you can also get a vectorized output from `apply_chat_template`. See the **Usage Examples** below for more details on how to use it.

## Usage examples

### Single input inference


```python
import torch
from transformers import AutoProcessor, FastVlmForConditionalGeneration

# Load the model in half-precision
model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.bfloat16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```


### Batched inference

FastVLM also supports batched inference. Here is how you can do it:

```python
import torch
from transformers import AutoProcessor, FastVlmForConditionalGeneration

# Load the model in half-precision
model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")


# Prepare a batch of two prompts
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    [conversation_1, conversation_2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt"
).to(model.device, torch.bfloat16)


# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```


## Note regarding reproducing original implementation

In order to match the logits of the [original implementation](https://github.com/apple/ml-fastvlm), one needs to set the default timm attention implementation to the most basic version(not fused):

```
import os
# at the beginning of your script
os.environ["TIMM_FUSED_ATTN"] = "0"
```

In addition, the layer norm used by Apple doesn't use the standard LayerNorm class form Torch and therefore our logits diverge. To get exactly the same values, one needs to manually change timm/layers/norm.py:

```
class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    _fast_norm: torch.jit.Final[bool]

    def __init__():
        ... # not important

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x
```
Please note, that this is only needed in oder to get the exact same numerical values on the output of the model. It's not necessary to make this change to use FastVLM.

<!-- ### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization, please refer to the [Flash Attention 2 section of performance docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one). -->

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with image-to-text transformers (here using the example of Llava).

<PipelineTag pipeline="image-to-text"/>

- A [Google Colab demo](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing) on how to run Llava on a free-tier Google colab instance leveraging 4-bit inference.
- A [similar notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Inference_with_LLaVa_for_multimodal_generation.ipynb) showcasing batched inference. 🌎

## FastVlmConfig

[[autodoc]] FastVlmConfig

## FastVlmModel

[[autodoc]] FastVlmModel

## FastVlmForConditionalGeneration

[[autodoc]] FastVlmForConditionalGeneration
    - forward