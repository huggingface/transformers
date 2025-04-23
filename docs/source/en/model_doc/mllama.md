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
    </div>
</div>

# MLlama

[MLlama](https://huggingface.co/papers/2407.21783), or Llama 3.2 Vision, is a multimodal version of [Llama 3](./llama3), available in 11B and 90B parameters, and as an instruction-tuned and base variant. This model integrates a separately trained vision adapter (image encoder, image adapter, and video adapter) to handle visual tasks like image reasoning, captioning, and answering. The instruction-tuned variants are post-trained with supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human helpfulness and safety values.

You can find all the original MLlama checkpoints under the [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) collection.

> [!TIP]
> Click on the MLlama models in the right sidebar for more examples of how to apply MLlama to different vision-language tasks like image captioning, visual question answering, and reasoning.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    device=0,
    torch_dtype=torch.bfloat16
)
messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ],
]
pipeline(text=messages, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ],
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")
output = model.generate(**inputs, max_new_tokens=25)
print(processor.decode(output[0]))
```

</hfoption>
</hfoptions>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mllama_architecture.png"/>
</div>

## Notes

- Use [`MllamaForConditionalGeneration`] for image-text and text inputs.
- Use [`MllamaForCausalLM`] for generation with text only inputs to avoid unnecessarily loading the vision components.
- Samples can contain multiple images and each sample can have different number of images. In these cases, the processor pads the inputs to the maximum number of images across samples and to a maximum number of tiles within each image.
- Use the `<|image|>` token to indicate where an image should be inserted.
- MLlama's input and output embeddings are not tied which means the `lm_head` layer has one less token (the `<|image|>` placeholder token) and fails if you want to calculate loss on image tokens or apply logit processors. For training, make sure to mask the `<|image|>` token in `labels` because the model should not be trained on predicting them.
- For CUDA index errors during generation, expand the `lm_head` by one token:

```python
old_embeddings = model.get_output_embeddings()
num_tokens = model.vocab_size + 1
resized_embeddings = model._get_resized_lm_head(old_embeddings, new_num_tokens=num_tokens, mean_resizing=True)
resized_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
model.set_output_embeddings(resized_embeddings)
```

## MllamaConfig

[[autodoc]] MllamaConfig

## MllamaProcessor

[[autodoc]] MllamaProcessor

## MllamaImageProcessor

[[autodoc]] MllamaImageProcessor

## MllamaForConditionalGeneration

[[autodoc]] MllamaForConditionalGeneration
    - forward

## MllamaForCausalLM

[[autodoc]] MllamaForCausalLM
    - forward

## MllamaVisionModel

[[autodoc]] MllamaVisionModel
    - forward

## MllamaTextModel

[[autodoc]] MllamaTextModel
    - forward





