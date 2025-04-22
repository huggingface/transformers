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

You can find all the original MLlama checkpoints under the [meta-llama](https://huggingface.co/meta-llama) collection.

> [!TIP]
> Click on the MLlama models in the right sidebar for more examples of how to apply MLlama to different vision-language tasks like image captioning, visual question answering, and reasoning.

The example below demonstrates how to generate text based on an image with either Pipeline or the model class directly:

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("image-to-text", model="meta-llama/Llama-3.2-11B-Vision")
result = pipe("path/to/your/image.jpg")
print(result[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-11B-Vision", device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

image = Image.open("path/to/your/image.jpg")
inputs = processor(text="Describe this image", images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bit precision:

```python
import torch
from transformers import AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16
)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mllama_architecture.png"/>
</div>

## Notes

- MLlama has special handling for image tokens using `<|image|>` as a placeholder in the text
- The model's input and output embeddings are not tied, requiring special handling for the `lm_head` layer
- When training, mask out the `<|image|>` tokens in labels
- For CUDA index errors during generation, expand the `lm_head`:

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

## MllamaTextModel

[[autodoc]] MllamaTextModel
    - forward


