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
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-09-25.*

# Mllama

[MLlama](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) incorporate vision capabilities by adding adapter weights that connect a pre-trained image encoder to the pre-trained language model via cross-attention layers, aligning image and text representations without altering the original language model parameters. Training involved multi-stage pretraining on large-scale noisy and medium-scale high-quality (image, text) pairs, followed by supervised fine-tuning, rejection sampling, and direct preference optimization, including synthetic data augmentation and safety mitigation. Lightweight 1B and 3B models were created using structured pruning and knowledge distillation from larger models, retaining performance while being device-efficient. Post-training also extends context length up to 128K tokens and blends synthetic and real data to optimize capabilities like reasoning, summarization, instruction following, and tool use.

<hfoptions id="usage">
<hfoption id=MllamaForConditionalGeneration">

```py
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", dtype="auto")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ],
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=25)
print(processor.decode(output[0]))
```

</hfoption>
</hfoptions>


## MllamaConfig

[[autodoc]] MllamaConfig

## MllamaProcessor

[[autodoc]] MllamaProcessor

## MllamaImageProcessor

[[autodoc]] MllamaImageProcessor

## MllamaImageProcessorFast

[[autodoc]] MllamaImageProcessorFast

## MllamaForConditionalGeneration

[[autodoc]] MllamaForConditionalGeneration
    - forward

## MllamaModel

[[autodoc]] MllamaModel
    - forward

## MllamaForCausalLM

[[autodoc]] MllamaForCausalLM
    - forward

## MllamaTextModel

[[autodoc]] MllamaTextModel
    - forward

## MllamaForCausalLM

[[autodoc]] MllamaForCausalLM
    - forward

## MllamaVisionModel

[[autodoc]] MllamaVisionModel
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="meta-llama/Llama-3.2-11B-Vision", dtype="auto")
pipeline("What does this image show?", "path/to/image.png")
```
