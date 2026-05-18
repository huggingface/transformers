<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-10-31 and added to Hugging Face Transformers on 2026-03-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# PI0

[PI0](https://huggingface.co/papers/2410.24164) is a vision-language-action model for robotics manipulation. It jointly processes visual observations and language instructions to generate robot actions.

The abstract from the paper is as follows:
*Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems, as well as to address some of the deepest questions in artificial intelligence. However, bringing robot learning to the level of generality required for effective real-world systems faces major obstacles in terms of data, generalization, and robustness. In this paper, we discuss how generalist robot policies (i.e., robot foundation models) can address these challenges, and how we can design effective generalist robot policies for complex and highly dexterous tasks. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We then discuss how this model can be trained on a large and diverse dataset from multiple dexterous robot platforms, including single-arm robots, dual-arm robots, and mobile manipulators. We evaluate our model in terms of its ability to perform tasks in zero shot after pre-training, follow language instructions from people and from a high-level VLM policy, and its ability to acquire new skills via fine-tuning. Our results cover a wide variety of tasks, such as laundry folding, table cleaning, and assembling boxes.*


This model was contributed by [Molbap](https://huggingface.co/Molbap) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/Physical-Intelligence/openpi).

You can find all the checkpoints under the [PI0](https://huggingface.co/collections/lerobot/pi0) collection.

## Usage examples

```python
import torch

from transformers import PI0ForConditionalGeneration, PI0Processor
from transformers.image_utils import load_image


model = PI0ForConditionalGeneration.from_pretrained(
    "lerobot/pi0_base",
    device_map="auto",
    attn_implementation="sdpa"
)
processor = PI0Processor.from_pretrained("google/paligemma2-3b-mix-224")

prompt = "Pick up the object"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/vla_pi0.jpg")
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

state = torch.randn(1, 32) # change with actual robot state
actions = model.sample_actions(**inputs, state=state, num_steps=3)
print(actions)
```

## PI0Config

[[autodoc]] PI0Config

## PI0Processor

[[autodoc]] PI0Processor
    - __call__

## PI0ImageProcessor

[[autodoc]] PI0ImageProcessor
    - preprocess

## PI0Model

[[autodoc]] PI0Model
    - forward
    - embed_prefix

## PI0ForConditionalGeneration

[[autodoc]] PI0ForConditionalGeneration
    - forward
    - sample_actions
