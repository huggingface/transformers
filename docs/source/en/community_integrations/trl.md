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

# TRL

[TRL](https://huggingface.co/docs/trl/index) is a post-training framework for foundation models. It includes methods like SFT, GRPO, and DPO. Each method has a dedicated trainer that builds on the [`Trainer`] class and scales from a single GPU to multi-node clusters.

```py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

## Transformers integration

TRL extends Transformers APIs and adds method-specific settings.

- TRL trainers build on [`Trainer`]. Method-specific trainers like [`~trl.GRPOTrainer`] add generation, reward scoring, and loss computation. Config classes extend [`TrainingArguments`] with method-specific fields.

- Model loading uses [`AutoConfig.from_pretrained`] and [`PreTrainedModel.from_pretrained`] directly from Transformers.

- Data collators inherit from [`DataCollatorMixin`] and plug into the trainer pipeline.

## Resources

- [TRL](https://huggingface.co/docs/trl/index) docs
- [Fine Tuning with TRL](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/Fine%20tuning%20with%20TRL%20(Oct%2025).pdf) talk