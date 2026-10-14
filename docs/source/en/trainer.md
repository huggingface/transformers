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

# Trainer

[`Trainer`] is a complete training and evaluation loop for Transformers models. You only need a model and dataset to get started.

<Youtube id="nvBXf7s7vTI"/>

Underneath, [`Trainer`] handles batching, shuffling, and padding your dataset into tensors. The training loop runs the forward pass, calculates loss, backpropagates gradients, and updates weights. Configure the training run with [`TrainingArguments`] to customize everything from batch size and training duration to distributed strategies, compilation, and more.

## Next steps

- Start with the [fine-tuning](./training) tutorial for an introduction to training a large language model with [`Trainer`].
- Check the [Subclassing Trainer methods](./trainer_customize) guide for examples of how to subclass [`Trainer`] methods.
- See the [Data collators](./data_collators) guide to learn how to create a data collator for custom batch assembly.
- See the [Callbacks](./trainer_callbacks) guide to learn how to hook into training events.
