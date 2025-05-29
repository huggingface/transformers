<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed), powered by Zero Redundancy Optimizer (ZeRO), is an optimization library for training and fitting very large models onto a GPU. It is available in several ZeRO stages, where each stage progressively saves more GPU memory by partitioning the optimizer state, gradients, parameters, and enabling offloading to a CPU or NVMe. DeepSpeed is integrated with the [`Trainer`] class and most of the setup is automatically taken care of for you. 

However, if you want to use DeepSpeed without the [`Trainer`], Transformers provides a [`HfDeepSpeedConfig`] class.

<Tip>

Learn more about using DeepSpeed with [`Trainer`] in the [DeepSpeed](../deepspeed) guide.

</Tip>

## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all
