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

# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is a PyTorch optimization library that makes distributed training memory-efficient and fast. At it's core is the [Zero Redundancy Optimizer (ZeRO)](https://hf.co/papers/1910.02054) which enables training large models at scale. ZeRO works in several stages:

* stage 1, optimizer state partioning across GPUs
* stage 2, gradient partitioning across GPUs
* stage 3, parameteter partitioning across GPUs

In GPU-limited environments, ZeRO also enables offloading optimizer memory and computation from the GPU to the CPU to fit and train really large models on a single GPU. DeepSpeed is integrated with the Transformers [`Trainer`] class for all ZeRO stages and offloading. All you need to do is provide a config file or you can use a provided template. For inference, Transformers support stage 3 and offloading since it allows loading huge models.

This guide will walk you through how to deploy DeepSpeed training, the features you can enable, how to setup the config files for different ZeRO stages, offloading, and using DeepSpeed without the [`Trainer`].

## Installation

DeepSpeed is available to install from PyPI or Transformers (for more detailed installation options, take a look at the DeepSpeed [installation details](https://www.deepspeed.ai/tutorials/advanced-install/) or the GitHub [README](https://github.com/microsoft/deepspeed#installation)).

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>
<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>
</hfoptions>

<Tip>

If you're having difficulties installing DeepSpeed, check the [DeepSpeed CUDA installation](../debugging#deepspeed-cuda-installation) guide.

</Tip>

## Configuration

DeepSpeed works with the [`Trainer`] class by way of a config file containing all the parameters for configuring how you want setup your training run. When you execute your training script, DeepSpeed logs the configuration it received from [`Trainer`] to the console so you can see exactly what configuration was used.

<Tip>

Find a complete list of DeepSpeed configuration options on the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. You can also find more practical examples of various DeepSpeed configuration examples on the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) repository or the main [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository. To quickly find specific examples, you can:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# find examples with the Lamb optimizer
grep -i Lamb $(find . -name '*json')
```

</Tip>

The DeepSpeed configuration file is passed as a path to a JSON file if you're training from the command line interface or as a nested `dict` object if you're using the [`Trainer`] in a notebook setting.

<hfoptions id="pass-config">
<hfoption id="path to file">

```py
TrainingArguments(..., deepspeed="path/to/deepspeed_config.json")
```

</hfoption>
<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(..., deepspeed=ds_config_dict)
trainer = Trainer(model, args, ...)
```

</hfoption>
</hfoptions>

Some of the configuration parameters are shared by [`Trainer`] and DeepSpeed, and it can be difficult to identify errors when there are conflicting definitions. To make it easier, these shared configuration parameters are configured from the [`Trainer`] command line arguments.

### ZeRO configuration

### NVMe configuration

## DeepSpeed features

## Deployment
