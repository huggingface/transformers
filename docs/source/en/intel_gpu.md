<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Intel GPU

[`TrainingArguments`], [`Trainer`], and [`Pipeline`] detect Intel GPUs and set the backend to `xpu` automatically.

## Select and order XPUs

You can control which XPUs PyTorch sees and in what order. Prioritize faster XPUs or limit a run to a subset of available XPUs. It works with both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html), and doesn't require Accelerate or the [DeepSpeed integration](./main_classes/deepspeed).

Use the `ZE_AFFINITY_MASK` environment variable to select XPUs and set their order. Set it on the command line per run, or add it to `~/.bashrc` or another startup config file.

> [!WARNING]
> Avoid exporting environment variables because if you forget how an environment variable was set up, you may silently train on the wrong XPUs. Set the environment variable on the same command line as the training run.

For example, to select XPUs 0 and 2 out of four:

```cli
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

PyTorch sees only XPUs 0 and 2, which are mapped to `xpu:0` and `xpu:1`. To reverse the order (use XPU 2 as `xpu:0` and XPU 0 as `xpu:1`):

```cli
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```

Control the order of Intel XPUs with:

```cli
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

For more on device enumeration and sorting on Intel XPU, see the [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) documentation.
