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

# Hardware

Transformers runs on CPUs, NVIDIA GPUs, Apple Silicon, Google TPUs, AWS Trainium instances, and more through their PyTorch backends. For training, [`TrainingArguments`] detects the backend automatically. For inference, automatic detection doesn't cover every backend, so pass the device string to choose one explicitly. This page lists the device string and package for each backend, what [`TrainingArguments`] configures it for, and how Transformers decides which device to use.

## Find your hardware

Each backend has a device string that you pass to `device_map` or `device`. CPU and Apple Silicon work with the standard `torch` package. The other backends require a PyTorch extension package that registers the backend with PyTorch.

<!-- TODO: Add proper links to each backend's installation instructions. -->

| Hardware          | Device string | Package               | Guide                                                                                                                                        |
|-------------------|---------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| CPU               | `cpu`         | `torch`               | [CPU training](./perf_train_cpu), including mixed precision and distributed training.                                                        |
| NVIDIA GPU        | `cuda`        | `torch`               | [Installation](./installation#python).                                                                                                       |
| AWS Trainium      | `neuron`      | `torch-neuronx`       | TODO                                                                                                                                         |
| Google TPU        | `tpu`         | `torch-tpu`           | TODO                                                                                                                                         |
| Intel GPU         | `xpu`         | `torch`               | [Installation](./installation#python) and [accelerator selection](#select-and-order-accelerators), including device visibility and ordering. |
| Apple Silicon     | `mps`         | `torch`               | [MPS configuration](./perf_train_special), including model loading, mixed precision, and graph cache management.                             |
| Intel Gaudi       | `hpu`         | `habana-torch-plugin` | [HPU configuration](./perf_train_gaudi), including execution modes, mixed precision, compilation, and distributed training.                  |
| Huawei Ascend NPU | `npu`         | `torch-npu`           | TODO                                                                                                                                         |
| Cambricon MLU     | `mlu`         | `torch-mlu`           | TODO                                                                                                                                         |
| Moore Threads GPU | `musa`        | `torch-musa`          | TODO                                                                                                                                         |


> [!TIP]
> If a hardware backend is missing from this table and you believe Transformers supports it, please [open an issue](https://github.com/huggingface/transformers/issues/new/choose).

## How Transformers picks a device

Transformers picks a device differently for training and inference.

### Training

[`TrainingArguments`] detects the backend automatically through [Accelerate](https://hf.co/docs/accelerate/index). In a distributed run, each process uses the device Accelerate assigns to it. In a single process, [`TrainingArguments`] picks the first available backend, and falls back to CPU if none is available. Set `use_cpu=True` to force training on CPU.

### Inference

[`~PreTrainedModel.from_pretrained`] loads a model on CPU by default. Set `device_map="auto"` to place it on the available accelerators. Automatic placement detects CUDA, Intel XPU, Intel Gaudi (HPU), and Apple Silicon (MPS), among others, but it doesn't cover every backend. For other backends, such as AWS Neuron, pass the device string to `device_map` explicitly.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="neuron")
```

### Select and order accelerators

You can control which accelerators (CUDA, XPU, MPS, HPU, etc.) PyTorch sees and in what order. Prioritize faster devices or limit a run to a subset of available hardware. It works with both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html), and doesn't require Accelerate or the [DeepSpeed integration](./main_classes/deepspeed).

Use the hardware-specific environment variable to select accelerators and set their order. Set it on the command line per run, or add it to `~/.bashrc` or another startup config file.

> [!WARNING]
> Avoid exporting environment variables because if you forget how an environment variable was set up, you may silently train on the wrong accelerators. Set the environment variable on the same command line as the training run.

For example, to select accelerators 0 and 2 out of four:

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```cli
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):

```cli
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

To run without any GPUs:

```cli
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

Control the order of CUDA devices with `CUDA_DEVICE_ORDER`.

- Order by PCIe bus ID (matches `nvidia-smi`):

    ```cli
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```

- Order by compute capability (fastest first):

    ```cli
    export CUDA_DEVICE_ORDER=FASTEST_FIRST
    ```

</hfoption>
<hfoption id="Intel XPU">

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

</hfoption>
</hfoptions>

For techniques that can be used across hardware, see the [Optimization](./optimization_overview) overview. Support for each technique depends on the device and backend.
