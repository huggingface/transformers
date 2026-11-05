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

Transformers runs on CPUs, NVIDIA GPUs, AMD GPUs, Apple Silicon, Google TPUs, AWS Trainium instances, and more through their PyTorch backends.

- For training, [`TrainingArguments`] detects the backend automatically.
- For inference, automatic detection doesn't cover every backend, so pass the device string to choose one explicitly. This page lists the device string and package for each backend, what [`TrainingArguments`] configures it for, and how Transformers decides which device to use.

## Find your hardware

Each backend has a device string that you pass to `device_map` or `device`. CPU and Apple Silicon work with the standard `torch` package. The other backends require a PyTorch extension package that registers the backend with PyTorch.

| Hardware                         | Device string | Package               |
|----------------------------------|---------------|-----------------------|
| [CPU](./cpu)                     | `cpu`         | `torch`               |
| [NVIDIA GPU](./nvidia_gpu)       | `cuda`        | `torch`               |
| [AMD GPU](./amd_gpu)             | `cuda`        | `torch` (ROCm build)  |
| [AWS Trainium](./aws_trainium)   | `neuron`      | `torch-neuronx`       |
| [Google TPU](./google_tpu)       | `tpu`         | `torch-tpu`           |
| [Intel GPU](./intel_gpu)         | `xpu`         | `torch`               |
| [Apple Silicon](./apple_silicon) | `mps`         | `torch`               |
| [Intel Gaudi](./intel_gaudi)     | `hpu`         | `habana-torch-plugin` |
| Huawei Ascend NPU                | `npu`         | `torch-npu`           |
| Cambricon MLU                    | `mlu`         | `torch-mlu`           |
| Moore Threads GPU                | `musa`        | `torch-musa`          |

> [!TIP]
> If a hardware backend is missing from this table and you believe Transformers supports it, please [open an issue](https://github.com/huggingface/transformers/issues/new/choose).

## How Transformers picks a device

Transformers picks a device differently for training and inference.

### Training

[`TrainingArguments`] detects the backend automatically through [Accelerate](https://hf.co/docs/accelerate/index). In a distributed run, each process uses the device Accelerate assigns to it. In a single process, [`TrainingArguments`] picks the first available backend, and falls back to CPU if none is available. Set `use_cpu=True` to force training on CPU.

### Inference

[`~PreTrainedModel.from_pretrained`] loads a model on CPU by default.

Set `device_map="auto"` to place it on the available accelerators. Automatic placement relies on Accelerate to detect CUDA, Intel XPU, Intel Gaudi (HPU), and Apple Silicon (MPS), among others, but it doesn't cover every backend. For other backends, such as AWS Trainium and Google TPU, pass the device string to `device_map` explicitly.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="neuron")
```
