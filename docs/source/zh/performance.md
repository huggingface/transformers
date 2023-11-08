<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 性能与可扩展性

训练大型transformer模型并将其部署到生产环境会面临各种挑战。
在训练过程中，模型可能需要比可用的GPU内存更多的资源，或者表现出较慢的训练速度。在部署阶段，模型可能在生产环境中难以处理所需的吞吐量。

本文档旨在帮助您克服这些挑战，并找到适合您使用场景的最佳设置。教程分为训练和推理部分，因为每个部分都有不同的挑战和解决方案。在每个部分中，您将找到针对不同硬件配置的单独指南，例如单GPU与多GPU用于训练或CPU与GPU用于推理。

将此文档作为您的起点，进一步导航到与您的情况匹配的方法。

## 训练

高效训练大型transformer模型需要使用加速器硬件，如GPU或TPU。最常见的情况是您只有一个GPU。您应用于单个GPU上提高训练效率的方法可以扩展到其他设置，如多个GPU。然而，也有一些特定于多GPU或CPU训练的技术。我们在单独的部分中介绍它们。

* [在单个GPU上进行高效训练的方法和工具](perf_train_gpu_one)：从这里开始学习常见的方法，可以帮助优化GPU内存利用率、加快训练速度或两者兼备。
* [多GPU训练部分](perf_train_gpu_many)：探索此部分以了解适用于多GPU设置的进一步优化方法，例如数据并行、张量并行和流水线并行。
* [CPU训练部分](perf_train_cpu)：了解在CPU上的混合精度训练。
* [在多个CPU上进行高效训练](perf_train_cpu_many)：了解分布式CPU训练。
* [使用TensorFlow在TPU上进行训练](perf_train_tpu_tf)：如果您对TPU还不熟悉，请参考此部分，了解有关在TPU上进行训练和使用XLA的建议性介绍。
* [自定义硬件进行训练](perf_hardware)：在构建自己的深度学习机器时查找技巧和窍门。
* [使用Trainer API进行超参数搜索](hpo_train)


## 推理

在生产环境中对大型模型进行高效推理可能与训练它们一样具有挑战性。在接下来的部分中，我们将详细介绍如何在CPU和单/多GPU设置上进行推理的步骤。

* [在单个CPU上进行推理](perf_infer_cpu)
* [在单个GPU上进行推理](perf_infer_gpu_one)
* [多GPU推理](perf_infer_gpu_one)
* [TensorFlow模型的XLA集成](tf_xla)

## 训练和推理

在这里，您将找到适用于训练模型或使用它进行推理的技巧、窍门和技巧。

* [实例化大型模型](big_models)
* [解决性能问题](debugging)

## 贡献

这份文档还远远没有完成，还有很多需要添加的内容，所以如果你有补充或更正的内容，请毫不犹豫地提交一个PR（Pull Request），或者如果你不确定，可以创建一个Issue，我们可以在那里讨论细节。

在做出贡献时，如果A比B更好，请尽量包含可重复的基准测试和(或)该信息来源的链接（除非它直接来自您）。
