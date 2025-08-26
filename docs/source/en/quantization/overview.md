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

# Overview

Quantization lowers the memory requirements of loading and using a model by storing the weights in a lower precision while trying to preserve as much accuracy as possible. Weights are typically stored in full-precision (fp32) floating point representations, but half-precision (fp16 or bf16) are increasingly popular data types given the large size of models today. Some quantization methods can reduce the precision even further to integer representations, like int8 or int4.

Transformers supports many quantization methods, each with their pros and cons, so you can pick the best one for your specific use case. Some methods require calibration for greater accuracy and extreme compression (1-2 bits), while other methods work out of the box with on-the-fly quantization.

Use the Space below to help you pick a quantization method depending on your hardware and number of bits to quantize to.

| Quantization Method                       | On the fly quantization | CPU             | CUDA GPU | ROCm GPU  | Metal (Apple Silicon)              | Intel GPU       | Torch compile() | Bits         | PEFT Fine Tuning | Serializable with 🤗Transformers | 🤗Transformers Support  | Link to library                             |
|-------------------------------------------|----------------------|-----------------|----------|-----------|------------------------------------|-----------------|-----------------|--------------|------------------|-----------------------------|-------------------------|---------------------------------------------|
| [AQLM](./aqlm)                            | 🔴                   | 🟢              |     🟢     | 🔴        | 🔴                                 | 🟢              | 🟢              | 1/2          | 🟢               | 🟢                          | 🟢                      | https://github.com/Vahe1994/AQLM            |
| [AutoRound](./auto_round)                 | 🔴                   | 🟢               | 🟢          |   🔴        |   🔴                                |   🟢              |   🔴               | 2/3/4/8      |    🔴              |       🟢                      |    🟢                       |      https://github.com/intel/auto-round                                       |
| [AWQ](./awq)                              | 🔴                   | 🟢              | 🟢        | 🟢        | 🔴                                 | 🟢              | ?               | 4            | 🟢               | 🟢                          | 🟢                      | https://github.com/casper-hansen/AutoAWQ    |
| [bitsandbytes](./bitsandbytes)            | 🟢                   | 🟡 |     🟢     | 🟡 | 🔴                    | 🟡 | 🟢 | 4/8          | 🟢               | 🟢                          | 🟢                      | https://github.com/bitsandbytes-foundation/bitsandbytes |
| [compressed-tensors](./compressed_tensors) | 🔴                   | 🟢              |     🟢     | 🟢        | 🔴                                 | 🔴              | 🔴              | 1/8          | 🟢               | 🟢                          | 🟢                      | https://github.com/neuralmagic/compressed-tensors |
| [EETQ](./eetq)                            | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | ?               | 8            | 🟢               | 🟢                          | 🟢                      | https://github.com/NetEase-FuXi/EETQ        |
| [FP-Quant](./fp_quant)                          | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 4           | 🔴               | 🟢                          | 🟢                      | https://github.com/IST-DASLab/FP-Quant      |
| [GGUF / GGML (llama.cpp)](../gguf)        | 🟢                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🟢              | 🔴              | 1/8          | 🔴               | [See Notes](../gguf)     | [See Notes](../gguf) | https://github.com/ggerganov/llama.cpp      |
| [GPTQModel](./gptq)                       | 🔴                   | 🟢 | 🟢        | 🟢        | 🟢                                 | 🟢 | 🔴              | 2/3/4/8      | 🟢               | 🟢                          | 🟢                      | https://github.com/ModelCloud/GPTQModel        |
| [AutoGPTQ](./gptq)                        | 🔴                   | 🔴              | 🟢        | 🟢        | 🔴                                 | 🔴              | 🔴              | 2/3/4/8      | 🟢               | 🟢                          | 🟢                      | https://github.com/AutoGPTQ/AutoGPTQ        |
| [HIGGS](./higgs)                          | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 2/4          | 🔴               | 🟢                          | 🟢                      | https://github.com/HanGuo97/flute           |       
| [HQQ](./hqq)                              | 🟢                   | 🟢              | 🟢        | 🔴        | 🔴                                 | 🟢              | 🟢              | 1/8          | 🟢               | 🔴                          | 🟢                      | https://github.com/mobiusml/hqq/            |
| [optimum-quanto](./quanto)                | 🟢                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🟢              | 🟢              | 2/4/8        | 🔴               | 🔴                          | 🟢                      | https://github.com/huggingface/optimum-quanto       |
| [FBGEMM_FP8](./fbgemm_fp8)                | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🔴              | 8            | 🔴               | 🟢                          | 🟢                      | https://github.com/pytorch/FBGEMM       |
| [torchao](./torchao)                      | 🟢                   | 🟢               | 🟢        | 🔴        | 🟡 | 🟢              |                 | 4/8          |                  | 🟢🔴                        | 🟢                      | https://github.com/pytorch/ao       |
| [VPTQ](./vptq)                            | 🔴                   | 🔴              |     🟢     | 🟡        | 🔴                                 | 🔴              | 🟢              | 1/8          | 🔴               | 🟢                          | 🟢                      | https://github.com/microsoft/VPTQ            |
| [FINEGRAINED_FP8](./finegrained_fp8)      | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🟢              | 🔴              | 8            | 🔴               | 🟢                          | 🟢                      |        |
| [SpQR](./spqr)                            | 🔴                     |  🔴   | 🟢        | 🔴              |    🔴    | 🔴         |         🟢              | 3            |              🔴                     | 🟢           | 🟢                      | https://github.com/Vahe1994/SpQR/       |
| [Quark](./quark)                          | 🔴                     | 🟢 | 🟢      | 🟢      | 🟢                   | 🟢       | ?               | 2/4/6/8/9/16 | 🔴                | 🔴                               | 🟢                       | https://quark.docs.amd.com/latest/                      |

## Resources

If you are new to quantization, we recommend checking out these beginner-friendly quantization courses in collaboration with DeepLearning.AI.

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)

## User-Friendly Quantization Tools

If you are looking for a user-friendly quantization experience, you can use the following community spaces and notebooks: 

* [Bitsandbytes Space](https://huggingface.co/spaces/bnb-community/bnb-my-repo)
* [GGUF Space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
* [MLX Space](https://huggingface.co/spaces/mlx-community/mlx-my-repo)
* [AuoQuant Notebook](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing#scrollTo=ZC9Nsr9u5WhN)
