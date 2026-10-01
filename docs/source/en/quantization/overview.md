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

Use the [quantization picker](https://huggingface.co/spaces/stevhliu/quantization-picker) to find a method for your model and hardware. Pick a model, your hardware, and what you need the method to do: load the model on the fly, fine-tune it with PEFT, compile it with `torch.compile()`, or save the quantized result. The picker shows whether the model fits in memory at each precision, including the KV cache, and the **Compare** view lists the methods that match.

<iframe
	src="https://stevhliu-quantization-picker.hf.space"
	frameborder="0"
	width="100%"
	height="900"
></iframe>

## Resources

If you are new to quantization, we recommend checking out these beginner-friendly quantization courses in collaboration with DeepLearning.AI.

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)

## User-Friendly Quantization Tools

If you are looking for a user-friendly quantization experience, you can use the following community spaces and notebooks:

* [Bitsandbytes Space](https://huggingface.co/spaces/bnb-community/bnb-my-repo)
* [GGUF Space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
* [MLX Space](https://huggingface.co/spaces/mlx-community/mlx-my-repo)
* [AutoQuant Notebook](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing#scrollTo=ZC9Nsr9u5WhN)
