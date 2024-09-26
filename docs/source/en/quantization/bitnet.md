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

# BitNet

[BitNet](https://arxiv.org/abs/2402.17764) replaces traditional Linear layers in Multi-Head Attention and Feed-Forward Networks with specialized layers called BitLinear with ternary (or binary in the older version) precision. The BitLinear layers introduced here quantize the weights using ternary precision (with values of -1, 0, and 1) and quantize the activations to 8-bit precision.


<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/bitlinear.png" alt="Alt Text" />
  <figcaption>The architecture of BitNet with BitLinear layers</figcaption>
</figure>

During training, we start by quantizing the weights into ternary values, using symmetric per tensor quantization. First, we compute the average of the absolute values of the weight matrix and use this as a scale. We then divide the weights by the scale, round the values, constrain them between -1 and 1, and finally rescale them to continue in full precision.

$$
scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|}
$$

$$
W_q = \text{clamp}_{[-1,1]}(\text{round}(W*scale))
$$

$$
W_{dequantized} = W_q*scale_w
$$

Activations are then quantized to a specified bit-width (e.g., 8-bit) using [absmax](https://arxiv.org/pdf/2208.07339) quantization (symmetric per channel quantization). This involves scaling the activations into a range [−128,127[. The quantization formula is:

$$
scale_x = \frac{127}{|X|_{\text{max}, \, \text{dim}=-1}}
$$

$$
X_q = \text{clamp}_{[-128,127]}(\text{round}(X*scale))
$$

$$
X_{dequantized} = X_q * scale_x
$$

To learn more about how we trained, and fine-tuned bitnet models checkout the blogpost [here](https://huggingface.co/blog/1_58_llm_extreme_quantization)

## Load a BitNet Model from the Hub
BitNet models can't be quantized on the fly—they need to be pre-trained or fine-tuned with the quantization applied (it's a Quantization aware training technique). Once trained, these models are already quantized and available as packed versions on the hub.

A quantized model can be load : 

```py
from transformers import AutoModelForCausalLM
path = "/path/to/model"
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
```
## Pre-training / Fine-tuning a BitNet Model

If you're looking to pre-train or fine-tune your own 1.58-bit model using Nanotron, check out this [PR](https://github.com/huggingface/nanotron/pull/180), all you need to get started is there !

For fine-tuning, you'll need to convert the model from Hugging Face format to Nanotron format (which has some differences). You can find the conversion steps in this [PR](https://github.com/huggingface/nanotron/pull/174).

## Kernels

In our initial version, we chose to use `@torch.compile` to unpack the weights and perform the forward pass. It’s very straightforward to implement and delivers significant speed improvements. We plan to integrate additional optimized kernels in future versions.