<!--Copyright 2025 MiniMaxAI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MiniMax

## Overview

The MiniMax-Text-01 model was proposed in [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://huggingface.co/papers/2501.08313) by MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu.

The abstract from the paper is the following:

*We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01, which are comparable to top-tier models while offering superior capabilities in processing longer contexts. The core lies in lightning attention and its efficient scaling. To maximize computational capacity, we integrate it with Mixture of Experts (MoE), creating a model with 32 experts and 456 billion total parameters, of which 45.9 billion are activated for each token. We develop an optimized parallel strategy and highly efficient computation-communication overlap techniques for MoE and lightning attention. This approach enables us to conduct efficient training and inference on models with hundreds of billions of parameters across contexts spanning millions of tokens. The context window of MiniMax-Text-01 can reach up to 1 million tokens during training and extrapolate to 4 million tokens during inference at an affordable cost. Our vision-language model, MiniMax-VL-01 is built through continued training with 512 billion vision-language tokens. Experiments on both standard and in-house benchmarks show that our models match the performance of state-of-the-art models like GPT-4o and Claude-3.5-Sonnet while offering 20-32 times longer context window.*

### Architectural details

MiniMax is a powerful language model with 456 billion total parameters, of which 45.9 billion are activated per token. To better unlock the long context capabilities of the model, MiniMax adopts a hybrid architecture that combines Lightning Attention, Softmax Attention and Mixture-of-Experts (MoE). Leveraging advanced parallel strategies and innovative compute-communication overlap methods—such as Linear Attention Sequence Parallelism Plus (LASP+), varlen ring attention, Expert Tensor Parallel (ETP), etc., MiniMax's training context length is extended to 1 million tokens, and it can handle a context of up to 4 million tokens during the inference. On various academic benchmarks, MiniMax also demonstrates the performance of a top-tier model.

The architecture of MiniMax is briefly described as follows:

- Total Parameters: 456B
- Activated Parameters per Token: 45.9B
- Number Layers: 80
- Hybrid Attention: a softmax attention is positioned after every 7 lightning attention.
    - Number of attention heads: 64
    - Attention head dimension: 128
- Mixture of Experts:
    - Number of experts: 32
    - Expert hidden dimension: 9216
    - Top-2 routing strategy
- Positional Encoding: Rotary Position Embedding (RoPE) applied to half of the attention head dimension with a base frequency of 10,000,000
- Hidden Size: 6144
- Vocab Size: 200,064

For more details refer to the [release blog post](https://www.minimaxi.com/en/news/minimax-01-series-2).

### License

`MiniMax` is released under the MINIMAX MODEL LICENSE AGREEMENT.

## Usage tips

The pre-trained model can be used as follows:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"Mayonnaise can be made as follows: (...)"
```

As can be seen, the instruction-tuned model requires a [chat template](../chat_templating) to be applied to make sure the inputs are prepared in the right format.

## Speeding up MiniMax by using Flash Attention

The code snippets above showcase inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention-2, refer to the snippet below:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf", dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

### Sliding window Attention

The current implementation supports the sliding window attention mechanism and memory efficient cache management. 
To enable sliding window attention, just make sure to have a `flash-attn` version that is compatible with sliding window attention (`>=2.3.0`). 

The Flash Attention-2 model uses also a more memory efficient cache slicing mechanism - as recommended per the official implementation of Mistral model that use rolling cache mechanism we keep the cache size fixed (`self.config.sliding_window`), support batched generation only for `padding_side="left"` and use the absolute position of the current token to compute the positional embedding.

## Shrinking down MiniMax using quantization

As the MiniMax model has 456 billion parameters, that would require about 912GB of GPU RAM in half precision (float16), since each parameter is stored in 2 bytes. However, one can shrink down the size of the model using [quantization](../quantization). If the model is quantized to 4 bits (or half a byte per parameter), about 228 GB of RAM is required.

Quantizing a model is as simple as passing a `quantization_config` to the model. Below, we'll leverage the bitsandbytes quantization library (but refer to [this page](../quantization) for alternative quantization methods):

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

>>> # specify how to quantize the model
>>> quantization_config = BitsAndBytesConfig(
...         load_in_4bit=True,
...         bnb_4bit_quant_type="nf4",
...         bnb_4bit_compute_dtype="torch.float16",
... )

>>> model = AutoModelForCausalLM.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf", quantization_config=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")

>>> prompt = "My favourite condiment is"

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

This model was contributed by [geetu040](https://github.com/geetu040) and [Shakib-IO](https://github.com/Shakib-IO).
The original code can be found [here](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MiniMax. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-generation"/>

- The [Alignment Handbook](https://github.com/huggingface/alignment-handbook) by Hugging Face includes scripts and recipes to perform supervised fine-tuning (SFT) and direct preference optimization with Mistral-7B. This includes scripts for full fine-tuning, QLoRa on a single GPU as well as multi-GPU fine-tuning.
- [Causal language modeling task guide](../tasks/language_modeling)

## MiniMaxConfig

[[autodoc]] MiniMaxConfig

## MiniMaxModel

[[autodoc]] MiniMaxModel
    - forward

## MiniMaxForCausalLM

[[autodoc]] MiniMaxForCausalLM
    - forward

## MiniMaxForSequenceClassification

[[autodoc]] MiniMaxForSequenceClassification
    - forward

## MiniMaxForTokenClassification

[[autodoc]] MiniMaxForTokenClassification
    - forward

## MiniMaxForQuestionAnswering
[[autodoc]] MiniMaxForQuestionAnswering
    - forward
