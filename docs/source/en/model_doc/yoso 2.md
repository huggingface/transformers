<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# YOSO

## Overview

The YOSO model was proposed in [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://arxiv.org/abs/2111.09714)  
by Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. YOSO approximates standard softmax self-attention
via a Bernoulli sampling scheme based on Locality Sensitive Hashing (LSH). In principle, all the Bernoulli random variables can be sampled with
a single hash. 

The abstract from the paper is the following:

*Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is 
the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically 
on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling 
attention mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic complexity of such models to linear. 
We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random 
variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant). 
This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of 
LSH (to enable deployment on GPU architectures). We evaluate our algorithm on the GLUE benchmark with standard 512 sequence 
length where we see favorable performance relative to a standard pretrained Transformer. On the Long Range Arena (LRA) benchmark, 
for evaluating performance on long sequences, our method achieves results consistent with softmax self-attention but with sizable 
speed-ups and memory savings and often outperforms other efficient self-attention methods. Our code is available at this https URL*

Tips:

- The YOSO attention algorithm is implemented through custom CUDA kernels, functions written in CUDA C++ that can be executed multiple times
in parallel on a GPU.
- The kernels provide a `fast_hash` function, which approximates the random projections of the queries and keys using the Fast Hadamard Transform. Using these
hash codes, the `lsh_cumulation` function approximates self-attention via LSH-based Bernoulli sampling.
- To use the custom kernels, the user should set `config.use_expectation = False`. To ensure that the kernels are compiled successfully, 
the user must install the correct version of PyTorch and cudatoolkit. By default, `config.use_expectation = True`, which uses YOSO-E and 
does not require compiling CUDA kernels.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg"
alt="drawing" width="600"/> 

<small> YOSO Attention Algorithm. Taken from the <a href="https://arxiv.org/abs/2111.09714">original paper</a>.</small>

This model was contributed by [novice03](https://huggingface.co/novice03). The original code can be found [here](https://github.com/mlpen/YOSO).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## YosoConfig

[[autodoc]] YosoConfig


## YosoModel

[[autodoc]] YosoModel
    - forward


## YosoForMaskedLM

[[autodoc]] YosoForMaskedLM
    - forward


## YosoForSequenceClassification

[[autodoc]] YosoForSequenceClassification
    - forward

## YosoForMultipleChoice

[[autodoc]] YosoForMultipleChoice
    - forward


## YosoForTokenClassification

[[autodoc]] YosoForTokenClassification
    - forward


## YosoForQuestionAnswering

[[autodoc]] YosoForQuestionAnswering
    - forward