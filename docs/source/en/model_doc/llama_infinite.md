<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LlamaInfinite

## Overview

The LlamaInfinite model implementes LM-Infinite technique on the [Llama Model](https://huggingface.co/docs/transformers/main/model_doc/llama).
LM-Infinite was proposed in [LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models](https://arxiv.org/abs/2308.16137) by Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji and Sinong Wang.
It is the first ever proposed simple and effective technique to extend lengths of large language models, without any parameter updates. In the paper, results show that LM-Infinite can encode as long as 128k tokens on a single A100 GPU, and allows generating to infinite tokens, thanks to its $O(n)$ time and space complexity for encoding and $O(1)$ time and space complexity for decoding.
The current implementation supports feeding the whole long sequence, instead of feeding tokens one by one.

The abstract from the paper is the following:

*In recent years, there have been remarkable advancements in the performance of Transformer-based Large Language Models (LLMs) across various domains. As these LLMs are deployed for increasingly complex domains, they often face the need to follow longer user prompts or generate longer texts. In these situations, the length generalization failure of LLMs on long sequences becomes more prominent. Most pre-training schemes truncate training sequences to a fixed length. LLMs often struggle to generate fluent and coherent texts after longer contexts, even with relative positional encoding specifically designed to cope with this problem. Common solutions such as finetuning on longer corpora often involve daunting hardware and time costs and require careful training process design. To more efficiently extrapolate existing LLMs’ generation quality to longer texts, we theoretically and empirically investigate the main out-of-distribution (OOD) factors contributing to this problem. Inspired by this diagnosis, we propose a simple yet effective solution for on-the-fly length generalization, LM-Infinite. It involves only a $\Lambda$-shaped attention mask (to avoid excessive attended tokens) and a distance limit (to avoid unseen distances) while requiring no parameter updates or learning. We find it applicable to a variety of LLMs using relative-position encoding methods. LM-Infinite is computationally efficient with $O(n)$ time and space, and demonstrates consistent text generation fluency and quality to as long as 128k tokens on ArXiv and OpenWebText2 datasets, with 2.72x decoding speedup.*


Tips:

To use the LlamaInfinite model is extremely simple. Now you only need to substitute the Llama model with LlamaInfinite model in your code, while using the original LlamaTokenizer.

```

from transformers import LlamaInfiniteForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaInfiniteForCausalLM.from_pretrained("/output/path")

```


For example, if you want to use the LLaMA model from `decapoda-research/llama-7b-hf` checkpoint, simple run:

```

from transformers import LlamaInfiniteForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaInfiniteForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

```


If you have downloaded the Llama-2 checkpoints under the `llama-2` directory, you can load them using the following code:


```

from transformers import LlamaInfiniteForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("llama-2/llama-2-7b-hf")
model = LlamaInfiniteForCausalLM.from_pretrained("llama-2")

```



The functionalities for encoding and decoding are the same as the original Llama model, which is a great benefit if you are already familiar with the Huggingface Transformers library.
For more information on how to use Llama familty, please refer to the [Llama Model](https://huggingface.co/docs/transformers/v4.34.0/model_doc/llama) documentation.
Note that if you generate sequences with longer than 2048 tokens, the model will warn. You can safely ignore this warning for now.


## LlamaInfiniteConfig

[[autodoc]] LlamaInfiniteConfig



## LlamaTokenizer

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary


## LlamaInfiniteModel

[[autodoc]] LlamaInfiniteModel
    - forward


## LlamaInfiniteForCausalLM

[[autodoc]] LlamaInfiniteForCausalLM
    - forward

## LlamaInfiniteForSequenceClassification

[[autodoc]] LlamaInfiniteForSequenceClassification
    - forward

