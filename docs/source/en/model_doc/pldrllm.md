<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# PLDR-LLM

## Overview

PLDR-LLM is a large language model from power law decoder representations with KV-cache and G-cache support, which is a new foundational language model architecture that utilizes power law graph attention to generate deductive and inductive outputs. The deductive outputs can be used to observe and regualarize the model, and inductive outputs are the usual next-token prediction of an auto-regressive language model. PLDR-LLM learns a generalizable tensor operator through the power law graph attention (PLGA) module in each decoder layer. This characteristic enables the caching of the G tensor values, which can replace the deep PLGA neural net. 

The PLDR-LLM model was first proposed in [PLDR-LLM: Large Language Model from Power Law Decoder Representations](https://huggingface.co/papers/2410.16703) by Burc Gokden at Fromthesky Research Labs LLC.

The abstract from the paper is the following:

We present the Large Language Model from Power Law Decoder Representations (PLDR-LLM), a language model that leverages non-linear and linear transformations through Power Law Graph Attention mechanism to generate well-defined deductive and inductive outputs. We pretrain the PLDR-LLMs of varying layer sizes with a small batch size of 32 and ~8B tokens from the RefinedWeb dataset, and show that they achieve competitive performance in zero-shot and few-shot settings compared to scaled dot-product LLMs of similar model size reported in the literature. We show that deductive outputs of PLDR-LLMs can be used to compare model characteristics or improve the performance by introducing the Directed Acyclic Graph (DAG) loss as a metric and regularizer. Our results indicate that the initial maximum learning rate and warm-up steps have a lasting impact on deductive outputs throughout the pretraining. We provide a detailed description of PLDR-LLM architecture, its implementation and the pretraining procedure.

Original implementation in pytorch is from:
- [PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference](https://huggingface.co/papers/2502.13502)

Another paper explaining the details of the architecture are:
- [Power Law Graph Transformer for Machine Translation and Representation Learning](https://huggingface.co/papers/2107.02039)



## Notes:

- `cache_first_G=True` can be set for batched inference to enable G-cache from the first prompt in the batch used for all samples in the batch.
- `custom_G_type` configuration indicates whether model was pretrained with learned G values (`None`) or predefined tensors(`'identity'`, `'random'`, `'external'`). Models with `custom_G_type='identity'` are equivalent to LLMs with Scaled Dot Product (SDPA) attention.
- `reference_rope=True` configuration indicates that the model was pretrained with RoPE implementation used in the [original paper](https://huggingface.co/papers/2502.13502) and this is the case for models pretrained using implementation in pytorch for the paper.
- `output_pldr_attentions=True` returns the deductive outputs and learnable parameters of power law graph attention module as tuple containing:
the output of the residual metric learner (metric tensor, **A**), output (**A<sub>LM</sub>**) after application of iSwiGLU on metric tensor, learned exponents of potential tensor, learned weights for energy-curvature tensor, learned bias for energy-curvature tensor, energy-curvature tensor (**G<sub>LM<sub>**), and attention weights.

This model was contributed by [Burc Gokden](https://huggingface.co/fromthesky).
The original code from the paper can be found [here](https://github.com/burcgokden/PLDR-LLM-with-KVG-cache).

## Usage examples

Using `pipeline`:
```python
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="fromthesky/PLDR-LLM-v51-110M-3",
    device="cuda"
)

prompt="PLDR-LLM is a large language model architecture developed by Fromthesky Research Labs."
output=pipeline(prompt, top_p=0.6, top_k=0, temperature=1, do_sample=True, max_new_tokens=100)
print(output[0]["generated_text"])
```

Using `AutoModel`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device="cuda" # or "cpu"
model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="fromthesky/PLDR-LLM-v51-110M-3",
                                           device_map=device,
                                           trust_remote_code=True
                                          )
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="fromthesky/PLDR-LLM-v51-110M-3",
                                        add_eos_token=False,
                                        Legacy=False,
                                        trust_remote_code=True
                                       )
prompt="PLDR-LLM is a large language model architecture developed by Fromthesky Research Labs."
inputs = tokenizer([prompt], return_tensors="pt").to(device=device)
generated_ids = model.generate(**inputs,
                                     max_new_tokens=100, 
                                     top_p=0.6,
                                     top_k=0, 
                                     temperature=1, 
                                     do_sample=True,
                                     use_cache=True
                                    )
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```


## PldrllmConfig

[[autodoc]] PldrllmConfig

## PldrllmForCausalLM

[[autodoc]] PldrllmForCausalLM

## PldrllmModel

[[autodoc]] PldrllmModel
    - forward

## PldrllmPreTrainedModel

[[autodoc]] PldrllmPreTrainedModel
    - forward

## PldrllmForSequenceClassification

[[autodoc]] PldrllmForSequenceClassification

## PldrllmForQuestionAnswering

[[autodoc]] PldrllmForQuestionAnswering

## PldrllmForTokenClassification

[[autodoc]] PldrllmForTokenClassification

## PldrllmTokenizer

[[autodoc]] PldrllmTokenizer

## PldrllmTokenizerFast

[[autodoc]] PldrllmTokenizerFast
