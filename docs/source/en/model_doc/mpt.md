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

# MPT

## Overview

The MPT model was proposed by the [MosaicML](https://www.mosaicml.com/) team and released with multiple sizes and finetuned variants. The MPT models is a series of open source and commercially usable LLMs pre-trained on 1T tokens. 

MPT models are GPT-style decoder-only transformers with several improvements: performance-optimized layer implementations, architecture changes that provide greater training stability, and the elimination of context length limits by replacing positional embeddings with ALiBi. 

- MPT base: MPT base pre-trained models on next token prediction 
- MPT instruct: MPT base models fine-tuned on instruction based tasks
- MPT storywriter: MPT base models fine-tuned for 2500 steps on 65k-token excerpts of fiction books contained in the books3 corpus, this enables the model to handle very long sequences

The original code is available at the  [`llm-foundry`](https://github.com/mosaicml/llm-foundry/tree/main) repository.

Read more about it [in the release blogpost](https://www.mosaicml.com/blog/mpt-7b)

Tips:

- Learn more about some techniques behind training of the model [in this section of llm-foundry repository](https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#faqs)
- If you want to use the advanced version of the model (triton kernels, direct flash attention integration), you can still use the original model implementation by adding `trust_remote_code=True` when calling `from_pretrained`.

- [Fine-tuning Notebook](https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing) on how to fine-tune MPT-7B on a free Google Colab instance to turn the model into a Chatbot.


## MptConfig

[[autodoc]] MptConfig
    - all

## MptModel

[[autodoc]] MptModel
    - forward

## MptForCausalLM

[[autodoc]] MptForCausalLM
    - forward

## MptForSequenceClassification

[[autodoc]] MptForSequenceClassification
    - forward

## MptForTokenClassification

[[autodoc]] MptForTokenClassification
    - forward

## MptForQuestionAnswering

[[autodoc]] MptForQuestionAnswering
    - forward
