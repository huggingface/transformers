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

# BLOOM

## Overview

The BLOOM model has been proposed with its various versions through the [BigScience Workshop](https://bigscience.huggingface.co/). BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:

- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom) (176B parameters)

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLOOM. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

See also:
- [Causal language modeling task guide](../tasks/language_modeling)
- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)


⚡️ Inference
- A blog on [Optimization story: Bloom inference](https://huggingface.co/blog/bloom-inference-optimization).
- A blog on [Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts).

⚙️ Training
- A blog on [The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed).

## BloomConfig

[[autodoc]] BloomConfig
    - all

## BloomTokenizerFast

[[autodoc]] BloomTokenizerFast
    - all


<frameworkcontent>
<pt>

## BloomModel

[[autodoc]] BloomModel
    - forward

## BloomForCausalLM

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering
    - forward

</pt>
<jax>

## FlaxBloomModel

[[autodoc]] FlaxBloomModel
    - __call__

## FlaxBloomForCausalLM

[[autodoc]] FlaxBloomForCausalLM
    - __call__

</jax>
</frameworkcontent>


