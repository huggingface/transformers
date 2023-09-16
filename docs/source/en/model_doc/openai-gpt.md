<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OpenAI GPT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=openai-gpt">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-openai--gpt-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/openai-gpt">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

OpenAI GPT model was proposed in [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. It's a causal (unidirectional) transformer
pre-trained using language modeling on a large corpus will long range dependencies, the Toronto Book Corpus.

The abstract from the paper is the following:

*Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering,
semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to
perform adequately. We demonstrate that large gains on these tasks can be realized by generative pretraining of a
language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In
contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve
effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our
approach on a wide range of benchmarks for natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon
the state of the art in 9 out of the 12 tasks studied.*

Tips:

- GPT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- GPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.

[Write With Transformer](https://transformer.huggingface.co/doc/gpt) is a webapp created and hosted by Hugging Face
showcasing the generative capabilities of several models. GPT is one of them.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/openai/finetune-transformer-lm).

Note:

If you want to reproduce the original tokenization process of the *OpenAI GPT* paper, you will need to install `ftfy`
and `SpaCy`:

```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

If you don't install `ftfy` and `SpaCy`, the [`OpenAIGPTTokenizer`] will default to tokenize
using BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OpenAI GPT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A blog post on [outperforming OpenAI GPT-3 with SetFit for text-classification](https://www.philschmid.de/getting-started-setfit).
- See also: [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="text-generation"/>

- A blog on how to [Finetune a non-English GPT-2 Model with Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).
- A blog on [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) with GPT-2.
- A blog on [Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot), a large GPT-2 model.
- A blog on [Faster Text Generation with TensorFlow and XLA](https://huggingface.co/blog/tf-xla-generate) with GPT-2.
- A blog on [How to train a Language Model with Megatron-LM](https://huggingface.co/blog/megatron-training) with a GPT-2 model.
- A notebook on how to [finetune GPT2 to generate lyrics in the style of your favorite artist](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎
- A notebook on how to [finetune GPT2 to generate tweets in the style of your favorite Twitter user](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎
- [Causal language modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) chapter of the 🤗 Hugging Face Course.
- [`OpenAIGPTLMHeadModel`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [text generation example script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFOpenAIGPTLMHeadModel`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- See also: [Causal language modeling task guide](../tasks/language_modeling)

<PipelineTag pipeline="token-classification"/>

- A course material on [Byte-Pair Encoding tokenization](https://huggingface.co/course/en/chapter6/5).

## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer
    - save_vocabulary

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## OpenAI specific outputs

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

[[autodoc]] models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel
    - forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel
    - forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel
    - forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification
    - forward

## TFOpenAIGPTModel

[[autodoc]] TFOpenAIGPTModel
    - call

## TFOpenAIGPTLMHeadModel

[[autodoc]] TFOpenAIGPTLMHeadModel
    - call

## TFOpenAIGPTDoubleHeadsModel

[[autodoc]] TFOpenAIGPTDoubleHeadsModel
    - call

## TFOpenAIGPTForSequenceClassification

[[autodoc]] TFOpenAIGPTForSequenceClassification
    - call
