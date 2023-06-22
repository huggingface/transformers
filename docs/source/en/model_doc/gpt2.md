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

# OpenAI GPT2

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=gpt2">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-gpt2-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/gpt2">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

OpenAI GPT-2 model was proposed in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Alec
Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever from [OpenAI](https://huggingface.co/openai). It's a causal (unidirectional)
transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.

The abstract from the paper is the following:

*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million
web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some
text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks
across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than
10X the amount of data.*

Tips:

- GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- GPT-2 was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.
- The model can take the *past_key_values* (for PyTorch) or *past* (for TF) as input, which is the previously computed
  key/value attention pairs. Using this (*past_key_values* or *past*) value prevents the model from re-computing
  pre-computed values in the context of text generation. For PyTorch, see *past_key_values* argument of the
  [`GPT2Model.forward`] method, or for TF the *past* argument of the
  [`TFGPT2Model.call`] method for more information on its usage.
- Enabling the *scale_attn_by_inverse_layer_idx* and *reorder_and_upcast_attn* flags will apply the training stability
  improvements from [Mistral](https://github.com/stanford-crfm/mistral/) (for PyTorch only).

[Write With Transformer](https://transformer.huggingface.co/doc/gpt2-large) is a webapp created and hosted by
Hugging Face showcasing the generative capabilities of several models. GPT-2 is one of them and is available in five
different sizes: small, medium, large, xl and a distilled version of the small checkpoint: *distilgpt-2*.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://openai.com/blog/better-language-models/).

## Training

GPT-2 is a decoder-only Transformer. It is trained using teacher forcing. This means that during training, we feed the ground truth
previous tokens to the model in order to predict the next one. During training, one needs to provide the input sequences to the model as
`input_ids`, as well as `labels`. The latter are identical to the `input_ids`. In teacher-forcing style, the model will internally shift the
logits before calculating the loss. This ensures that the model is trained to predict the next token.

One can use [`GPT2LMHeadModel`] (or the Tensorflow/Flax variant), which includes the language modeling head on top of the Transformer decoder.

Suppose that we want to train the model for causal language modeling, then examples should be prepared for the model as follows:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> input_ids = tokenizer("Hugging Face is a company based in Paris and New York.", return_tensors="pt").input_ids
>>> labels = input_ids.clone()

>>> # the forward function internally shifts the logits to predict the next tokens and computes a cross-entropy loss
>>> loss = model(input_ids=input_ids, labels=labels).loss
>>> loss.item()
4.075357437133789
```

As you can see, only 2 inputs are required for the model in order to compute a loss: `input_ids` (which are the
`input_ids` of the encoded text) and `labels` (which are an exact copy of the `input_ids`, representing the text to be predicted).

However, the example above only shows a single training example. In practice, one trains deep learning models in
batches. What is typically done is one takes a large corpus of text, concatenates all text into one long sequence, and then chunks them up
into smaller blocks. Hence GPT-2 style models typically don't require padding or truncation of text during training, as one simply trains on blocks of
continuous text.

## Inference

At inference time, it is recommended to use [`~generation.GenerationMixin.generate`]. This method takes care of encoding the
input and auto-regressively generates the next tokens. Check out [this blog post](https://huggingface.co/blog/how-to-generate) to know
all the details about generating text with Transformers.

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> prompt = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.
... Even more surprising to the researchers was the fact that the unicorns spoke perfect English."""

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
>>> outputs = model.generate(input_ids, max_new_tokens=50)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.
Even more surprising to the researchers was the fact that the unicorns spoke perfect English. "The unicorns were very intelligent, and they were very intelligent," said Dr. David S. Siegel, a professor of anthropology at the University of California, Berkeley. "They were very intelligent, and they were very intelligent, and
```

The example above only shows a single example. You can also do batched inference, like so:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> tokenizer.padding_side = "left"

>>> # Define PAD Token = EOS Token = 50256
>>> tokenizer.pad_token = tokenizer.eos_token
>>> model.config.pad_token_id = model.config.eos_token_id

>>> # use different length sentences to test batching
>>> sentences = [
...         "Hello, my dog is a little",
...          "Today, I",
... ]

>>> inputs = tokenizer(sentences, return_tensors="pt", padding=True)

>>> output_sequences = model.generate(
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     do_sample=False,  # disable sampling to test if batching affects output
... )

>>> print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
["Hello, my dog is a little bit of a mess. I'm not sure if he's going", "Today, I'm going to be doing a lot of research on this. I"]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GPT2. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-generation"/>

- A blog on how to [Finetune a non-English GPT-2 Model with Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).
- A blog on [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) with GPT-2.
- A blog on [Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot), a large GPT-2 model.
- A blog on [Faster Text Generation with TensorFlow and XLA](https://huggingface.co/blog/tf-xla-generate) with GPT-2.
- A blog on [How to train a Language Model with Megatron-LM](https://huggingface.co/blog/megatron-training) with a GPT-2 model.
- A notebook on how to [finetune GPT2 to generate lyrics in the style of your favorite artist](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎
- A notebook on how to [finetune GPT2 to generate tweets in the style of your favorite Twitter user](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎
- [Causal language modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) chapter of the 🤗 Hugging Face Course.
- [`GPT2LMHeadModel`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [text generation example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation), and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFGPT2LMHeadModel`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxGPT2LMHeadModel`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb).
- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2 specific outputs

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

## GPT2Model

[[autodoc]] GPT2Model
    - forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel
    - forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel
    - forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering
    - forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification
    - forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification
    - forward

## TFGPT2Model

[[autodoc]] TFGPT2Model
    - call

## TFGPT2LMHeadModel

[[autodoc]] TFGPT2LMHeadModel
    - call

## TFGPT2DoubleHeadsModel

[[autodoc]] TFGPT2DoubleHeadsModel
    - call

## TFGPT2ForSequenceClassification

[[autodoc]] TFGPT2ForSequenceClassification
    - call

## TFSequenceClassifierOutputWithPast

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutputWithPast

## TFGPT2Tokenizer

[[autodoc]] TFGPT2Tokenizer

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model
    - __call__

## FlaxGPT2LMHeadModel

[[autodoc]] FlaxGPT2LMHeadModel
    - __call__
