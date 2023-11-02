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

# ALBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=albert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-albert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/albert-base-v2">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The ALBERT model was proposed in [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma,
Radu Soricut. It presents two parameter-reduction techniques to lower memory consumption and increase the training
speed of BERT:

- Splitting the embedding matrix into two smaller matrices.
- Using repeating layers split among groups.

The abstract from the paper is the following:

*Increasing model size when pretraining natural language representations often results in improved performance on
downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations,
longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction
techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows
that our proposed methods lead to models that scale much better compared to the original BERT. We also use a
self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks
with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and
SQuAD benchmarks while having fewer parameters compared to BERT-large.*

This model was contributed by [lysandre](https://huggingface.co/lysandre). This model jax version was contributed by
[kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/google-research/ALBERT).

## Usage tips

- ALBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather
  than the left.
- ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains
  similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same
  number of (repeating) layers.
- Embedding size E is different from hidden size H justified because the embeddings are context independent (one embedding vector represents one token), whereas hidden states are context dependent (one hidden state represents a sequence of tokens) so it's more logical to have H >> E. Also, the embedding matrix is large since it's V x E (V being the vocab size). If E < H, it has less parameters.
- Layers are split in groups that share parameters (to save memory).
Next sentence prediction is replaced by a sentence ordering prediction: in the inputs, we have two sentences A and B (that are consecutive) and we either feed A followed by B or B followed by A. The model must predict if they have been swapped or not.

<<<<<<< HEAD
## Resources
=======

This model was contributed by [lysandre](https://huggingface.co/lysandre). This model jax version was contributed by
[kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/google-research/ALBERT).

<<<<<<< HEAD
## Documentation Resources
>>>>>>> b66685ec6 (Updated albert.md doc for ALBERT model)
=======
## Resources
>>>>>>> 428b241d6 (Update docs/source/en/model_doc/albert.md)

The resources provided in the following sections consist of a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with AlBERT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.


<PipelineTag pipeline="text-classification"/>

- A blog post on how to train AlBERT with Blurr for sequence classification can be done by adapting this [example](https://huggingface.co/blog/fastai).
- A blog post on how to use [Ray to tune hyperparameters](https://huggingface.co/blog/ray-tune) can be adapted for AlBERT.
- A notebook on how to [Fine-tune ALBERT for sentence-pair classification](https://github.com/NadirEM/nlp-notebooks/blob/master/Fine_tune_ALBERT_sentence_pair_classification.ipynb). 🌎
- A notebook on how to [Fine Tuning Transformer for MultiLabel Text Classification](https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)

- Adapt this blog post on [Getting Started with Sentiment Analysis using Python](https://huggingface.co/blog/sentiment-analysis-python) for AlBERT.

- [`AlBertForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
- [`TFAlBertForSequenceClassification`] can be done by adapting this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification).
- [`FlaxAlBertForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- Check the [Text classification task guide](../tasks/sequence_classification) on how to use the model.


<PipelineTag pipeline="token-classification"/>

- [`AlBertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) and can be implemented by adapting the example in this [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFAlBertForTokenClassification`] is supported by adapting this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`AlBertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) and by adapting the [ example in this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFAlBertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`FlaxAlBertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [`FlaxAlBertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Token classification task guide](../tasks/token_classification) on how to use the model.

<PipelineTag pipeline="fill-mask"/>

- [`AlBertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFAlBertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxAlBertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Masked language modeling task guide](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`AlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFAlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxAlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Question answering task guide](../tasks/question_answering) on how to use the model.

**Multiple choice**

- [`AlbertForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFAlbertForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

- Check the  [Multiple choice task guide](../tasks/multiple_choice) on how to use the model.

⚗️ Optimization

- Blog posts on methods and tools [ for efficient training on a single GPU
](https://huggingface.co/docs/transformers/perf_train_gpu_one), [efficient Inference on a Single GPU](https://huggingface.co/docs/transformers/perf_infer_gpu_one) and [efficient Inference on CPU
](https://huggingface.co/docs/transformers/perf_infer_cpu) are also available.
- A blog post on how [Optimizing Transformers for GPUs with 🤗 Optimum](https://www.philschmid.de/optimizing-transformers-with-optimum-gpu) can be modified for ALBERT.
- A blog post on [Optimizing Transformers with Hugging Face Optimum](https://huggingface.co/blog/optimum-inference) and [an example can be found here](https://www.philschmid.de/optimizing-transformers-with-optimum).


⚡️ Inference

- A blog post on how to [Accelerate BERT inference with Hugging Face Transformers and AWS Inferentia](https://huggingface.co/blog/bert-inferentia-sagemaker) can be adapted for AlBERT.
- A blog post on [Serverless Inference with Hugging Face's Transformers, DistilBERT and Amazon SageMaker](https://www.philschmid.de/sagemaker-serverless-huggingface-distilbert) can be adapted for AlBERT.



🚀 Deploy

- A blog post on how to [train ALBERT for natural language processing with TensorFlow on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/train-albert-for-natural-language-processing-with-tensorflow-on-amazon-sagemaker/).
- A post on how to [achieve 35% faster training with Hugging Face Deep Learning Containers on Amazon SageMaker](https://dataintegration.info/achieve-35-faster-training-with-hugging-face-deep-learning-containers-on-amazon-sagemaker).
- Adapt the guide on this blog post on how to [Deploy BERT with Hugging Face Transformers, Amazon SageMaker and Terraform module](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker) for ALBERT.

## AlbertConfig

[[autodoc]] AlbertConfig

## AlbertTokenizer

[[autodoc]] AlbertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## AlbertTokenizerFast

[[autodoc]] AlbertTokenizerFast

## Albert specific outputs

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput

[[autodoc]] models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput

<frameworkcontent>
<pt>

## AlbertModel

[[autodoc]] AlbertModel
    - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining
    - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM
    - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification
    - forward

## AlbertForMultipleChoice

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification
    - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering
    - forward

</pt>

<tf>

## TFAlbertModel

[[autodoc]] TFAlbertModel
    - call

## TFAlbertForPreTraining

[[autodoc]] TFAlbertForPreTraining
    - call

## TFAlbertForMaskedLM

[[autodoc]] TFAlbertForMaskedLM
    - call

## TFAlbertForSequenceClassification

[[autodoc]] TFAlbertForSequenceClassification
    - call

## TFAlbertForMultipleChoice

[[autodoc]] TFAlbertForMultipleChoice
    - call

## TFAlbertForTokenClassification

[[autodoc]] TFAlbertForTokenClassification
    - call

## TFAlbertForQuestionAnswering

[[autodoc]] TFAlbertForQuestionAnswering
    - call

</tf>
<jax>

## FlaxAlbertModel

[[autodoc]] FlaxAlbertModel
    - __call__

## FlaxAlbertForPreTraining

[[autodoc]] FlaxAlbertForPreTraining
    - __call__

## FlaxAlbertForMaskedLM

[[autodoc]] FlaxAlbertForMaskedLM
    - __call__

## FlaxAlbertForSequenceClassification

[[autodoc]] FlaxAlbertForSequenceClassification
    - __call__

## FlaxAlbertForMultipleChoice

[[autodoc]] FlaxAlbertForMultipleChoice
    - __call__

## FlaxAlbertForTokenClassification

[[autodoc]] FlaxAlbertForTokenClassification
    - __call__

## FlaxAlbertForQuestionAnswering

[[autodoc]] FlaxAlbertForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>


