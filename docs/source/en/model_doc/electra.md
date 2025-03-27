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

# ELECTRA

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The ELECTRA model was proposed in the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators](https://openreview.net/pdf?id=r1xMH1BtvB). ELECTRA is a new pretraining approach which trains two
transformer models: the generator and the discriminator. The generator's role is to replace tokens in a sequence, and
is therefore trained as a masked language model. The discriminator, which is the model we're interested in, tries to
identify which tokens were replaced by the generator in the sequence.

The abstract from the paper is the following:

*Masked language modeling (MLM) pretraining methods such as BERT corrupt the input by replacing some tokens with [MASK]
and then train a model to reconstruct the original tokens. While they produce good results when transferred to
downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a
more sample-efficient pretraining task called replaced token detection. Instead of masking the input, our approach
corrupts it by replacing some tokens with plausible alternatives sampled from a small generator network. Then, instead
of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that
predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments
demonstrate this new pretraining task is more efficient than MLM because the task is defined over all input tokens
rather than just the small subset that was masked out. As a result, the contextual representations learned by our
approach substantially outperform the ones learned by BERT given the same model size, data, and compute. The gains are
particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained
using 30x more compute) on the GLUE natural language understanding benchmark. Our approach also works well at scale,
where it performs comparably to RoBERTa and XLNet while using less than 1/4 of their compute and outperforms them when
using the same amount of compute.*

This model was contributed by [lysandre](https://huggingface.co/lysandre). The original code can be found [here](https://github.com/google-research/electra).

## Usage tips

- ELECTRA is the pretraining approach, therefore there is nearly no changes done to the underlying model: BERT. The
  only change is the separation of the embedding size and the hidden size: the embedding size is generally smaller,
  while the hidden size is larger. An additional projection layer (linear) is used to project the embeddings from their
  embedding size to the hidden size. In the case where the embedding size is the same as the hidden size, no projection
  layer is used.
- ELECTRA is a transformer model pretrained with the use of another (small) masked language model. The inputs are corrupted by that language model, which takes an input text that is randomly masked and outputs a text in which ELECTRA has to predict which token is an original and which one has been replaced. Like for GAN training, the small language model is trained for a few steps (but with the original texts as objective, not to fool the ELECTRA model like in a traditional GAN setting) then the ELECTRA model is trained for a few steps.
- The ELECTRA checkpoints saved using [Google Research's implementation](https://github.com/google-research/electra)
  contain both the generator and discriminator. The conversion script requires the user to name which model to export
  into the correct architecture. Once converted to the HuggingFace format, these checkpoints may be loaded into all
  available ELECTRA models, however. This means that the discriminator may be loaded in the
  [`ElectraForMaskedLM`] model, and the generator may be loaded in the
  [`ElectraForPreTraining`] model (the classification head will be randomly initialized as it
  doesn't exist in the generator).

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```py
import torch
from transformers import ElectraModel
model = ElectraModel.from_pretrained("google/electra-base-generator", torch_dtype=torch.float16, attn_implementation="sdpa")
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (NVIDIA Tesla T4-15GB, PyTorch 2.5.1, OS Ubuntu 22.04.3 LTS) using float16 with `google/electra-base-generator` and `google/electra-base-discriminator` models with a MaskedLM head, we saw the following speedups during training and inference.

#### Benchmarks For `google/electra-base-generator`
#### Training Benchmark

| num_training_steps | batch_sizes | sequence_lengths | is_cuda | is_half | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | sdpa peak mem (MB) | Mem saving (%) |
|--------------------|-------------|------------------|---------|---------|----------------------------|---------------------------|-------------|---------------------|--------------------|----------------|
| 100                | 2           | 32               | True    | True    | 0.0306                     | 0.024                     | 27.5264     | 249.684             | 249.684            | 0.0            |
| 100                | 2           | 64               | True    | True    | 0.0306                     | 0.024                     | 29.1752     | 253.593             | 253.593            | 0.0            |
| 100                | 2           | 128              | True    | True    | 0.0296                     | 0.023                     | 27.5778     | 261.409             | 261.409            | 0.0            |
| 100                | 2           | 256              | True    | True    | 0.0298                     | 0.023                     | 28.2746     | 299.593             | 277.04             | 8.141          |
| 100                | 4           | 32               | True    | True    | 0.0304                     | 0.024                     | 27.8372     | 253.593             | 253.593            | 0.0            |
| 100                | 4           | 64               | True    | True    | 0.0298                     | 0.023                     | 28.8354     | 261.409             | 261.409            | 0.0            |
| 100                | 4           | 128              | True    | True    | 0.0294                     | 0.0232                    | 28.4396     | 283.864             | 277.04             | 2.463          |
| 100                | 4           | 256              | True    | True    | 0.0312                     | 0.0258                    | 21.7372     | 517.463             | 453.529            | 14.097         |
| 100                | 8           | 32               | True    | True    | 0.0296                     | 0.0234                    | 26.9524     | 261.409             | 261.813            | -0.154         |
| 100                | 8           | 64               | True    | True    | 0.0302                     | 0.023                     | 30.7350     | 277.243             | 277.446            | -0.073         |
| 100                | 8           | 128              | True    | True    | 0.0312                     | 0.025                     | 24.5406     | 486.006             | 453.529            | 7.161          |
| 100                | 8           | 256              | True    | True    | 0.0484                     | 0.042                     | 14.5422     | 947.451             | 821.915            | 15.274         |
| 100                | 16          | 32               | True    | True    | 0.0298                     | 0.0232                    | 29.4218     | 277.446             | 277.446            | 0.0            |
| 100                | 16          | 64               | True    | True    | 0.0308                     | 0.025                     | 24.5678     | 470.277             | 454.745            | 3.416          |
| 100                | 16          | 128              | True    | True    | 0.0466                     | 0.041                     | 13.2722     | 884.536             | 821.915            | 7.619          |
| 100                | 16          | 256              | True    | True    | 0.0970                     | 0.0832                    | 16.9086     | 1804.67             | 1553.798           | 16.146         |

#### Inference Benchmark

| num_batches | batch_sizes | sequence_lengths | is_cuda | is_half | use_mask | Per token latency (eager - ms) | Per token latency (sdpa - ms) | Speedup (%) | Mem eager (MB) | Mem sdpa (MB) | Mem saved (%) |
|-------------|-------------|------------------|---------|---------|----------|-------------------------------|------------------------------|-------------|----------------|---------------|---------------|
| 50          | 2           | 32               | True    | True    | True     | 0.1612                        | 0.1174                       | 37.65       | 85.004         | 85.004        | 0.0           |
| 50          | 2           | 64               | True    | True    | True     | 0.078                         | 0.0588                       | 32.2768     | 92.949         | 92.949        | 0.0           |
| 50          | 2           | 128              | True    | True    | True     | 0.0424                        | 0.029                        | 45.766      | 108.84         | 109.043       | -0.186        |
| 50          | 2           | 256              | True    | True    | True     | 0.0206                        | 0.016                        | 28.5706     | 140.825        | 140.825       | 0.0           |
| 50          | 4           | 32               | True    | True    | True     | 0.0766                        | 0.0588                       | 30.376      | 92.949         | 92.949        | 0.0           |
| 50          | 4           | 64               | True    | True    | True     | 0.0384                        | 0.029                        | 32.6964     | 109.043        | 109.043       | 0.0           |
| 50          | 4           | 128              | True    | True    | True     | 0.0202                        | 0.0158                       | 27.629      | 140.825        | 140.825       | 0.0           |
| 50          | 4           | 256              | True    | True    | True     | 0.011                         | 0.009                        | 22.6098     | 204.997        | 204.997       | 0.0           |
| 50          | 8           | 32               | True    | True    | True     | 0.0382                        | 0.029                        | 32.3578     | 109.043        | 109.043       | 0.0           |
| 50          | 8           | 64               | True    | True    | True     | 0.0192                        | 0.0158                       | 24.546      | 140.825        | 140.825       | 0.0           |
| 50          | 8           | 128              | True    | True    | True     | 0.011                         | 0.009                        | 22.6078     | 204.997        | 204.997       | 0.0           |
| 50          | 8           | 256              | True    | True    | True     | 0.0072                        | 0.0062                       | 15.3892     | 332.935        | 332.935       | 0.0           |
| 50          | 16          | 32               | True    | True    | True     | 0.0196                        | 0.0152                       | 28.3326     | 140.825        | 140.825       | 0.0           |
| 50          | 16          | 64               | True    | True    | True     | 0.011                         | 0.009                        | 22.301      | 204.997        | 204.997       | 0.0           |
| 50          | 16          | 128              | True    | True    | True     | 0.007                         | 0.006                        | 13.89       | 332.935        | 332.935       | 0.0           |
| 50          | 16          | 256              | True    | True    | True     | 0.0072                        | 0.0062                       | 17.4658     | 585.568        | 585.568       | 0.0           |

#### Benchmarks For `google/electra-base-discriminator`
#### Training Benchmark

| num_training_steps | batch_sizes | sequence_lengths | is_cuda | is_half | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | sdpa peak mem (MB) | Mem saving (%) |
|--------------------|-------------|------------------|---------|---------|----------------------------|---------------------------|-------------|---------------------|--------------------|----------------|
| 100                | 2           | 32               | True    | True    | 0.0312                     | 0.0242                    | 29.8518     | 568.647             | 566.126            | 0.445          |
| 100                | 2           | 64               | True    | True    | 0.0308                     | 0.024                     | 29.3112     | 572.187             | 570.846            | 0.235          |
| 100                | 2           | 128              | True    | True    | 0.0312                     | 0.0248                    | 25.9492     | 588.412             | 587.232            | 0.201          |
| 100                | 2           | 256              | True    | True    | 0.031                      | 0.0298                    | 5.5548      | 631.001             | 602.644            | 4.705          |
| 100                | 4           | 32               | True    | True    | 0.0312                     | 0.0238                    | 30.2574     | 572.187             | 570.846            | 0.235          |
| 100                | 4           | 64               | True    | True    | 0.0312                     | 0.0242                    | 28.247      | 588.412             | 587.232            | 0.201          |
| 100                | 4           | 128              | True    | True    | 0.0314                     | 0.0286                    | 10.2442     | 604.379             | 602.644            | 0.288          |
| 100                | 4           | 256              | True    | True    | 0.053                      | 0.0498                    | 6.4654      | 1022.623            | 831.197            | 23.0304        |
| 100                | 8           | 32               | True    | True    | 0.031                      | 0.0242                    | 28.8872     | 587.263             | 586.813            | 0.077          |
| 100                | 8           | 64               | True    | True    | 0.0308                     | 0.0278                    | 10.7408     | 602.151             | 601.596            | 0.0924         |
| 100                | 8           | 128              | True    | True    | 0.0492                     | 0.0474                    | 3.8086      | 927.308             | 831.197            | 11.5626        |
| 100                | 8           | 256              | True    | True    | 0.105                      | 0.0976                    | 7.4594      | 1798.541            | 1418.122           | 26.8258        |
| 100                | 16          | 32               | True    | True    | 0.0314                     | 0.0286                    | 10.8604     | 598.278             | 595.934            | 0.3934         |
| 100                | 16          | 64               | True    | True    | 0.048                      | 0.0464                    | 3.7276      | 877.185             | 828.890            | 5.8264         |
| 100                | 16          | 128              | True    | True    | 0.0986                     | 0.0944                    | 4.3308      | 1605.603            | 1415.605           | 13.4216        |
| 100                | 16          | 256              | True    | True    | 0.2188                     | 0.2036                    | 7.4636      | 3318.753            | 2564.827           | 29.395         |

#### Inference Benchmark

| num_batches | batch_sizes | sequence_lengths | is_cuda | is_half | use_mask | Per token latency (eager - ms) | Per token latency (sdpa - ms) | Speedup (%) | Mem eager (MB) | Mem sdpa (MB) | Mem saved (%) |
|-------------|-------------|------------------|---------|---------|----------|-------------------------------|------------------------------|-------------|----------------|---------------|---------------|
| 50          | 2           | 32               | True    | True    | True     | 0.1528                        | 0.1178                       | 29.8324     | 244.13         | 243.868       | 0.107         |
| 50          | 2           | 64               | True    | True    | True     | 0.0744                        | 0.056                        | 33.508      | 252.141        | 251.878       | 0.104         |
| 50          | 2           | 128              | True    | True    | True     | 0.0416                        | 0.0298                       | 39.0368     | 268.163        | 268.103       | 0.022         |
| 50          | 2           | 256              | True    | True    | True     | 0.021                         | 0.0182                       | 15.2404     | 300.409        | 300.147       | 0.087         |
| 50          | 4           | 32               | True    | True    | True     | 0.0728                        | 0.0568                       | 27.5098     | 252.141        | 251.878       | 0.104         |
| 50          | 4           | 64               | True    | True    | True     | 0.037                         | 0.03                         | 23.255      | 268.365        | 268.103       | 0.098         |
| 50          | 4           | 128              | True    | True    | True     | 0.0208                        | 0.0176                       | 16.2526     | 300.409        | 300.147       | 0.087         |
| 50          | 4           | 256              | True    | True    | True     | 0.0204                        | 0.0184                       | 7.9674      | 365.106        | 364.844       | 0.072         |
| 50          | 8           | 32               | True    | True    | True     | 0.0382                        | 0.0294                       | 28.8184     | 268.365        | 268.103       | 0.098         |
| 50          | 8           | 64               | True    | True    | True     | 0.0204                        | 0.0176                       | 16.746      | 300.409        | 300.147       | 0.087         |
| 50          | 8           | 128              | True    | True    | True     | 0.0188                        | 0.0184                       | 2.9468      | 365.106        | 364.844       | 0.072         |
| 50          | 8           | 256              | True    | True    | True     | 0.0214                        | 0.0198                       | 9.643       | 494.093        | 493.962       | 0.027         |
| 50          | 16          | 32               | True    | True    | True     | 0.0202                        | 0.0186                       | 8.4296      | 300.409        | 300.147       | 0.087         |
| 50          | 16          | 64               | True    | True    | True     | 0.0184                        | 0.0184                       | -1.118      | 365.106        | 364.844       | 0.072         |
| 50          | 16          | 128              | True    | True    | True     | 0.0204                        | 0.0194                       | 2.7932      | 494.093        | 493.962       | 0.027         |
| 50          | 16          | 256              | True    | True    | True     | 0.0234                        | 0.0212                       | 11.1332     | 748.823        | 748.561       | 0.035         |

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra specific outputs

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

<frameworkcontent>
<pt>

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

</pt>
<tf>

## TFElectraModel

[[autodoc]] TFElectraModel
    - call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining
    - call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM
    - call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification
    - call

## TFElectraForMultipleChoice

[[autodoc]] TFElectraForMultipleChoice
    - call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification
    - call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering
    - call

</tf>
<jax>

## FlaxElectraModel

[[autodoc]] FlaxElectraModel
    - __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining
    - __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM
    - __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM
    - __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification
    - __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice
    - __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification
    - __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>
