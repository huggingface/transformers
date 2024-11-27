<!--Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MegatronBERT

## Overview

The MegatronBERT model was proposed in [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism](https://arxiv.org/abs/1909.08053) by Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper and Bryan Catanzaro.

The abstract from the paper is the following:

*Recent work in language modeling demonstrates that training large transformer models advances the state of the art in
Natural Language Processing applications. However, very large models can be quite difficult to train due to memory
constraints. In this work, we present our techniques for training very large transformer models and implement a simple,
efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our
approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model
parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We
illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain
15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline
that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance
the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9
billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in
BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we
achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA
accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy
of 89.4%).*

This model was contributed by [jdemouth](https://huggingface.co/jdemouth). The original code can be found [here](https://github.com/NVIDIA/Megatron-LM). 
That repository contains a multi-GPU and multi-node implementation of the Megatron Language models. In particular, 
it contains a hybrid model parallel approach using "tensor parallel" and "pipeline parallel" techniques.

## Usage tips

We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) checkpoints
for use to evaluate or finetuning downstream tasks.

To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and setup the NVIDIA GPU Cloud (NGC)
Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

BERT-345M-uncased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip
```

BERT-345M-cased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip
```

Once you have obtained the checkpoints from NVIDIA GPU Cloud (NGC), you have to convert them to a format that will
easily be loaded by Hugging Face Transformers and our port of the BERT code.

The following commands allow you to do the conversion. We assume that the folder `models/megatron_bert` contains
`megatron_bert_345m_v0_1_{cased, uncased}.zip` and that the commands are run from inside that folder:

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip
```

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel
    - forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM
    - forward

## MegatronBertForCausalLM

[[autodoc]] MegatronBertForCausalLM
    - forward

## MegatronBertForNextSentencePrediction

[[autodoc]] MegatronBertForNextSentencePrediction
    - forward

## MegatronBertForPreTraining

[[autodoc]] MegatronBertForPreTraining
    - forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification
    - forward

## MegatronBertForMultipleChoice

[[autodoc]] MegatronBertForMultipleChoice
    - forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification
    - forward

## MegatronBertForQuestionAnswering

[[autodoc]] MegatronBertForQuestionAnswering
    - forward
