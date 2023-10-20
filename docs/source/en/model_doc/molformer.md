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

# MoLFormer

## Overview

The MoLFormer model was proposed in [Large-Scale Chemical Language Representations Capture Molecular Structure and Properties](https://arxiv.org/abs/2106.09553) by Jerret Ross, Brian Belgodere, Vijil Chenthamarakshan, Inkit Padhi, Youssef Mroueh, and Payel Das.
It features efficient linear attention and rotary positional embeddings and is pre-trained on a set of 1.1B molecular SMILES sequences from PubChem and ZINC.

The abstract from the paper is the following:

*Models based on machine learning can enable accurate and fast molecular property predictions, which is of interest in drug discovery and material design. Various supervised machine learning models have demonstrated promising performance, but the vast chemical space and the limited availability of property labels make supervised learning challenging. Recently, unsupervised transformer-based language models pretrained on a large unlabelled corpus have produced state-of-the-art results in many downstream natural language processing tasks. Inspired by this development, we present molecular embeddings obtained by training an efficient transformer encoder model, MoLFormer, which uses rotary positional embeddings. This model employs a linear attention mechanism, coupled with highly distributed training, on SMILES sequences of 1.1 billion unlabelled molecules from the PubChem and ZINC datasets. We show that the learned molecular representation outperforms existing baselines, including supervised and self-supervised graph neural networks and language models, on several downstream tasks from ten benchmark datasets. They perform competitively on two others. Further analyses, specifically through the lens of attention, demonstrate that MoLFormer trained on chemical SMILES indeed learns the spatial relationships between atoms within a molecule. These results provide encouraging evidence that large-scale molecular language models can capture sufficient chemical and structural information to predict various distinct molecular properties, including quantum-chemical properties.*

Tips:

- MoLFormer uses linear attention from [Performers](https://arxiv.org/abs/2009.14794). This means the full attention matrix is not calculated by default and using the `output_attentions=True` option will result in significant performance loss. Furthermore, the encoder model does not support full attention masks (e.g., causal attention) -- the attention mask must be the same for each token in the sentence (molecule).
- The config contains an option to enable deterministic generalized random feature weights in eval mode (`deterministic_eval=True`). This is disabled by default according to the original implementation.
- The original MoLFormer implementation does not use tied embedding weights.
- Only encoder models are supported.

This model was contributed by [IBM](https://huggingface.co/ibm).
The original code can be found [here](https://github.com/IBM/molformer).


## MolformerConfig

[[autodoc]] MolformerConfig

## MolformerTokenizer

[[autodoc]] MolformerTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## MolformerTokenizerFast

[[autodoc]] MolformerTokenizerFast
    - build_inputs_with_special_tokens

## MolformerModel

[[autodoc]] MolformerModel
    - forward

## MolformerForMaskedLM

[[autodoc]] MolformerForMaskedLM
    - forward

## MolformerForSequenceClassification

[[autodoc]] MolformerForSequenceClassification
    - forward
