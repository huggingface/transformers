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

# ModernBert

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=modernbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-modernbert-blueviolet">
</a>
<!-- <a href="">
<img alt="Paper page" src="https://img.shields.io/badge/Paper%20page--green">
</a> -->
</div>

## Overview

The ModernBert model was proposed in []() by ...

It builds on BERT and modifies ...

The abstract from the paper is the following:

**

The original code can be found [here]().

## Usage tips

- This implementation is similar to [`BertModel`] ...
- ModernBert doesn't have `token_type_ids`, so you don't need to indicate which token belongs to which segment. 
- ModernBert is similar to BERT but with ...

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RoBERTa. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="sentence-similarity"/>

...

<PipelineTag pipeline="fill-mask"/>

- [Masked language modeling task guide](../tasks/masked_language_modeling)


## ModernBertConfig

[[autodoc]] ModernBertConfig

<frameworkcontent>
<pt>

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

</pt>
</frameworkcontent>
