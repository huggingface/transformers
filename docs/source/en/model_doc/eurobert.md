<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EuroBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=eurobert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-eurobert-blueviolet">
</a>

## Overview

EuroBERT is a multilingual encoder model based on a refreshed transformer architecture. It supports a mixture of European and widely spoken languages.

Check out all the available models [here](https://huggingface.co/EuroBERT).

## EuroBertConfig

[[autodoc]] EuroBertConfig

<frameworkcontent>
<pt>

## EuroBertModel

[[autodoc]] EuroBertModel
    - forward

## EuroBertForMaskedLM

[[autodoc]] EuroBertForMaskedLM
    - forward

## EuroBertForSequenceClassification

[[autodoc]] EuroBertForSequenceClassification
    - forward

</pt>
</frameworkcontent>