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

# RetriBERT

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The RetriBERT model was proposed in the blog post [Explain Anything Like I'm Five: A Model for Open Domain Long Form
Question Answering](https://yjernite.github.io/lfqa.html). RetriBERT is a small model that uses either a single or
pair of BERT encoders with lower-dimension projection for dense semantic indexing of text.

This model was contributed by [yjernite](https://huggingface.co/yjernite). Code to train and use the model can be
found [here](https://github.com/huggingface/transformers/tree/main/examples/research-projects/distillation).


## RetriBertConfig

[[autodoc]] RetriBertConfig

## RetriBertTokenizer

[[autodoc]] RetriBertTokenizer

## RetriBertTokenizerFast

[[autodoc]] RetriBertTokenizerFast

## RetriBertModel

[[autodoc]] RetriBertModel
    - forward
