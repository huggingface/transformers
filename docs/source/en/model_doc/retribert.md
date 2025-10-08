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
*This model was released on 2020-06-12 and added to Hugging Face Transformers on 2023-06-20 and contributed by [yjernite](https://huggingface.co/yjernite).*

> [!WARNING]
> This model is in maintenance mode only, so we won't accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: `pip install -U transformers==4.30.0`.

# RetriBERT

[RetriBERT](https://yjernite.github.io/lfqa.html) is a compact model designed for dense semantic indexing, utilizing either a single or a pair of BERT encoders with a reduced-dimensional projection layer. This architecture enables efficient retrieval of relevant passages by encoding text into dense vectors. It was developed to facilitate open-domain long-form question answering (LFQA) tasks, particularly when training data is limited. By leveraging the ELI5 dataset, RetriBERT demonstrates how dense retrieval systems can be trained without extensive supervision or task-specific pretraining, making such models more accessible.

## RetriBertConfig

[[autodoc]] RetriBertConfig

## RetriBertTokenizer

[[autodoc]] RetriBertTokenizer

## RetriBertTokenizerFast

[[autodoc]] RetriBertTokenizerFast

## RetriBertModel

[[autodoc]] RetriBertModel
    - forward
