<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Table Transformer

## Overview

The Table Transformer model was proposed in [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061) by
Brandon Smock, Rohith Pesala, Robin Abraham. The authors introduce a new dataset, PubTables-1M, to benchmark progress in table extraction from unstructured documents,
as well as table structure recognition and functional analysis. The authors train 2 [DETR](detr) models, one for table detection and one for table structure recognition, dubbed Table Transformers.

The abstract from the paper is the following:

*Recently, significant progress has been made applying machine learning to the problem of table structure inference and extraction from unstructured documents.
However, one of the greatest challenges remains the creation of datasets with complete, unambiguous ground truth at scale. To address this, we develop a new, more
comprehensive dataset for table extraction, called PubTables-1M. PubTables-1M contains nearly one million tables from scientific articles, supports multiple input
modalities, and contains detailed header and location information for table structures, making it useful for a wide variety of modeling approaches. It also addresses a significant
source of ground truth inconsistency observed in prior datasets called oversegmentation, using a novel canonicalization procedure. We demonstrate that these improvements lead to a
significant increase in training performance and a more reliable estimate of model performance at evaluation for table structure recognition. Further, we show that transformer-based
object detection models trained on PubTables-1M produce excellent results for all three tasks of detection, structure recognition, and functional analysis without the need for any
special customization for these tasks.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/table_transformer_architecture.jpeg"
alt="drawing" width="600"/>

<small> Table detection and table structure recognition clarified. Taken from the <a href="https://arxiv.org/abs/2110.00061">original paper</a>. </small>

The authors released 2 models, one for [table detection](https://huggingface.co/microsoft/table-transformer-detection) in 
documents, one for [table structure recognition](https://huggingface.co/microsoft/table-transformer-structure-recognition) 
(the task of recognizing the individual rows, columns etc. in a table).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be
found [here](https://github.com/microsoft/table-transformer).

## Resources

<PipelineTag pipeline="object-detection"/>

- A demo notebook for the Table Transformer can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer).
- It turns out padding of images is quite important for detection. An interesting Github thread with replies from the authors can be found [here](https://github.com/microsoft/table-transformer/issues/68).

## TableTransformerConfig

[[autodoc]] TableTransformerConfig

## TableTransformerModel

[[autodoc]] TableTransformerModel
    - forward

## TableTransformerForObjectDetection

[[autodoc]] TableTransformerForObjectDetection
    - forward
