<!--Copyright 2022 The HuggingFace Team, the DGL Team, Rensselaer Polytechnic Institute and IBM T. J. Watson Research Center. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License. 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EGT

## Overview

The EGT model was proposed in [Global Self-Attention as a Replacement for Graph Convolution](https://arxiv.org/abs/2108.03348)  by 
Md Shamim Hussain, Mohammed J. Zaki and Dharmashankar Subramanian. It is a Graph Transformer model, modified to allow computations on graphs instead of text sequences by generating embeddings and features of interest during preprocessing and collation, then using a modified attention.

The abstract from the paper is the following:

*We propose an extension to the transformer neural network architecture for general-purpose graph learning by adding a dedicated pathway for pairwise structural information, called edge channels. The resultant framework - which we call Edge-augmented Graph Transformer (EGT) - can directly accept, process and output structural information of arbitrary form, which is important for effective learning on graph-structured data. Our model exclusively uses global self-attention as an aggregation mechanism rather than static localized convolutional aggregation. This allows for unconstrained long-range dynamic interactions between nodes. Moreover, the edge channels allow the structural information to evolve from layer to layer, and prediction tasks on edges/links can be performed directly from the output embeddings of these channels. We verify the performance of EGT in a wide range of graph-learning experiments on benchmark datasets, in which it outperforms Convolutional/Message-Passing Graph Neural Networks. EGT sets a new state-of-the-art for the quantum-chemical regression task on the OGB-LSC PCQM4Mv2 dataset containing 3.8 million molecular graphs. Our findings indicate that global self-attention based aggregation can serve as a flexible, adaptive and effective replacement of graph convolution for general-purpose graph learning. Therefore, convolutional local neighborhood aggregation is not an essential inductive bias.*

Tips:

This model will not work well on large graphs (more than 100 nodes/edges), as it will make the memory explode.
You can reduce the batch size, or increase your RAM, but it will be hard to go above 700 nodes/edges.

This model does not use a tokenizer, but instead a special collator during training.

## TBD
This model was contributed by [Zhiteng](https://github.com/ZHITENGLI). The original code can be found [here](https://github.com/shamim-hussain/egt).

## EGTConfig

[[autodoc]] EGTConfig


## EGTModel

[[autodoc]] EGTModel
    - forward


## EGTForGraphClassification

[[autodoc]] EGTForGraphClassification
    - forward
