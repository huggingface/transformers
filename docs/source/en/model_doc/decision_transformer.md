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

# Decision Transformer

## Overview

The Decision Transformer model was proposed in [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)  
by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.

The abstract from the paper is the following:

*We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. 
This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances
 in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that 
 casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or 
 compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked 
 Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our 
 Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, 
 Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on 
 Atari, OpenAI Gym, and Key-to-Door tasks.*

Tips:

This version of the model is for tasks where the state is a vector, image-based states will come soon.

This model was contributed by [edbeeching](https://huggingface.co/edbeeching). The original code can be found [here](https://github.com/kzl/decision-transformer).

## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig


## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model
    - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel
    - forward
