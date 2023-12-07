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

# ChatGLM

## Overview

The initial idea of ChatGLM model was proposed in [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) by Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang, and paper [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414) by Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, Jie Tang.

It has been iterating for 3 versions, from ChatGLM and ChatGLM2 to the most recent ChatGLM3.



The abstract from the paper is the following:

*There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and encoder-decoder models (e.g., T5). However, none of the pretraining frameworks performs the best for all tasks of three main categories including natural language understanding (NLU), unconditional generation, and conditional generation. We propose a General Language Model (GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over BERT and T5 on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and unconditional generation, GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25× parameters of BERTLarge, demonstrating its generalizability to different downstream tasks*

Tips:

- TODO conversion tips if needed

This model was contributed by [THUDM](<https://huggingface.co/THUDM). The most recent code can be found [here](https://github.com/thudm/chatglm3).


## ChatGlmConfig

[[autodoc]] ChatGlmConfig

## ChatGlmModel

[[autodoc]] ChatGlmModel
    - forward

## ChatGlmForCausalLM

[[autodoc]] ChatGlmForCausalLM
    - forward

## ChatGlmForSequenceClassification

[[autodoc]] ChatGlmForSequenceClassification
    - forward
