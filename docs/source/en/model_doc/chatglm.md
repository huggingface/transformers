<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ChatGLM

## Overview

The initial idea of ChatGLM model was proposed in [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) by Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang, and paper [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414) by Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, Jie Tang.

It has been iterating for 3 versions, from ChatGLM and ChatGLM2 to the most recent ChatGLM3.

This model was contributed by [THUDM](<https://huggingface.co/THUDM). The most recent code can be found [here](https://github.com/thudm/chatglm3).

## ChatGLMConfig

[[autodoc]] ChatGLMConfig


## ChatGLMTokenizer

[[autodoc]] ChatGLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## ChatGLMForSequenceClassification

[[autodoc]] transformers.ChatGLMForSequenceClassification
    - all


## ChatGLMForConditionalGeneration

[[autodoc]] transformers.ChatGLMForConditionalGeneration
    - all


## ChatGLMModel

[[autodoc]] transformers.ChatGLMModel
    - all


## ChatGLMPreTrainedModel

[[autodoc]] transformers.ChatGLMPreTrainedModel
    - all
