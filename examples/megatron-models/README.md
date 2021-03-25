<!---
# ##############################################################################################
# 
# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# ##############################################################################################
-->

# How to run Megatron BERT and GPT2 using Transformers

## Get the checkpoints from the NVIDIA GPU Cloud 

The first step is to create a directory called `models` from the `examples/megatron-models` folder.

```
mkdir models
```

You can download the checkpoints from the NVIDIA GPU Cloud (NGC). For that you
have to [sign up](https://ngc.nvidia.com/signup) for and setup the NVIDIA GPU
Cloud (NGC) Registry CLI.  Further documentation for downloading models can be
found in the [NGC
documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

### BERT 345M cased

```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O models/megatron_bert_345m_v0_1_cased.zip
```

### BERT 345M uncased

```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O models/megatron_bert_345m_v0_1_uncased.zip
```

### GPT2 345M 

```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O models/megatron_gpt2_345m_v0_0.zip
```

## Converting the checkpoints

In order to be loaded into `Transformers`, the checkpoints have to be converted. You should run the following
commands for that purpose.

For the conversion, we use scripts stored in
`src/transformers/models/megatron_bert` and
`src/transformers/models/megatron_gpt2`. We define the relative path as:

```
export PATH_TO_TRANSFORMERS=../../src/transformers
```

### BERT 345M cased

```
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py models/megatron_bert_345m_v0_1_cased.zip
```

### BERT 345M uncased

```
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py models/megatron_bert_345m_v0_1_uncased.zip
```

### GPT2 345M 

```
python3 $PATH_TO_TRANSFORMERS/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py models/megatron_gpt2_345m_v0_0.zip
```

## Running the samples

For BERT, we created a simple example that runs two tasks using the Megatron BERT checkpoints using
the Transformers API. The first task is `MegatronBERTForMaskedLM` and the second one is 
`MegatronBERTForNextSentencePrediction`.

### Masked LM

```
python3 ./run_bert.py --masked-lm ./models/megatron_bert_345m_v0_1_cased
python3 ./run_bert.py --masked-lm ./models/megatron_bert_345m_v0_1_uncased
```

### Next sentence prediction

```
python3 ./run_bert.py ./models/megatron_bert_345m_v0_1_cased
python3 ./run_bert.py ./models/megatron_bert_345m_v0_1_uncased
```

### Text generation

For GPT2, we created a simple for text generation.

```
python3 ./run_gpt2.py models/megatron_gpt2_345m_v0_0
```

