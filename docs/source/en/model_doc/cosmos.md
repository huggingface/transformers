<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Cosmos

## Overview

The Cosmos model was proposed in [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575) by Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, Daniel Dworakowski, Jiaojiao Fan, Michele Fenzi, Francesco Ferroni, Sanja Fidler, Dieter Fox, Songwei Ge, Yunhao Ge, Jinwei Gu, Siddharth Gururani, Ethan He, Jiahui Huang, Jacob Huffman, Pooya Jannaty, Jingyi Jin, Seung Wook Kim, Gergely Klár, Grace Lam, Shiyi Lan, Laura Leal-Taixe, Anqi Li, Zhaoshuo Li, Chen-Hsuan Lin, Tsung-Yi Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Arsalan Mousavian, Seungjun Nah, Sriharsha Niverty, David Page, Despoina Paschalidou, Zeeshan Patel, Lindsey Pavao, Morteza Ramezanali, Fitsum Reda, Xiaowei Ren, Vasanth Rao Naik Sabavat, Ed Schmerling, Stella Shi, Bartosz Stefaniak, Shitao Tang, Lyne Tchapmi, Przemek Tredak, Wei-Cheng Tseng, Jibin Varghese, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Xinyue Wei, Jay Zhangjie Wu, Jiashu Xu, Wei Yang, Lin Yen-Chen, Xiaohui Zeng, Yu Zeng, Jing Zhang, Qinsheng Zhang, Yuxuan Zhang, Qingqing Zhao, Artur Zolkowski.


The abstract from the paper is the following:

*Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make our platform open-source and our models open-weight with permissive licenses available via this https URL.*

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/NVIDIA/Cosmos/tree/main).


## Usage examples

Cosmos can generate by conditioning either on video/image or text+video/image. The video used to condition has to be exactly 9 frames in length, while the image is treated as a single frame video.

Below is an example of generating by conditioning on video only.

```python
import torch
import imageio
from transformers.image_utils import load_video
from transformers import CosmosProcessor, CosmosForConditionalGeneration

model_id = "NVIDIA/Cosmos-4B-hf"
processor = CosmosProcessor.from_pretrained(model_id)

model = CosmosForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Generate from last 9 frames of the video
video, _ = load_video("cosmos1/models/autoregressive/assets/v1p0/input.mp4", backend="decord")[-9:]
inputs = proc(videos=video, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

out = model.generate(**inputs, max_new_tokens=7680)

# Decode the video and save. 
video_decoded = model.model.decode_video_tokens(out)
video_decoded = video_decoded.permute(0, 2, 1, 3, 4).float()
video_processed = proc.postprocess([video_decoded[0]], return_tensors="np")
imageio.mimsave("generated_video.mp4", video_processed['pixel_values'].squeeze(0), fps=25)

```

To condition on text input as well, we just pass it along to the processor. The rest is same as in video conditioning.

```python
import torch
import imageio
from transformers.image_utils import load_video
from transformers import CosmosProcessor, CosmosForConditionalGeneration

model_id = "NVIDIA/Cosmos-5B-hf"
processor = CosmosProcessor.from_pretrained(model_id)

model = CosmosForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Generate from last 9 frames of the video
video, _ = load_video("cosmos1/models/autoregressive/assets/v1p0/input.mp4", backend="decord")[-9:]
text = "A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions."
inputs = proc(videos=video, text=text, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

out = model.generate(**inputs, max_new_tokens=7680)

# Remove the first token which is `BOS`. Decode the video and save. 
video_decoded = model.model.decode_video_tokens(out[:, 1:])
video_decoded = video_decoded.permute(0, 2, 1, 3, 4).float()
video_processed = proc.postprocess([video_decoded[0]], return_tensors="np")
imageio.mimsave("generated_video.mp4", video_processed['pixel_values'].squeeze(0), fps=25)

```

## CosmosVideoProcessor

[[autodoc]] CosmosVideoProcessor

## CosmosProcessor

[[autodoc]] CosmosProcessor

## CosmosConfig

[[autodoc]] CosmosConfig

## CosmosVQVAEConfig

[[autodoc]] CosmosVQVAEConfig

## CosmosTextConfig

[[autodoc]] CosmosTextConfig

## CosmosVQVAE

[[autodoc]] CosmosVQVAE
    - forward

## CosmosTextModel

[[autodoc]] CosmosTextModel
    - forward

## CosmosTextPreTrainedModel

[[autodoc]] CosmosTextPreTrainedModel
    - forward

## CosmosPreTrainedModel

[[autodoc]] CosmosPreTrainedModel
    - forward

## CosmosModel

[[autodoc]] CosmosModel
    - forward

## CosmosForConditionalGeneration

[[autodoc]] CosmosForConditionalGeneration
    - forward
