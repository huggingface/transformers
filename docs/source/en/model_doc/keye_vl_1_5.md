<!--Copyright 2025 The Kuai Keye Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Keye-VL-1.5
[Keye-VL-1.5](https://huggingface.co/papers/2509.01563) is a multimodal foundation model developed based on the Keye-VL-Preview architecture, featuring 8 billion parameters and demonstrating outstanding performance in short video understanding tasks. Compared to its predecessor, this model has been trained on a more refined dataset and has been enhanced to support the Slow-Fast architecture as well as 128K long-context processing. By incorporating a comprehensive pre-training and post-training pipeline that integrates reinforcement learning and alignment techniques, the model not only improves performance in specialized scenarios but also maintains robust general-purpose vision-language capabilities.

The abstract from the paper is the following:

*In recent years, the development of Large Language Models (LLMs) has significantly advanced, extending their capabilities to multimodal tasks through Multimodal Large Language Models (MLLMs). However, video understanding remains a challenging area due to the dynamic and information-dense nature of videos. Existing models struggle with the trade-off between spatial resolution and temporal coverage when processing video content. We present Keye-VL-1.5, which addresses fundamental challenges in video comprehension through three key innovations. First, we introduce a novel SlowFast video encoding strategy that dynamically allocates computational resources based on inter-frame similarity, processing key frames with significant visual changes at higher resolution (Slow pathway) while handling relatively static frames with increased temporal coverage at lower resolution (Fast pathway). Second, we implement a progressive four-stage pre-training methodology that systematically extends the model’s context length from 8K to 128K tokens, enabling processing of longer videos and more complex visual content. Third, we develop a comprehensive post-training pipeline focusing on reasoning enhancement and human preference alignment, incorporating a 5-step chain-of-thought data construction process, iterative GSPO-based reinforcement learning with progressive prompt hinting for difficult cases, and alignment training. Through extensive evaluation on public benchmarks and rigorous internal human assessment, Keye-VL-1.5 demonstrates significant improvements over existing models, particularly excelling in video understanding tasks while maintaining competitive performance on general multimodal benchmarks.*

You can find the original Keye-VL-1.5 checkpoint under the [Keye-VL-1.5](https://huggingface.co/Kwai-Keye/Keye-VL-1_5-8B).
## Fusion Processor Op
We have launched a performance-optimized version for the video preprocessing pipeline of the `keye-vl-1.5` model, significantly enhancing processing efficiency through the fusion and parallelization of key operators. Specific optimizations include the fusion of operations such as resizing `numpy.ndarray`, `RGB-to-HSV` conversion, scaling, and normalization, along with the implementation of `SIMD` parallel computing using the `SSE2` instruction set.

We conducted systematic testing on the `VideoMME` dataset (paper link: https://arxiv.org/pdf/2405.21075). Under the condition of maintaining a video frame rate of `fps=2`, the extracted video frames underwent input preprocessing via the `KeyeVL1_5Processor` class. Experimental results demonstrate that the `C++` implementation achieved more than a `7`-fold improvement in processing efficiency compared to the original Python version. When videos were processed in batches with multiple frames (`64` frames), the processing speed further increased, reaching a speedup ratio of `16` times, while benchmark scores remained stable with no performance degradation.

The performance comparison across different versions is shown in the table below:

| Version        | fps=2 (Avg Process Time / Score) | nframes=64 (Avg Process Time / Score) |
|----------------|----------------------|--------------------------|
| Operator Fusion     | 0.4178s / 74.4       | 0.0545s / 74.1           |
| Original         | 2.9676s / 74.7       | 0.9097s / 73.9           |
| Speedup Ratio     | 7.10x                | 16.69x                   |

The experimental data clearly indicate that the `C++` implementation not only achieves significant improvements in processing efficiency but also maintains the stability of model output quality.


For practical use, it can be compiled with the following command:
```bash
g++ -std=c++17 -mavx2 -O3 -fopenmp -shared -fPIC -o processing_keye_vl_1_5.so processing_keye_vl_1_5.cpp
```

You can adjust the parallelism of operator execution at runtime by setting the `KEYE_VL_UTILS_PARALLEL_NUM` environment variable, which has a default value of `64`. The setting method is as follows:
```bash
export KEYE_VL_UTILS_PARALLEL_NUM=64
```

If you need to disable this fused operator, you can set the environment variable `ENABLE_FUSION_PROCESSOR_OP=0`. This option is enabled by default (`1`).

However, the overall pipeline of this performance-optimized version is tightly coupled with the processing logic of `keye-vl-1.5`. If you intend to implement custom processor operations, it is advisable to avoid using this optimized version.


# Keye-VL-Preview
[Keye-VL-Preview](https://huggingface.co/papers/2507.01949) is an 8-billion-parameter multimodal foundation model, excels in short-video understanding while maintaining robust general-purpose vision-language abilities through a comprehensive pre- and post-training process, including reinforcement learning and alignment.

The abstract from the paper is the following:

*While Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities on static images, they often fall short in comprehending dynamic, information-dense short-form videos, a dominant medium in today’s digital landscape. To bridge this gap, we introduce Kwai Keye-VL, an 8-billion-parameter multimodal foundation model engineered for leading-edge performance in short-video understanding while maintaining robust general-purpose vision-language abilities. The development of Keye-VL rests on two core pillars: a massive, high-quality dataset exceeding 600 billion tokens with a strong emphasis on video, and an innovative training recipe. This recipe features a four-stage pre-training process for solid vision-language alignment, followed by a meticulous two-phase post-training process. The first post-training stage enhances foundational capabilities like instruction following, while the second phase focuses on stimulating advanced reasoning. In this second phase, a key innovation is our five-mode “cold-start” data mixture, which includes “thinking”, “non-thinking”, “auto-think”, “think with image”, and high-quality video data. This mixture teaches the model to decide when and how to reason. Subsequent reinforcement learning (RL) and alignment steps further enhance these reasoning capabilities and correct abnormal model behaviors, such as repetitive outputs. To validate our approach, we conduct extensive evaluations, showing that Keye-VL achieves state-of-the-art results on public video benchmarks and remains highly competitive on general image-based tasks (Figure 1). Furthermore, we develop and release the KC-MMBench, a new benchmark tailored for real-world short-video scenarios, where Keye-VL shows a significant advantage. Comprehensive human evaluations also confirm that our model provides a superior user experience compared to other leading models of a similar scale. This paper details the architecture, data construction strategy, and training methodology of Keye-VL, offering valuable insights for building the next generation of MLLMs for the video era.*

You can find the original Keye-VL-Preview checkpoint under the [Keye-VL-Preview](https://huggingface.co/Kwai-Keye/Keye-VL-8B-Preview).

The example below demonstrates how to generate text based on an image with the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
# pip install --upgrade keye-vl-utils==1.5.2 -i https://pypi.org/simple
from keye_vl_utils import process_vision_info

model = AutoModel.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B", trust_remote_code=True)
url = "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "image": url,
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }

]
# Since we support the slow-fast architecture and keye-vl-utils has additional return parameters, 
# we did not adopt the combined form of: 
# inputs = processor.apply_chat_template( messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt" )

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **mm_processor_kwargs
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```
</hfoption>

### Notes

- Use Keye-VL-1.5 for video inputs by setting `"type": "video"` as shown below.
    ```python

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/path/to/video.mp4"},
                {"type": "text", "text": "What happened in the video?"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text)
    ```
- Use Keye-VL-1.5 for a mixed batch of inputs (images, videos, text).
    ```python
    import torch
    from transformers import AutoModel, AutoProcessor
    # pip install --upgrade keye-vl-utils==1.5.2 -i https://pypi.org/simple
    from keye_vl_utils import process_vision_info
    
    model = AutoModel.from_pretrained(
        "Kwai-Keye/Keye-VL-1_5-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B", trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": "Hello, how are you?"}
            ]
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking. How can I assist you today?"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you describe these images and video?"}, 
                {"type": "image"}, 
                {"type": "image"}, 
                {"type": "video"}, 
                {"type": "text", "text": "These are from my vacation."}
            ]
        },
        {
            "role": "assistant",
            "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
        },
        {
            "role": "user",
            "content": "It was a trip to the mountains. Can you see the details in the images and video?"
        }
    ]
    
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Expected output: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
    
    ```

## KeyeVL1_5Config

[[autodoc]] KeyeVL1_5Config

## KeyeVL1_5Processor

[[autodoc]] KeyeVL1_5Processor

## KeyeVL1_5ForConditionalGeneration

[[autodoc]] KeyeVL1_5ForConditionalGeneration

## KeyeVL1_5Model

[[autodoc]] KeyeVL1_5Model

## KeyeVL1_5TextConfig

[[autodoc]] KeyeVL1_5TextConfig

## KeyeVL1_5TextModel

[[autodoc]] KeyeVL1_5TextModel

## KeyeVL1_5VisionModel

[[autodoc]] KeyeVL1_5VisionModel

## KeyeVL1_5ImageProcessor

[[autodoc]] KeyeVL1_5ImageProcessor
