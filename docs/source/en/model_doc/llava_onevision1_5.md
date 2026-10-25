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
*This model was contributed to Hugging Face Transformers on 2026-10-22.*

# LLaVA-OneVision-1.5

LLaVA-OneVision-1.5 was introduced in [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://huggingface.co/papers/2509.23661).
It combines a native resolution vision encoder with a Qwen3 language backbone. The vision encoder adds a learned class token to each image, applies full attention, removes the class token, and merges spatial patches into visual features for the language model. The language backbone uses ordinary one dimensional rotary positions for the combined image and text sequence.

The implementation reuses the Qwen2.5-VL processor and supports the converted [4B checkpoint](https://huggingface.co/Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf). Use left padding for batched generation.

## Abstract

The [paper's abstract](https://arxiv.org/abs/2509.23661) describes an open framework for training vision-language models from scratch at a reduced cost. Its release includes a concept-balanced collection of 85 million pretraining examples, a curated instruction dataset containing 22 million examples, and a training system that uses offline parallel packing. The authors report a training budget of $16,000. In their evaluation, the 8B model exceeds Qwen2.5-VL-7B on 18 out of 27 benchmarks, while the 4B model exceeds Qwen2.5-VL-3B on all 27. The report also describes a lightweight reinforcement learning stage that improves multimodal reasoning through chain-of-thought responses.

## Usage

```python
from transformers import AutoProcessor, LlavaOnevision15ForConditionalGeneration

model_id = "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaOnevision15ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    {"type": "text", "text": "Describe the image."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64)
processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

## Export

The vision encoder can be exported with `DynamoExporter`, which prepares grid metadata before tracing. The shared vision attention export adapter requires equal patch counts per frame. The number of frames and their patch counts are specialized to the export inputs; export a new graph when these change.

## LlavaOnevision15Config

[[autodoc]] LlavaOnevision15Config

## LlavaOnevision15VisionConfig

[[autodoc]] LlavaOnevision15VisionConfig

## LlavaOnevision15VisionModel

[[autodoc]] LlavaOnevision15VisionModel
    - forward

## LlavaOnevision15Model

[[autodoc]] LlavaOnevision15Model
    - forward

## LlavaOnevision15ForConditionalGeneration

[[autodoc]] LlavaOnevision15ForConditionalGeneration
    - forward
