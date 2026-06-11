<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-11.*

# Step3p7

Step3p7 is a multimodal vision-language model from StepFun. It combines a Step3p7 text decoder with a StepRobotics vision encoder and a multimodal projector to generate text conditioned on text and image inputs.

The Transformers implementation supports loading original Step3p7 checkpoints with the standard [`AutoConfig`], [`AutoProcessor`], and [`AutoModelForCausalLM`] APIs.

## Usage example

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "stepfun-ai/Step-3.7-Flash"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Step3p7Config

[[autodoc]] Step3p7Config

## Step3p7TextConfig

[[autodoc]] Step3p7TextConfig

## StepRoboticsVisionEncoderConfig

[[autodoc]] StepRoboticsVisionEncoderConfig

## Step3VisionProcessor

[[autodoc]] Step3VisionProcessor
    - preprocess

## Step3VisionProcessorPil

[[autodoc]] Step3VisionProcessorPil
    - preprocess

## Step3VLProcessor

[[autodoc]] Step3VLProcessor
    - __call__

## Step3p7Model

[[autodoc]] Step3p7Model
    - forward
    - get_image_features

## Step3p7TextModel

[[autodoc]] Step3p7TextModel
    - forward

## Step3p7ForConditionalGeneration

[[autodoc]] Step3p7ForConditionalGeneration
    - forward
