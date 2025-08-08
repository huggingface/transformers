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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# CLAP

[CLAP (Contrastive Language-Audio Pretraining)](https://huggingface.co/papers/2211.06687) is a multimodal model that combines audio data with natural language descriptions through contrastive learning.

It incorporates feature fusion and keyword-to-caption augmentation to process variable-length audio inputs and to improve performance. CLAP doesn't require task-specific training data and can learn meaningful audio representations through natural language.

You can find all the original CLAP checkpoints under the [CLAP](https://huggingface.co/collections/laion/clap-contrastive-language-audio-pretraining-65415c0b18373b607262a490) collection.

> [!TIP]
> This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).
>
> Click on the CLAP models in the right sidebar for more examples of how to apply CLAP to different audio retrieval and classification tasks.

The example below demonstrates how to extract text embeddings with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("laion/clap-htsat-unfused", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

texts = ["the sound of a cat", "the sound of a dog", "music playing"]

inputs = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

print(f"Text embeddings shape: {text_features.shape}")
print(f"Text embeddings: {text_features}")
```

</hfoption>
</hfoptions>

## ClapConfig

[[autodoc]] ClapConfig
    - from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward
