<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-15.*


# Glmasr

## Overview

**GLM-ASR-Nano-2512** is a robust, open-source speech recognition model with **1.5B parameters**. Designed for
real-world complexity, it outperforms OpenAI Whisper V3 on multiple benchmarks while maintaining a compact size.

Key capabilities include:

* **Exceptional Dialect Support**
  Beyond standard Mandarin and English, the model is highly optimized for **Cantonese (粤语)** and other dialects,
  effectively bridging the gap in dialectal speech recognition.

* **Low-Volume Speech Robustness**
  Specifically trained for **"Whisper/Quiet Speech"** scenarios. It captures and accurately transcribes extremely
  low-volume audio that traditional models often miss.

* **SOTA Performance**
  Achieves the **lowest average error rate (4.10)** among comparable open-source models, showing significant advantages
  in Chinese benchmarks (Wenet Meeting, Aishell-1, etc..).

you can check the [model card](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for more details and our 
[github repo](https://github.com/zai-org/GLM-ASR).

## Usage examples

```python
import torch
from transformers import GlmasrForConditionalGeneration, AutoProcessor, pipeline
from datasets import load_dataset

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "zai-org/GLM-ASR-Nano-2512"
model = GlmasrForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

## GlmasrEncoderConfig

[[autodoc]] GlmasrEncoderConfig

## GlmasrConfig

[[autodoc]] GlmasrConfig

## GlmasrPreTrainedModel

[[autodoc]] GlmasrPreTrainedModel
    - forward

## GlmasrEncoder

[[autodoc]] GlmasrEncoder
    - forward

## GlmasrForConditionalGeneration

[[autodoc]] GlmasrForConditionalGeneration
    - forward