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

# Kyutai Speech-To-Text 
## Overview

Kyutai STT is a speech-to-text model architecture based on the [Mimi codec](https://huggingface.co/docs/transformers/en/model_doc/mimi), which encodes audio into discrete tokens in a streaming fashion, and a [Moshi-like](https://huggingface.co/docs/transformers/en/model_doc/moshi) autoregressive decoder. Kyutai’s lab has released two model checkpoints:
- [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr): a 1B-parameter model capable of transcribing both English and French
- [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en): a 2.6B-parameter model focused solely on English, optimized for maximum transcription accuracy

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/kyutai_stt.png"/>
</div>

## Usage Tips

### Inference

```python
import torch
from datasets import load_dataset, Audio
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

# 1. load the model and the processor
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "kyutai/stt-2.6b-en-trfs"

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype="auto")

# 2. load audio samples
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# 3. prepare the model inputs
inputs = processor(
    ds[0]["audio"]["array"],
)
inputs.to(torch_device)

# 4. infer the model
output_tokens = model.generate(**inputs)

# 5. decode the generated tokens
print(processor.batch_decode(output_tokens, skip_special_tokens=True))
```

### Batched Inference

```python
import torch
from datasets import load_dataset, Audio
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

# 1. load the model and the processor
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "kyutai/stt-2.6b-en-trfs"

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype="auto")

# 2. load audio samples
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# 3. prepare the model inputs
audio_arrays = [ds[i]["audio"]["array"] for i in range(4)]
inputs = processor(audio_arrays, return_tensors="pt", padding=True)
inputs = inputs.to(torch_device)

# 4. infer the model
output_tokens = model.generate(**inputs)

# 5. decode the generated tokens
decoded_outputs = processor.batch_decode(output_tokens, skip_special_tokens=True)
for output in decoded_outputs:
    print(output)
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).
The original code can be found [here](https://github.com/kyutai-labs/moshi).


## KyutaiSpeechToTextConfig

[[autodoc]] KyutaiSpeechToTextConfig

## KyutaiSpeechToTextProcessor

[[autodoc]] KyutaiSpeechToTextProcessor
    - __call__

## KyutaiSpeechToTextFeatureExtractor

[[autodoc]] KyutaiSpeechToTextFeatureExtractor

## KyutaiSpeechToTextForConditionalGeneration

[[autodoc]] KyutaiSpeechToTextForConditionalGeneration
    - forward
    - generate

## KyutaiSpeechToTextModel

[[autodoc]] KyutaiSpeechToTextModel
