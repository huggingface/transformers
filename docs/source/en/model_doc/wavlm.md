<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-10-26 and added to Hugging Face Transformers on 2021-12-16 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# WavLM

[WavLM](https://huggingface.co/papers/2110.13900) proposes a new pre-trained model built on the HuBERT framework, designed to address full-stack speech processing tasks. It incorporates gated relative position bias in the Transformer structure to enhance recognition capabilities and uses an utterance mixing training strategy to improve speaker discrimination. By scaling the training dataset to 94k hours, WavLM Large achieves top performance on the SUPERB benchmark and significantly improves various speech processing tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="microsoft/wavlm-base", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCTC

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("microsoft/wavlm-base")
model = AutoModelForCTC.from_pretrained("microsoft/wavlm-base", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel
    - forward

## WavLMForCTC

[[autodoc]] WavLMForCTC
    - forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification
    - forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification
    - forward

## WavLMForXVector

[[autodoc]] WavLMForXVector
    - forward

