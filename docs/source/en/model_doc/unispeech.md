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
*This model was released on 2021-01-19 and added to Hugging Face Transformers on 2021-10-26 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# UniSpeech

[UniSpeech](https://huggingface.co/papers/2101.07597) proposes a unified pre-training method that leverages both labeled and unlabeled data to learn speech representations. It employs multi-task learning, combining supervised phonetic CTC learning with phonetically-aware contrastive self-supervised learning. This approach enhances the capture of phonetic structures and improves generalization across languages and domains. Evaluations on the CommonVoice corpus show UniSpeech outperforms self-supervised pretraining and supervised transfer learning in speech recognition, reducing phone error rates by up to 13.4% and 17.8% respectively. Additionally, UniSpeech demonstrates strong transferability in domain-shift speech recognition tasks, achieving a 6% reduction in word error rates compared to previous methods.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="microsoft/unispeech-large-1500h-cv", dtype="auto")
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

processor = AutoProcessor.from_pretrained("microsoft/unispeech-large-1500h-cv")
model = AutoModelForCTC.from_pretrained("microsoft/unispeech-large-1500h-cv", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## UniSpeechConfig

[[autodoc]] UniSpeechConfig

## UniSpeech specific outputs

[[autodoc]] models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput

## UniSpeechModel

[[autodoc]] UniSpeechModel
    - forward

## UniSpeechForCTC

[[autodoc]] UniSpeechForCTC
    - forward

## UniSpeechForSequenceClassification

[[autodoc]] UniSpeechForSequenceClassification
    - forward

## UniSpeechForPreTraining

[[autodoc]] UniSpeechForPreTraining
    - forward

