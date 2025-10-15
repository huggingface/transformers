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
*This model was released on 2021-06-14 and added to Hugging Face Transformers on 2021-06-16 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Hubert

[Hubert](https://huggingface.co/papers/2106.07447) addresses challenges in self-supervised speech representation learning by introducing Hidden-Unit BERT (HuBERT). HuBERT employs an offline clustering step to generate aligned target labels for a BERT-like prediction loss, focusing on masked regions to learn a combined acoustic and language model. Using k-means clustering with two iterations, HuBERT matches or surpasses wav2vec 2.0 on Librispeech and Libri-light benchmarks across various fine-tuning subsets. A 1B parameter HuBERT model achieves significant relative WER reductions on dev-other and test-other subsets.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/hubert-large-ls960-ft", dtype="auto")
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

processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModelForCTC.from_pretrained("facebook/hubert-base-ls960", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Usage tips

HuBERT models require raw audio as a 1D float array sampled at 16kHz. This format preserves the original audio signal without preprocessing.

## HubertConfig

[[autodoc]] HubertConfig

## HubertModel

[[autodoc]] HubertModel
    - forward

## HubertForCTC

[[autodoc]] HubertForCTC
    - forward

## HubertForSequenceClassification

[[autodoc]] HubertForSequenceClassification
    - forward

