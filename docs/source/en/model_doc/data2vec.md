<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-02-07 and added to Hugging Face Transformers on 2022-03-01 and contributed by [edugp](https://huggingface.co/edugp) and [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Data2Vec

[Data2Vec](https://huggingface.co/papers/2202.03555) presents a unified framework for self-supervised learning applicable to speech, NLP, and computer vision. It employs a self-distillation setup using a standard Transformer architecture to predict latent representations of the full input data based on a masked view. Unlike traditional methods that predict modality-specific, local targets, data2vec focuses on predicting contextualized latent representations that encapsulate information from the entire input. Experiments across speech recognition, image classification, and natural language understanding benchmarks show state-of-the-art or competitive performance.

<hfoptions id="usage">
<hfoption id="Data2VecAudioForCTC">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Data2VecAudioForCTC

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/data2vec-audio-base-960h", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig

## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
    - forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification
    - forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC
    - forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification
    - forward

## Data2VecAudioForXVector

[[autodoc]] Data2VecAudioForXVector
    - forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel
    - forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM
    - forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM
    - forward

## Data2VecTextForSequenceClassification

[[autodoc]] Data2VecTextForSequenceClassification
    - forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice
    - forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification
    - forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering
    - forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel
    - forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification
    - forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation
    - forward

