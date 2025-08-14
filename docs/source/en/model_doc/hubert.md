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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# HuBERT

[HuBERT](https://huggingface.co/papers/2106.07447) is a self-supervised speech model to cluster aligned target labels for BERT-like prediction loss and applying the prediction loss only over masked regions to force the model to learn both acoustic and language modeling over continuous inputs. It addresses the challenges of multiple sound units per utterance, no lexicon during pre-training, and variable-length sound units without explicit segmentation.

You can find all the original HuBERT checkpoints under the [HuBERT](https://huggingface.co/collections/facebook/hubert-651fca95d57549832161e6b6) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the HuBERT models in the right sidebar for more examples of how to apply HuBERT to different audio tasks.

The example below demonstrates how to automatically transcribe speech into text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="automatic-speech-recognition",
    model="facebook/hubert-large-ls960-ft",
    dtype=torch.float16,
    device=0
)

pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModelForCTC.from_pretrained("facebook/hubert-base-ls960", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
```

</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision.
Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```python
import torch
from transformers import AutoProcessor, AutoModelForCTC, BitsAndBytesConfig
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModelForCTC.from_pretrained("facebook/hubert-base-ls960", quantization_config=bnb_config, dtype=torch.float16, device_map="auto", attn_implementation="sdpa")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
```

## Notes

- HuBERT models expect raw audio input as a 1D float array sampled at 16kHz.
- If you want to use a `head_mask`, use the model with `attn_implementation="eager"`.
  ```python
  model = HubertModel.from_pretrained("facebook/hubert-base-ls960", attn_implementation="eager")
  ```

## HubertConfig

[[autodoc]] HubertConfig
    - all

<frameworkcontent>
<pt>

## HubertModel

[[autodoc]] HubertModel
    - forward

## HubertForCTC

[[autodoc]] HubertForCTC
    - forward

## HubertForSequenceClassification

[[autodoc]] HubertForSequenceClassification
    - forward

</pt>
<tf>

## TFHubertModel

[[autodoc]] TFHubertModel
    - call

## TFHubertForCTC

[[autodoc]] TFHubertForCTC
    - call

</tf>
</frameworkcontent>
