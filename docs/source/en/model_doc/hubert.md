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

[HuBERT](https://huggingface.co/papers/2106.07447) is a self-supervised speech model that learns to understand audio by first clustering audio patterns (called "hidden units") and then predicting those. Think of it like fill-in-the-blank for audio. This makes it really good at learning from raw speech without needing lots of transcriptions, and it performs well on tasks like automatic speech recognition.

You can find all the original HuBERT checkpoints under the [HuBERT](https://huggingface.co/collections/facebook/hubert-651fca95d57549832161e6b6) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

The example below demonstrates how to perform Automatic Speech Recognition (ASR) with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

asr = pipeline(
    task="automatic-speech-recognition",
    model="facebook/hubert-large-ls960-ft",
    torch_dtype=torch.float16,
    device=0
)

result = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(result["text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, HubertForCTC
from datasets import load_dataset

# Load and sort dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")

# Process audio input and run inference
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# Transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
```

</hfoption>

<!-- Not Applicable -->
<hfoption id="transformers-cli">
</hfoption>

</hfoptions>

## Flash Attention 2

Flash Attention 2 is a highly optimized version of the model that can significantly reduce inference time and memory usage on compatible hardware.

### Installation

Check to see if your hardware supports Flash Attention 2 by reviewing the [official compatibility list](https://github.com/Dao-AILab/flash-attention#installation-and-features). 
If your hardware is compatible, install the latest version with:

```bash
pip install -U flash-attn --no-build-isolation
```

If your hardware is not compatible, you can still use attention kernel optimizations through [Better Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

### Usage

You can enable Flash Attention 2 by setting the attn_implementation argument to "flash_attention_2" when loading the model:

```python
from transformers import HubertModel
import torch

model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
).to("cuda")
```

> [!NOTE]
> Flash Attention 2 currently only works with PyTorch models and requires CUDA-compatible hardware with Ampere or newer GPUs.

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. 
Refer to the [Quantization](https://huggingface.co/docs/transformers/en/quantization/overview) overview for more available quantization backends.

The example below uses [PyTorch Dynamic Quantization](https://pytorch.org/docs/stable/quantization.html#dynamic-quantization) to only quantize the weights to 8-bit integers (int8).

```python
import torch
from transformers import HubertForCTC

# Load the pretrained model
model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")

# Apply dynamic quantization to Linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model,              # the original model
    {torch.nn.Linear},  # specify layers to quantize
    dtype=torch.qint8   # 8-bit integer weights
)

# Save or use the quantized model
quantized_model.eval()
```

## Notes

- HuBERT models expect raw audio input as a 1D float array, sampled at 16kHz.
- These models are typically fine-tuned using CTC (Connectionist Temporal Classification), 
  so the output must be decoded using a tokenizer like [`Wav2Vec2CTCTokenizer`] or via [`AutoProcessor`], which wraps all preprocessing steps.
- If you want to use a `head_mask`, use the model with `attn_implementation="eager"`:
  ```python
  model = HubertModel.from_pretrained("facebook/hubert-base-ls960", attn_implementation="eager")
  ```

## Resources

- [Audio classification task guide](https://huggingface.co/docs/transformers/main/en/tasks/audio_classification)
- [Automatic speech recognition task guide](https://huggingface.co/docs/transformers/main/en/tasks/asr)
- [HuBERT research paper](https://arxiv.org/abs/2106.07447)
- [Original HuBERT model on Hugging Face Hub](https://huggingface.co/facebook/hubert-base-ls960)

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
