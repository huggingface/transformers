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
*This model was released on 2021-09-14 and added to Hugging Face Transformers on 2021-10-15 and contributed by [anton-l](https://huggingface.co/anton-l).*

# SEW-D

[SEW-D](https://huggingface.co/papers/2109.06870) focuses on optimizing architecture designs for automatic speech recognition (ASR) by improving both performance and efficiency in pre-trained models. It introduces SEW (Squeezed and Efficient Wav2vec) with disentangled attention, achieving a 1.9x inference speedup and a 13.5% relative reduction in word error rate compared to wav2vec 2.0 under a 100h-960h semi-supervised setup on LibriSpeech. SEW also reduces word error rate by 25-50% across different model sizes while maintaining similar inference times.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="asapp/sew-d-tiny-100k", dtype="auto")
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

processor = AutoProcessor.from_pretrained("asapp/sew-d-tiny-100k")
model = AutoModelForCTC.from_pretrained("asapp/sew-d-tiny-100k", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## SEWDConfig

[[autodoc]] SEWDConfig

## SEWDModel

[[autodoc]] SEWDModel
    - forward

## SEWDForCTC

[[autodoc]] SEWDForCTC
    - forward

## SEWDForSequenceClassification

[[autodoc]] SEWDForSequenceClassification
    - forward

