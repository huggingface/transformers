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
*This model was released on 2021-10-30 and added to Hugging Face Transformers on 2023-06-20 and contributed by [cwkeam](https://huggingface.co/cwkeam).*

> [!WARNING]
> This model is in maintenance mode only, so we won’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: pip install -U transformers==4.30.0.

# M-CTC-T

[M-CTC-T](https://huggingface.co/papers/2111.00161) is a 1B-param transformer encoder with a CTC head for 8065 character labels and a language identification head for 60 languages. Trained on Common Voice and VoxPopuli datasets, it uses pseudo-labeling to enhance performance across multiple languages, including low-resource ones. The model processes Mel filterbank features from 16Khz audio signals and demonstrates improved performance and transferability to LibriSpeech.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="speechbrain/m-ctc-t-large", dtype="auto")
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

processor = AutoProcessor.from_pretrained("speechbrain/m-ctc-t-large")
model = AutoModelForCTC.from_pretrained("speechbrain/m-ctc-t-large", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## MCTCTConfig

[[autodoc]] MCTCTConfig

## MCTCTFeatureExtractor

[[autodoc]] MCTCTFeatureExtractor
    - __call__

## MCTCTProcessor

[[autodoc]] MCTCTProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## MCTCTModel

[[autodoc]] MCTCTModel
    - forward

## MCTCTForCTC

[[autodoc]] MCTCTForCTC
    - forward

