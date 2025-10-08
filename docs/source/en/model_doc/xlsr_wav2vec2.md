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
*This model was released on 2020-06-24 and added to Hugging Face Transformers on 2023-06-20.*

# XLSR-Wav2Vec2

[XLSR-Wav2Vec2](https://huggingface.co/papers/2006.13979) learns cross-lingual speech representations by pretraining a single model from raw speech waveforms across multiple languages. Built on wav2vec 2.0, it solves a contrastive task over masked latent speech representations and jointly learns a quantization of the latents shared across languages. Fine-tuned on labeled data, XLSR significantly outperforms monolingual pretraining, reducing phoneme error rate by 72% on CommonVoice and improving word error rate by 16% on BABEL. The model demonstrates shared latent discrete speech representations across languages, with increased sharing among related languages. XLSR-53, pretrained in 53 languages, is released to catalyze research in low-resource speech understanding.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-xls-r-300m", dtype="auto")
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

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

