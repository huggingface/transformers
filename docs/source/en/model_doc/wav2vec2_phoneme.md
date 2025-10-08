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
*This model was released on 2021-09-23 and added to Hugging Face Transformers on 2021-12-17 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# Wav2Vec2Phoneme

[Wav2Vec2Phoneme](https://huggingface.co/papers/2109.11680) extends zero-shot cross-lingual transfer learning by fine-tuning a multilingually pretrained Wav2Vec 2.0 model to transcribe unseen languages. It achieves this by mapping phonemes of training languages to the target language using articulatory features, outperforming prior methods that relied on task-specific architectures and partial monolingual pretraining.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-lv-60-espeak-cv-ft", dtype="auto")
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

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Wav2Vec2PhonemeCTCTokenizer

[[autodoc]] Wav2Vec2PhonemeCTCTokenizer
	- __call__
	- batch_decode
	- decode
	- phonemize

