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
*This model was released on 2021-04-14 and added to Hugging Face Transformers on 2023-06-20 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# Speech2Text2

[Speech2Text2](https://huggingface.co/papers/2104.06678) a speech translation model that leverages large unlabeled speech and text datasets through both pretraining and self-training. It uses wav2vec 2.0 for speech representation learning on the Libri-Light corpus and integrates language modeling trained on CommonCrawl text data. By combining these components with a single round of self-training and language model–assisted decoding, the approach achieves a 2.6 BLEU improvement on average across four CoVoST 2 language pairs. Notably, the method requires no additional supervised data beyond the original speech translation dataset.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/s2t-wav2vec2-large-en-de", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="SpeechEncoderDecoderModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCTC

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
model = AutoModelForCTC.from_pretrained("facebook/s2t-wav2vec2-large-en-de", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Speech2Text2Config

[[autodoc]] Speech2Text2Config

## Speech2TextTokenizer

[[autodoc]] Speech2Text2Tokenizer
    - batch_decode
    - decode
    - save_vocabulary

## Speech2Text2Processor

[[autodoc]] Speech2Text2Processor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Speech2Text2ForCausalLM

[[autodoc]] Speech2Text2ForCausalLM
    - forward
