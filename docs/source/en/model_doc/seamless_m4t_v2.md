<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-10-12 and added to Hugging Face Transformers on 2023-11-30 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

# SeamlessM4T-v2

[SeamlessM4T-v2](https://huggingface.co/papers/2310.08461) is an advanced multilingual and multimodal model that supports speech-to-speech, speech-to-text, text-to-speech, text-to-text translation, and automatic speech recognition. Built on the UnitY2 framework, it includes an expanded SeamlessAlign dataset with 114,800 hours of aligned data for 76 languages. SeamlessM4T-v2 serves as the foundation for SeamlessExpressive, which preserves vocal styles and prosody, and SeamlessStreaming, which uses Efficient Monotonic Multihead Attention (EMMA) for low-latency simultaneous translation. The models are evaluated using novel metrics for prosody, latency, and robustness, and human evaluations assess meaning, naturalness, and expressivity. Additionally, the system includes red-teaming for toxicity detection, gender bias evaluation, and watermarking to mitigate deepfake risks. Seamless, the resulting system, enables real-time expressive cross-lingual communication.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="facebook/seamless-m4t-v2-large")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="SeamlessM4Tv2ForSpeechToText">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Usage tips

- [`SeamlessM4Tv2Model`] is the top-level model for generating speech and text. Use dedicated models for specific tasks to reduce memory footprint.
- Change the speaker for speech synthesis with the `speaker_id` argument. Some speaker IDs work better for specific languages.
- Use different generation strategies for text generation. For example, `.generate(input_ids=input_ids, text_num_beams=4, text_do_sample=True)` performs multinomial beam-search decoding on the text model. Speech generation supports greedy (default) or multinomial sampling with `.generate(..., speech_do_sample=True, speech_temperature=0.6)`.
- Set `return_intermediate_token_ids=True` with [`SeamlessM4Tv2Model`] to return both speech and text.

## SeamlessM4Tv2Model

[[autodoc]] SeamlessM4Tv2Model
    - generate

## SeamlessM4Tv2ForTextToSpeech

[[autodoc]] SeamlessM4Tv2ForTextToSpeech
    - generate

## SeamlessM4Tv2ForSpeechToSpeech

[[autodoc]] SeamlessM4Tv2ForSpeechToSpeech
    - generate

## SeamlessM4Tv2ForTextToText

[[autodoc]] transformers.SeamlessM4Tv2ForTextToText
    - forward
    - generate

## SeamlessM4Tv2ForSpeechToText

[[autodoc]] transformers.SeamlessM4Tv2ForSpeechToText
    - forward
    - generate

## SeamlessM4Tv2Config

[[autodoc]] SeamlessM4Tv2Config

