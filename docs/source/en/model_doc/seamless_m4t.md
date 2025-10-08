<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-10-12 and added to Hugging Face Transformers on 2023-10-23 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

# SeamlessM4T

[SeamlessM4T](https://huggingface.co/papers/2310.08461) is a unified model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition across up to 100 languages. Leveraging 1 million hours of open speech audio data and w2v-BERT 2.0 for self-supervised speech representations, SeamlessM4T achieves state-of-the-art performance, particularly setting a new standard on FLEURS with a 20% BLEU improvement in direct speech-to-text translation. Compared to cascaded models, it enhances into-English translation by 1.3 BLEU points in speech-to-text and 2.6 ASR-BLEU points in speech-to-speech. The model also demonstrates robustness against background noise and speaker variations and has been evaluated for gender bias and toxicity to ensure translation safety. All contributions are open-sourced.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="facebook/hf-seamless-m4t-medium")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="SeamlessM4TForSpeechToText">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, SeamlessM4TForSpeechToText

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TForSpeechToText.from_pretrained("facebook/hf-seamless-m4t-medium", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## SeamlessM4TModel

[[autodoc]] SeamlessM4TModel
    - generate

## SeamlessM4TForTextToSpeech

[[autodoc]] SeamlessM4TForTextToSpeech
    - generate

## SeamlessM4TForSpeechToSpeech

[[autodoc]] SeamlessM4TForSpeechToSpeech
    - generate

## SeamlessM4TForTextToText

[[autodoc]] transformers.SeamlessM4TForTextToText
    - forward
    - generate

## SeamlessM4TForSpeechToText

[[autodoc]] transformers.SeamlessM4TForSpeechToText
    - forward
    - generate

## SeamlessM4TConfig

[[autodoc]] SeamlessM4TConfig

## SeamlessM4TTokenizer

[[autodoc]] SeamlessM4TTokenizer
    - __call__
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SeamlessM4TTokenizerFast

[[autodoc]] SeamlessM4TTokenizerFast
    - __call__

## SeamlessM4TFeatureExtractor

[[autodoc]] SeamlessM4TFeatureExtractor
    - __call__

## SeamlessM4TProcessor

[[autodoc]] SeamlessM4TProcessor
    - __call__

## SeamlessM4TCodeHifiGan

[[autodoc]] SeamlessM4TCodeHifiGan

## SeamlessM4THifiGan

[[autodoc]] SeamlessM4THifiGan

## SeamlessM4TTextToUnitModel

[[autodoc]] SeamlessM4TTextToUnitModel

## SeamlessM4TTextToUnitForConditionalGeneration

[[autodoc]] SeamlessM4TTextToUnitForConditionalGeneration

