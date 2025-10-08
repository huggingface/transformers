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
*This model was released on 2020-06-20 and added to Hugging Face Transformers on 2021-02-02 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Wav2Vec2

[wav2vec 2.0](https://huggingface.co/papers/2006.11477) demonstrates that learning speech representations through self-supervised learning followed by fine-tuning on transcribed speech can surpass the best semi-supervised methods with a simpler approach. The model masks speech input in the latent space and solves a contrastive task over quantized latent representations. Experiments on Librispeech show 1.8/3.3 WER on clean/other test sets using all labeled data. With just one hour of labeled data, it outperforms previous state-of-the-art methods on the 100-hour subset using 100 times less labeled data. Even with ten minutes of labeled data and pre-training on 53k hours of unlabeled data, it achieves 4.8/8.2 WER, highlighting its effectiveness with limited labeled data.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h", dtype="auto")
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

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Wav2Vec2Config

[[autodoc]] Wav2Vec2Config

## Wav2Vec2CTCTokenizer

[[autodoc]] Wav2Vec2CTCTokenizer
    - __call__
    - save_vocabulary
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_pretrained
    - from_pretrained

## Wav2Vec2FeatureExtractor

[[autodoc]] Wav2Vec2FeatureExtractor
    - __call__

## Wav2Vec2Processor

[[autodoc]] Wav2Vec2Processor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Wav2Vec2ProcessorWithLM

[[autodoc]] Wav2Vec2ProcessorWithLM
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Wav2Vec2 specific outputs

[[autodoc]] models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput

## Wav2Vec2Model

[[autodoc]] Wav2Vec2Model
    - forward

## Wav2Vec2ForCTC

[[autodoc]] Wav2Vec2ForCTC
    - forward
    - load_adapter

## Wav2Vec2ForSequenceClassification

[[autodoc]] Wav2Vec2ForSequenceClassification
    - forward

## Wav2Vec2ForAudioFrameClassification

[[autodoc]] Wav2Vec2ForAudioFrameClassification
    - forward

## Wav2Vec2ForXVector

[[autodoc]] Wav2Vec2ForXVector
    - forward

## Wav2Vec2ForPreTraining

[[autodoc]] Wav2Vec2ForPreTraining
    - forward

