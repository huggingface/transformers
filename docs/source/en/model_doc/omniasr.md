<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-11-12 and contributed to Hugging Face Transformers on 2026-10-06.*

# OmniASR

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

OmniASR (Omnilingual ASR) is a suite of multilingual speech-recognition models from Meta FAIR, introduced in
[Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages](https://huggingface.co/papers/2511.09690), that covers more than
1,600 languages — including hundreds never previously supported by any ASR system. Each model pairs a Wav2Vec2-style
audio encoder (convolutional layers that downsample the 16 kHz waveform ~320× to a 50 Hz frame rate,
followed by a pre-norm Transformer encoder) with one of two heads:

- **CTC variant** ([`OmniASRCTCForCTC`], see [OmniASR CTC](omniasr_ctc)): a single linear projection to the vocabulary, decoded non-autoregressively with greedy CTC.
- **ALM (audio language model) variant** ([`OmniASRForConditionalGeneration`], this page): the audio embeddings are linearly projected into a Llama decoder which autoregressively generates the transcription. This variant additionally supports optional **language conditioning**: passing a language code such as `"eng_Latn"` makes the processor write the matching language token into the decoder context, which generally improves transcription quality.

Checkpoints can be found in [this collection](https://huggingface.co/collections/bezzam/omnilingual-asr-transformers-compatible): 4x ALM checkpoints of various sizes (300M, 1B, 3B, 7B). Original weights are released by Meta, and can be found on their [GitHub repo](https://github.com/facebookresearch/omnilingual-asr#model-architectures).


This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Usage

> [!NOTE]
> The simplest way to transcribe audio is with `apply_transcription_request`, which handles the chat template formatting for you, namely it is a convenience wrapper for `apply_chat_template` (see [Chat template](#chat-template) below).

### Simple transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "bezzam/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
)
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

### Language hint

Without a language, the model runs in its language-agnostic mode. Passing a language code (e.g. `"eng_Latn"`,
`"fra_Latn"`, `"cmn_Hans"`) writes the matching language token into the decoder prompt, which generally improves
transcription quality. The list of supported languages and naming convention follows the original [here](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py#L9).
For the languages that have one, the more common ISO 639-1 codes (e.g. `"en"`, `"fr"`, `"zh"`) are also accepted.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "bezzam/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav"

# Without language hint (language-agnostic)
inputs = processor.apply_transcription_request(audio=audio).to(model.device, dtype=model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(f"Language-agnostic: {processor.decode(generated_ids, skip_special_tokens=True)[0]}")

# With language hint
inputs = processor.apply_transcription_request(audio=audio, language="zh").to(model.device, dtype=model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(f"Language hint:     {processor.decode(generated_ids, skip_special_tokens=True)[0]}")
```

### Batch inference

Batch inference is possible by passing a list of audios and, if provided, a single language for the whole batch or a
list of languages (one per audio, `None` for the language-agnostic mode).

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "bezzam/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = [
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav",
]

inputs = processor.apply_transcription_request(audio, language=["en", None])
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
for i, text in enumerate(processor.decode(generated_ids, skip_special_tokens=True)):
    print(f"Audio {i + 1}: {text}")
```

### Chat template

The model also accepts chat template inputs. The `apply_transcription_request` usage
[above](#simple-transcription) is a convenience wrapper for `apply_chat_template`. The language hint is given as a
`{"type": "language", ...}` item of the user turn; leaving it out selects the language-agnostic mode.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "bezzam/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
                {"type": "language", "language": "en"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav",
                },
                {"type": "language", "language": "zh"},
            ],
        },
    ],
]

inputs = processor.apply_chat_template(
    conversations, tokenize=True, add_generation_prompt=True, return_dict=True
)
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
for text in processor.decode(generated_ids, skip_special_tokens=True):
    print(text)
```

### Training

The model can be trained with the loss it outputs. Put the target transcript in the assistant turn and pass
`output_labels=True`. Everything but the transcript and its EOS (the audio placeholders, the language prompt and the
padding) is masked automatically. Note that OmniASR transcribes in lowercase, so the (uppercase) LibriSpeech
transcripts are lowercased below.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "bezzam/omniasr-llm-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = [text.lower() for text in ds["text"][:NUM_SAMPLES]]

conversations = [
    [
        {
            "role": "user",
            # Any supported language code, or drop the language item to train the language-agnostic mode
            "content": [{"type": "audio", "audio": audio}, {"type": "language", "language": "en"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": text}]},
    ]
    for audio, text in zip(speech_samples, text_samples)
]

inputs = processor.apply_chat_template(
    conversations, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
)
inputs.to(model.device, dtype=model.dtype)

outputs = model(**inputs)
print("Loss:", outputs.loss.item())
outputs.loss.backward()
```

### Torch compile

For autoregressive transcription, `torch.compile` accelerates the per-token forward passes inside `generate` by
providing a `CompileConfig` object, which requires a static cache.

Note that the `StaticCache` is created once and reused across `generate` calls to avoid recompiles.

On an A100, we observed a speed-up of ~2.3 for a batch size of 4 ([script](https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-example_compile_llm-py)).

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, CompileConfig, StaticCache

model_id = "bezzam/omniasr-llm-300m-v2"
NUM_SAMPLES = 4
num_warmup = 3
max_new_tokens = 256

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request(
    audio=[audio_url] * NUM_SAMPLES,
    language="en",
).to(model.device, dtype=model.dtype)

# Created once and reused across `generate` calls, to avoid recompiling at every call
static_cache = StaticCache(
    config=model.config.get_text_config(decoder=True),
    max_cache_len=inputs["input_ids"].shape[1] + max_new_tokens,
)
compile_config = CompileConfig()

# Warmup
with torch.inference_mode():
    for _ in range(num_warmup):
        static_cache.reset()
        _ = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            past_key_values=static_cache, compile_config=compile_config,
        )
torch.cuda.synchronize()

# Apply model
with torch.inference_mode():
    static_cache.reset()
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        past_key_values=static_cache, compile_config=compile_config,
    )
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

## OmniASRAudioConfig

[[autodoc]] OmniASRAudioConfig

## OmniASRConfig

[[autodoc]] OmniASRConfig

## OmniASRFeatureExtractor

[[autodoc]] OmniASRFeatureExtractor

## OmniASRProcessor

[[autodoc]] OmniASRProcessor

## OmniASRAudioModel

[[autodoc]] OmniASRAudioModel
    - forward

## OmniASRModel

[[autodoc]] OmniASRModel
    - forward

## OmniASRForConditionalGeneration

[[autodoc]] OmniASRForConditionalGeneration
    - forward
