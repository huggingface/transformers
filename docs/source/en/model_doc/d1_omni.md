<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-06.*

# D1 Omni

[D1 Omni](https://huggingface.co/LiquidAI/d1-omni-600M) answers typed questions over text and images, or text and
16 kHz mono audio, without generating tokens. Its bidirectional LFM2 encoder reads media prefixes and text; a
bidirectional decision head scores the option markers. The released checkpoint uses SigLIP2 vision and a
FastConformer audio encoder. A request can contain images or audio, but cannot contain both.

The native implementation reuses Transformers components through its modular source. It loads the publisher's
checkpoint without `trust_remote_code`; checkpoint conversion handles the fused head projections and audio names.
Use float32 to match the released model. The publisher reports that bfloat16 can change close decisions.

```python
from transformers import AutoModelForDecision, AutoProcessor, pipeline

model_id = "LiquidAI/d1-omni-600M"
model = AutoModelForDecision.from_pretrained(model_id, dtype="float32")
processor = AutoProcessor.from_pretrained(model_id)
decide = pipeline("decision", model=model, processor=processor, tokenizer=processor.tokenizer)
answers = decide({"state": "My invoice was charged twice.", "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"}
}})
```

The processor also accepts `state`, a list of `questions`, and either `images` or `audio`. Audio must be mono at
16 kHz; clips are truncated to 30 seconds and padded to at least half a second. Images use the released tiling and
pixel-unshuffle projection. The processor preserves the publisher's prompt escaping and token budgets.

Training passes integer `labels` to the model and receives `loss` and uncalibrated `logits`. Labels index the options
in publisher order: Choice preserves insertion order, Score uses ascending levels, and Noul uses **false, true**.
Use `-100` to ignore a question. `marker_mask` excludes padded options. `num_items_in_batch` normalizes the summed
loss by the labeled-question count across accumulated microbatches.

The model offers plain cross-entropy over active options as a training adaptation. LiquidAI's released code
contains inference only and does not specify the original training objective.
Temperature calibration is applied by `decide` only to text-only inference, matching the release; training uses raw logits.
Null answers are unsupported.

## D1OmniConfig

[[autodoc]] D1OmniConfig

## D1OmniModel

[[autodoc]] D1OmniModel
    - forward

## D1OmniForDecision

[[autodoc]] D1OmniForDecision
    - forward
    - decide

## D1OmniProcessor

[[autodoc]] D1OmniProcessor
    - __call__

## D1OmniTextConfig

[[autodoc]] D1OmniTextConfig

## D1OmniTextModel

[[autodoc]] D1OmniTextModel
    - forward

## D1OmniAudioEncoderConfig

[[autodoc]] D1OmniAudioEncoderConfig

## D1OmniAudioEncoder

[[autodoc]] D1OmniAudioEncoder
    - forward

## D1OmniImageProcessor

[[autodoc]] D1OmniImageProcessor

## D1OmniFeatureExtractor

[[autodoc]] D1OmniFeatureExtractor
    - __call__
