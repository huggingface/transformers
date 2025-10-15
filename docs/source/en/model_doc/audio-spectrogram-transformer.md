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
*This model was released on 2021-04-05 and added to Hugging Face Transformers on 2022-11-21 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Audio Spectrogram Transformer

[Audio Spectrogram Transformer](https://huggingface.co/papers/2104.01778) applies a Vision Transformer to audio by converting audio into spectrograms, achieving state-of-the-art results in audio classification without using convolutional layers. It outperforms existing models on benchmarks like AudioSet, ESC-50, and Speech Commands V2, demonstrating the effectiveness of purely attention-based models in this domain.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="audio-classification",model="MIT/ast-finetuned-audioset-10-10-0.4593", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="AutoModel"

```py
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
print(f"Predicted label: {model.config.id2label[predicted_class_ids]}")
```

</hfoption>
</hfoptions>

## Usage tips

- When fine-tuning the Audio Spectrogram Transformer (AST) on your own dataset, normalize your input data to have a mean of 0 and standard deviation of 0.5. [`ASTFeatureExtractor`] handles this automatically. It uses AudioSet's mean and standard deviation by default. Check `ast/src/get_norm_stats.py` to see how the AST authors compute normalization statistics for downstream datasets.
- AST requires a low learning rate and converges quickly. The original authors used a learning rate 10 times smaller than their CNN model from the [PSLA](https://huggingface.co/papers/2102.01243) paper. Experiment with different learning rates and schedulers to find what works best for your specific task.

## ASTConfig

[[autodoc]] ASTConfig

## ASTFeatureExtractor

[[autodoc]] ASTFeatureExtractor
    - __call__

## ASTModel

[[autodoc]] ASTModel
    - forward

## ASTForAudioClassification

[[autodoc]] ASTForAudioClassification
    - forward

