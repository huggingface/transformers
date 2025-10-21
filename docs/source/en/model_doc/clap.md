<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-11-12 and added to Hugging Face Transformers on 2023-02-16 and contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).*

# CLAP

[CLAP](https://huggingface.co/papers/2211.06687) is a neural network trained on a large dataset of audio-text pairs to develop a multimodal representation. It uses a SWINTransformer for audio feature extraction from log-Mel spectrograms and a RoBERTa model for text feature extraction. Both feature sets are projected into a shared latent space, where their similarity is measured using a dot product. The model incorporates feature fusion and keyword-to-caption augmentation to handle variable audio lengths and improve performance. Evaluations across text-to-audio retrieval, zero-shot audio classification, and supervised audio classification show that CLAP achieves superior results in text-to-audio retrieval and state-of-the-art performance in zero-shot audio classification, comparable to non-zero-shot models.

<hfoptions id="usage">
<hfoption id="ClapModel">

```py
from datasets import load_dataset
from transformers import AutoProcessor, ClapModel

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
audio_sample = dataset["train"]["audio"][0]["array"]

model = ClapModel.from_pretrained("laion/clap-htsat-unfused", dtype="auto")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["Sound of a dog", "Sound of vacuum cleaner"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio
probs = logits_per_audio.softmax(dim=-1)

for i, prob in enumerate(probs[0]):
    print(f"{input_text[i]}: {prob.item():.3f}")
```

</hfoption>
</hfoptions>

## ClapConfig

[[autodoc]] ClapConfig

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward

