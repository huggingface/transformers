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
*This model was released on 2021-07-30 and added to Hugging Face Transformers on 2021-12-08 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Perceiver

[Perceiver IO](https://huggingface.co/papers/2107.14795) is a general-purpose machine learning architecture designed to handle data from arbitrary domains while scaling linearly with input and output size. It extends the original Perceiver model with a flexible querying mechanism that allows outputs of varying sizes and semantics, eliminating the need for task-specific design. The architecture achieves strong performance across diverse tasks, including natural language processing, visual understanding, multi-task and multi-modal reasoning, and even StarCraft II gameplay. Notably, it surpasses a BERT baseline on the GLUE benchmark without input tokenization and sets a state-of-the-art result on Sintel optical flow estimation without specialized multiscale mechanisms.

<hfoptions id="usage">
<hfoption id="PerceiverForImageClassificationLearned">

```py
import requests
from transformers import AutoImageProcessor, PerceiverForImageClassificationLearned
from PIL import Image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned", dtype="auto")

inputs = image_processor(images=image, return_tensors="pt").pixel_values
outputs = model(inputs=inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

</hfoption>
</hfoptions>

## Perceiver specific outputs

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverModelOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverDecoderOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassifierOutput

## PerceiverConfig

[[autodoc]] PerceiverConfig

## PerceiverTokenizer

[[autodoc]] PerceiverTokenizer
    - __call__

## PerceiverImageProcessor

[[autodoc]] PerceiverImageProcessor
    - preprocess

## PerceiverImageProcessorFast

[[autodoc]] PerceiverImageProcessorFast
    - preprocess

## PerceiverTextPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverTextPreprocessor

## PerceiverImagePreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverImagePreprocessor

## PerceiverOneHotPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOneHotPreprocessor

## PerceiverAudioPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor

## PerceiverMultimodalPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor

## PerceiverProjectionDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionDecoder

## PerceiverBasicDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicDecoder

## PerceiverClassificationDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationDecoder

## PerceiverOpticalFlowDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder

## PerceiverBasicVideoAutoencodingDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicVideoAutoencodingDecoder

## PerceiverMultimodalDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder

## PerceiverProjectionPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor

## PerceiverAudioPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor

## PerceiverClassificationPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor

## PerceiverMultimodalPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor

## PerceiverModel

[[autodoc]] PerceiverModel
    - forward

## PerceiverForMaskedLM

[[autodoc]] PerceiverForMaskedLM
    - forward

## PerceiverForSequenceClassification

[[autodoc]] PerceiverForSequenceClassification
    - forward

## PerceiverForImageClassificationLearned

[[autodoc]] PerceiverForImageClassificationLearned
    - forward

## PerceiverForImageClassificationFourier

[[autodoc]] PerceiverForImageClassificationFourier
    - forward

## PerceiverForImageClassificationConvProcessing

[[autodoc]] PerceiverForImageClassificationConvProcessing
    - forward

## PerceiverForOpticalFlow

[[autodoc]] PerceiverForOpticalFlow
    - forward

## PerceiverForMultimodalAutoencoding

[[autodoc]] PerceiverForMultimodalAutoencoding
    - forward

