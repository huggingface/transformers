<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on 2019-12-03 and added to Hugging Face Transformers on 2021-11-18 and contributed by [nielsr](https://huggingface.co/nielsr) and [7](https://github.com/openai/image-gpt/issues/7).*

# ImageGPT

[ImageGPT](https://huggingface.co/papers/1912.04958) is a GPT-2-like model designed to predict the next pixel value in an image, enabling both unconditional and conditional image generation. Trained on low-resolution ImageNet without labels, the model demonstrates strong image representations through linear probing and fine-tuning. It achieves 96.3% accuracy on CIFAR-10 with a linear probe, surpassing a supervised Wide ResNet, and 99.0% accuracy with full fine-tuning, matching top supervised pre-trained models. Additionally, it competes with self-supervised benchmarks on ImageNet when using pixel-based encodings, achieving 69.0% top-1 accuracy with a linear probe.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="openai/imagegpt-small", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
model = AutoModelForImageClassification.from_pretrained("openai/imagegpt-small", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- ImageGPT is almost exactly the same as GPT-2, with two exceptions: it uses a different activation function ("quick gelu") and the layer normalization layers don't mean center the inputs. ImageGPT also doesn't have tied input and output embeddings.
- The attention mechanism of Transformers scales quadratically with sequence length. The authors pre-trained ImageGPT on smaller input resolutions like 32x32 and 64x64. However, feeding a sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large.
- The authors applied k-means clustering to the (R,G,B) pixel values with k=512. This creates a 32*32 = 1024-long sequence of integers in the range 0..511. This shrinks the sequence length at the cost of a bigger embedding matrix. The vocabulary size of ImageGPT is 512, plus 1 for a special "start of sentence" (SOS) token used at the beginning of every sequence.
- Use [`ImageGPTImageProcessor`] to prepare images for the model.
- Despite being pre-trained entirely unsupervised (without any labels), ImageGPT produces performant image features useful for downstream tasks like image classification. The authors showed that features in the middle of the network are the most performant and work as-is to train a linear model (like a sklearn logistic regression model). This is called "linear probing".
- Obtain features by first forwarding the image through the model, then specifying `output_hidden_states=True`, and then average-pool the hidden states at whatever layer you like.
- Fine-tune the entire model on a downstream dataset, similar to BERT. Use [`ImageGPTForImageClassification`] for this.

## ImageGPTConfig

[[autodoc]] ImageGPTConfig

## ImageGPTImageProcessor

[[autodoc]] ImageGPTImageProcessor
    - preprocess

## ImageGPTImageProcessorFast

[[autodoc]] ImageGPTImageProcessorFast
    - preprocess

## ImageGPTModel

[[autodoc]] ImageGPTModel
    - forward

## ImageGPTForCausalImageModeling

[[autodoc]] ImageGPTForCausalImageModeling
    - forward

## ImageGPTForImageClassification

[[autodoc]] ImageGPTForImageClassification
    - forward

