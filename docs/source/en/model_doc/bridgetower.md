<!--Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-06-17 and added to Hugging Face Transformers on 2023-01-25 and contributed by [anahita-b](https://huggingface.co/anahita-b), [Tile](https://huggingface.co/Tile), and [shaoyent](https://huggingface.co/shaoyent).*

# BridgeTower

[BridgeTower](https://huggingface.co/papers/2206.08657) introduces bridge layers connecting the top layers of uni-modal encoders to each layer of the cross-modal encoder, enabling effective bottom-up cross-modal alignment and fusion. Pre-trained with only 4M images, BRIDGETOWER achieves state-of-the-art performance on various vision-language tasks, outperforming previous models with similar pre-training data and minimal additional parameters and computational costs. When scaled, it surpasses models trained on much larger datasets.

<hfoptions id="usage">
<hfoption id="BridgeTowerForContrastiveLearning">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, BridgeTowerForContrastiveLearning

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["An image of a cat walking in the snow", "A football player scoring a goal"]

processor = AutoProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc", dtype="auto")

scores = dict()
for text in texts:
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    # Get similarity score by computing cosine similarity
    score = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=1).item()
    scores[text] = score
    print(f"Text: '{text}' - Score: {score:.4f}")

best_text = max(scores, key=scores.get)
print(f"\nBest matching text: '{best_text}' with score: {scores[best_text]:.4f}")
```

</hfoption>
</hfoptions>

## Usage tips

- [`BridgeTowerProcessor`] wraps [`RobertaTokenizer`] and [`BridgeTowerImageProcessor`] into a single instance to encode text and prepare images.
- BridgeTower uses [`RobertaTokenizer`] to generate text embeddings and OpenAI's CLIP/ViT model to compute visual embeddings.
- Pre-trained checkpoints for BridgeTower-base and BridgeTower masked language modeling and image-text matching are available.
- See Table 5 for BridgeTower's performance on image retrieval and other downstream tasks.
- This model requires PyTorch 1.10 or higher.

## BridgeTowerConfig

[[autodoc]] BridgeTowerConfig

## BridgeTowerTextConfig

[[autodoc]] BridgeTowerTextConfig

## BridgeTowerVisionConfig

[[autodoc]] BridgeTowerVisionConfig

## BridgeTowerImageProcessor

[[autodoc]] BridgeTowerImageProcessor
    - preprocess

## BridgeTowerImageProcessorFast

[[autodoc]] BridgeTowerImageProcessorFast
    - preprocess

## BridgeTowerProcessor

[[autodoc]] BridgeTowerProcessor
    - __call__

## BridgeTowerModel

[[autodoc]] BridgeTowerModel
    - forward

## BridgeTowerForContrastiveLearning

[[autodoc]] BridgeTowerForContrastiveLearning
    - forward

## BridgeTowerForMaskedLM

[[autodoc]] BridgeTowerForMaskedLM
    - forward

## BridgeTowerForImageAndTextRetrieval

[[autodoc]] BridgeTowerForImageAndTextRetrieval
    - forward

