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
*This model was released on 2019-08-09 and added to Hugging Face Transformers on 2021-06-02.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# VisualBERT

[VisualBERT](https://huggingface.co/papers/1908.03557) is a vision-and-language model. It uses an approach called "early fusion", where inputs are fed together into a single Transformer stack initialized from [BERT](./bert). Self-attention implicitly aligns words with their corresponding image objects. It processes text with visual features from object-detector regions instead of raw pixels.

You can find all the original VisualBERT checkpoints under the [UCLA NLP](https://huggingface.co/uclanlp/models?search=visualbert) organization.


> [!TIP]
> This model was contributed by [gchhablani](https://huggingface.co/gchhablani).
> Click on the VisualBERT models in the right sidebar for more examples of how to apply VisualBERT to different image and language tasks.

The example below demonstrates how to answer a question based on an image with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
import torchvision
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import requests
from io import BytesIO

def get_visual_embeddings_simple(image, device=None):
    
    model = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Image must be a PIL Image or path to image file")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    batch_size = features.shape[0]
    feature_dim = features.shape[1]
    visual_seq_length = 10
    
    visual_embeds = features.squeeze(-1).squeeze(-1).unsqueeze(1).expand(batch_size, visual_seq_length, feature_dim)
    
    return visual_embeds

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
image = Image.open(BytesIO(response.content))
    
visual_embeds = get_visual_embeddings_simple(image)
    
inputs = tokenizer("What is shown in this image?", return_tensors="pt")
    
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
})
    
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer_idx = logits.argmax(-1).item()

print(f"Predicted answer: {predicted_answer_idx}")
```

</hfoption>
</hfoptions>

## Notes

- Use a fine-tuned checkpoint for downstream tasks, like `visualbert-vqa` for visual question answering. Otherwise, use one of the pretrained checkpoints.
- The fine-tuned detector and weights aren't provided (available in the research projects), but the states can be directly loaded into the detector.
- The text input is concatenated in front of the visual embeddings in the embedding layer and is expected to be bound by `[CLS]` and [`SEP`] tokens.
- The segment ids must be set appropriately for the text and visual parts.
- Use [`BertTokenizer`] to encode the text and implement a custom detector/image processor to get the visual embeddings.

## Resources

- Refer to this [notebook](https://github.com/huggingface/transformers-research-projects/tree/main/visual_bert) for an example of using VisualBERT for visual question answering.
- Refer to this [notebook](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing) for an example of how to generate visual embeddings.

## VisualBertConfig

[[autodoc]] VisualBertConfig

## VisualBertModel

[[autodoc]] VisualBertModel
    - forward

## VisualBertForPreTraining

[[autodoc]] VisualBertForPreTraining
    - forward

## VisualBertForQuestionAnswering

[[autodoc]] VisualBertForQuestionAnswering
    - forward

## VisualBertForMultipleChoice

[[autodoc]] VisualBertForMultipleChoice
    - forward

## VisualBertForVisualReasoning

[[autodoc]] VisualBertForVisualReasoning
    - forward

## VisualBertForRegionToPhraseAlignment

[[autodoc]] VisualBertForRegionToPhraseAlignment
    - forward
