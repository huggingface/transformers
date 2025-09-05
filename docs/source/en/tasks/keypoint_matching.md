<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Keypoint matching

Keypoint matching matches different points of interests that belong to same object appearing in two different images. Most modern keypoint matchers only take images as input. They output following:

- **Keypoint coordinations (x,y):** They output two separate lists of keypoint coordinations where first one is location of the keypoint in the first image, and second one (on the same index) is the coordination of the keypoint matched. 
- **Matching scores:** Scores assigned to the keypoint matches.

In this tutorial we will extract keypoint matches with [EfficientLoFTR model trained with MatchAnything framework](https://huggingface.co/zju-community/matchanything_eloftr), and refine the matches. There's two ways to do this, using `AutoModelForKeypointMatching` or `keypoint-matching` pipeline. Let's see the former in action. This model is only 16M parameters so we can conveniently run on CPU.

```python
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
import torch

processor = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr")
model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr"))
```

Load two images that have same objects of interest, one photo is taken second apart, colors are edited, further cropped and rotated.

<div style="display: flex; align-items: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" 
         alt="Bee" 
         style="height: 200px; object-fit: contain; margin-right: 10px;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_edited.jpg" 
         alt="Cats" 
         style="height: 200px; object-fit: contain;">
</div>

```python 
from transformers.image_utils import load_image
image1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
image2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_edited.jpg")

images = [image1, image2]
```

We can pass the images to processor and infer.

```python
inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

We can postprocess the outputs. The threshold parameter is used to refine noise (lower confidence thresholds) in the output matches.

```python
image_sizes = [[(image.height, image.width) for image in images]]

outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
print(outputs)
```

Here's the outputs.

```
[{'keypoints0': tensor([[4514,  550],
          [4813,  683],
          [1972, 1547],
          ...
          [3916, 3408]], dtype=torch.int32),
  'keypoints1': tensor([[2280,  463],
          [2378,  613],
          [2231,  887],
          ...
          [1521, 2560]], dtype=torch.int32),
  'matching_scores': tensor([0.2189, 0.2073, 0.2414, ...
    ])}]
``` 

We have trimmed the output but there's 401 matches!

```python
len(outputs[0]["keypoints0"])
# 401
``` 

We can visualize them using `visualize_keypoint_matching` method of the processor. 

```python
plot_images = processor.visualize_keypoint_matching(images, outputs)
plot_images
```

![Matched Image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/matched_bees.png)

Optionally, you can use the keypoint matching pipeline. 

```python
from transformers import pipeline 

image_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_edited.jpg"

pipe = pipeline("keypoint-matching", model="zju-community/matchanything_eloftr")
pipe([image_1, image_2])
```

The output looks like following.

```bash
[{'keypoint_image_0': {'x': 2444, 'y': 2869},
  'keypoint_image_1': {'x': 837, 'y': 1500},
  'score': 0.9756593704223633},
 {'keypoint_image_0': {'x': 1248, 'y': 2819},
  'keypoint_image_1': {'x': 862, 'y': 866},
  'score': 0.9735618829727173},
 {'keypoint_image_0': {'x': 1547, 'y': 3317},
  'keypoint_image_1': {'x': 1436, 'y': 1500},
  ...
 }
]
```
