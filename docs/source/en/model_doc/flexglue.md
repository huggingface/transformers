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
*This model was published in HF papers on 2026-10-30 and contributed to Hugging Face Transformers on 2026-11-15.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# FlexGlue

[FlexGlue](https://huggingface.co/papers/2610.32976) is a feature matcher that works with any keypoints, with dense grids, or with no keypoints at all. It matches through dense features: a [DINOv3](./dinov3) ConvNeXt with a light neck encodes each image into a stride-8 feature map, a transformer with self and cross attention refines the feature maps of both images jointly, and each query point finds its correspondence in the other image by correlation, a windowed 1.5-entmax readout and a regressed sub-cell offset. FlexGlue also predicts whether the two images are covisible at all, which rejects look-alike pairs such as repeated facades.

> [!TIP]
> This model was contributed by [Sanskar1724](https://github.com/Sanskar1724).
>
> Click on the FlexGlue models in the right sidebar for more examples of how to apply FlexGlue to different computer vision tasks.

The example below demonstrates how to match keypoints between two images with the [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

keypoint_matcher = pipeline(task="keypoint-matching", model="ETH-CVG/flexglue")

url_0 = "https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin1.jpg"
url_1 = "https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin2.jpg"

results = keypoint_matcher([url_0, url_1], threshold=0.5)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from transformers.image_utils import load_image

image1 = load_image("https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin1.jpg")
image2 = load_image("https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin2.jpg")
images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("ETH-CVG/flexglue")
model = AutoModelForKeypointMatching.from_pretrained("ETH-CVG/flexglue")

inputs = processor(images, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# Post-process to get the matches in the coordinates of the original images
image_sizes = [[(image.height, image.width) for image in images]]
processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
```

</hfoption>
</hfoptions>

## Notes

- Without keypoints, FlexGlue predicts dense correspondences on a grid of both images (`coarse_stride` pixels apart, 64x64 cells for 512x512 inputs), checks them for cycle consistency, and samples up to `max_num_samples` matches from them, spread over the images as RoMa v2 does. The sampling is random, so set a seed with `torch.manual_seed` for reproducible matches. The dense grids are returned too.

    ```py
    outputs.dense_correspondences  # (batch_size, 2, 64, 64, 2), relative (x, y) in the other image
    outputs.dense_confidence  # (batch_size, 2, 64, 64), 0 where a check fails
    ```

- FlexGlue matches any keypoints, for example those of [SuperPoint](./superpoint). Pass them as relative coordinates in [0, 1] with the center of the top-left pixel at `(0.5 / width, 0.5 / height)`. By default, each keypoint is paired with its predicted correspondence in the other image, for both images. With `sparse_matching=True`, the keypoints are matched to each other with mutual nearest neighbors, as LightGlue does.

    ```py
    keypoints = torch.rand(1, 2, 1024, 2)  # (batch_size, 2, num_keypoints, 2)
    outputs = model(**inputs, keypoints=keypoints)

    model.config.sparse_matching = True
    outputs = model(**inputs, keypoints=keypoints)
    ```

- `covisibility_score` is the probability that the two images show the same place. With `min_covisibility > 0` in the config, all the matches of a pair below this score are rejected.
- Use fewer transformer layers for faster matching, at some cost in accuracy. All the layers come with their own match head.

    ```py
    model = AutoModelForKeypointMatching.from_pretrained("ETH-CVG/flexglue", num_hidden_layers=6)
    ```

- The weights of the DINOv3 ConvNeXt backbone are distributed under the [DINOv3 License](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md). The other weights are released under the MIT license.

## Resources

- Refer to the [original FlexGlue repository](https://github.com/cvg/FlexGlue) for more examples and implementation details.

## FlexGlueConfig

[[autodoc]] FlexGlueConfig

## FlexGlueImageProcessor

[[autodoc]] FlexGlueImageProcessor
    - preprocess
    - post_process_keypoint_matching
    - visualize_keypoint_matching

## FlexGlueImageProcessorPil

[[autodoc]] FlexGlueImageProcessorPil
    - preprocess
    - post_process_keypoint_matching
    - visualize_keypoint_matching

## FlexGlueModel

[[autodoc]] FlexGlueModel
    - forward

## FlexGlueForKeypointMatching

[[autodoc]] FlexGlueForKeypointMatching
    - forward
