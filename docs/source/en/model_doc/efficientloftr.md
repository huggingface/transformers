<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# EfficientLoFTR

[EfficientLoFTR](https://huggingface.co/papers/2403.04765) is an efficient detector-free local feature matching method that produces semi-dense matches across images with sparse-like speed. It builds upon the original [LoFTR](https://huggingface.co/papers/2104.00680) architecture but introduces significant improvements for both efficiency and accuracy. The key innovation is an aggregated attention mechanism with adaptive token selection that makes the model ~2.5× faster than LoFTR while achieving higher accuracy. EfficientLoFTR can even surpass state-of-the-art efficient sparse matching pipelines like [SuperPoint](./superpoint) + [LightGlue](./lightglue) in terms of speed, making it suitable for large-scale or latency-sensitive applications such as image retrieval and 3D reconstruction.

> [!TIP]
> This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
>
> Click on the EfficientLoFTR models in the right sidebar for more examples of how to apply EfficientLoFTR to different computer vision tasks.

The example below demonstrates how to match keypoints between two images with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Post-process to get keypoints and matches
image_sizes = [[(image.height, image.width) for image in images]]
processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
```

</hfoption>
</hfoptions>

## Notes

- EfficientLoFTR is designed for efficiency while maintaining high accuracy. It uses an aggregated attention mechanism with adaptive token selection to reduce computational overhead compared to the original LoFTR.

    ```py
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    import torch
    from PIL import Image
    import requests
    
    processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")
    
    # EfficientLoFTR requires pairs of images
    images = [image1, image2]
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    
    # Extract matching information
    keypoints = outputs.keypoints        # Keypoints in both images
    matches = outputs.matches            # Matching indices 
    matching_scores = outputs.matching_scores  # Confidence scores
    ```

- The model produces semi-dense matches, offering a good balance between the density of matches and computational efficiency. It excels in handling large viewpoint changes and texture-poor scenarios.

- For better visualization and analysis, use the [`~EfficientLoFTRImageProcessor.post_process_keypoint_matching`] method to get matches in a more readable format.

    ```py
    # Process outputs for visualization
    image_sizes = [[(image.height, image.width) for image in images]]
    processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    
    for i, output in enumerate(processed_outputs):
        print(f"For the image pair {i}")
        for keypoint0, keypoint1, matching_score in zip(
                output["keypoints0"], output["keypoints1"], output["matching_scores"]
        ):
            print(f"Keypoint at {keypoint0.numpy()} matches with keypoint at {keypoint1.numpy()} with score {matching_score}")
    ```

- Visualize the matches between the images using the built-in plotting functionality.

    ```py
    # Easy visualization using the built-in plotting method
    visualized_images = processor.visualize_keypoint_matching(images, processed_outputs)
    ```

- EfficientLoFTR uses a novel two-stage correlation layer that achieves accurate subpixel correspondences, improving upon the original LoFTR's fine correlation module.

<div class="flex justify-center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2nJZQlFToCYp_iLurvcZ4.png">
</div>

## Resources

- Refer to the [original EfficientLoFTR repository](https://github.com/zju3dv/EfficientLoFTR) for more examples and implementation details.
- [EfficientLoFTR project page](https://zju3dv.github.io/efficientloftr/) with interactive demos and additional information.

## EfficientLoFTRConfig

[[autodoc]] EfficientLoFTRConfig

## EfficientLoFTRImageProcessor

[[autodoc]] EfficientLoFTRImageProcessor

- preprocess
- post_process_keypoint_matching
- visualize_keypoint_matching

<frameworkcontent>
<pt>
## EfficientLoFTRModel

[[autodoc]] EfficientLoFTRModel

- forward

## EfficientLoFTRForKeypointMatching

[[autodoc]] EfficientLoFTRForKeypointMatching

- forward

</pt>
</frameworkcontent>