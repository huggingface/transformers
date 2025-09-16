<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-23 and added to Hugging Face Transformers on 2025-06-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# LightGlue

[LightGlue](https://huggingface.co/papers/2306.13643) is a deep neural network that learns to match local features across images. It revisits multiple design decisions of SuperGlue and derives simple but effective improvements. Cumulatively, these improvements make LightGlue more efficient - in terms of both memory and computation, more accurate, and much easier to train. Similar to [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor), this model consists of matching two sets of local features extracted from two images, with the goal of being faster than SuperGlue. Paired with the [SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and estimate the pose between them.

You can find all the original LightGlue checkpoints under the [ETH-CVG](https://huggingface.co/ETH-CVG) organization.

> [!TIP]
> This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
>
> Click on the LightGlue models in the right sidebar for more examples of how to apply LightGlue to different computer vision tasks.

The example below demonstrates how to match keypoints between two images with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

keypoint_matcher = pipeline(task="keypoint-matching", model="ETH-CVG/lightglue_superpoint")

url_0 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"

results = keypoint_matcher([url_0, url_1], threshold=0.9)
print(results[0])
# {'keypoint_image_0': {'x': ..., 'y': ...}, 'keypoint_image_1': {'x': ..., 'y': ...}, 'score': ...}
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

inputs = processor(images, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# Post-process to get keypoints and matches
image_sizes = [[(image.height, image.width) for image in images]]
processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
```

</hfoption>
</hfoptions>

## Notes

- LightGlue is adaptive to the task difficulty. Inference is much faster on image pairs that are intuitively easy to match, for example, because of a larger visual overlap or limited appearance change.

    ```py
    from transformers import AutoImageProcessor, AutoModel
    import torch
    from PIL import Image
    import requests
    
    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
    
    # LightGlue requires pairs of images
    images = [image1, image2]
    inputs = processor(images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    
    # Extract matching information
    keypoints0 = outputs.keypoints0  # Keypoints in first image
    keypoints1 = outputs.keypoints1  # Keypoints in second image
    matches = outputs.matches        # Matching indices
    matching_scores = outputs.matching_scores  # Confidence scores
    ```

- The model outputs matching indices, keypoints, and confidence scores for each match, similar to SuperGlue but with improved efficiency.
- For better visualization and analysis, use the [`LightGlueImageProcessor.post_process_keypoint_matching`] method to get matches in a more readable format.

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
    processor.visualize_keypoint_matching(images, processed_outputs)
    ```

<div class="flex justify-center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/duPp09ty8NRZlMZS18ccP.png">
</div>

## Resources

- Refer to the [original LightGlue repository](https://github.com/cvg/LightGlue) for more examples and implementation details.

## LightGlueConfig

[[autodoc]] LightGlueConfig

## LightGlueImageProcessor

[[autodoc]] LightGlueImageProcessor

- preprocess
- post_process_keypoint_matching
- visualize_keypoint_matching

<frameworkcontent>
<pt>
## LightGlueForKeypointMatching

[[autodoc]] LightGlueForKeypointMatching

- forward

</pt>
</frameworkcontent>
