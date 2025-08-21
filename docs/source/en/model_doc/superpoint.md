<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2017-12-20 and added to Hugging Face Transformers on 2024-03-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# SuperPoint

[SuperPoint](https://huggingface.co/papers/1712.07629) is the result of self-supervised training of a fully-convolutional network for interest point detection and description. The model is able to detect interest points that are repeatable under homographic transformations and provide a descriptor for each point. Usage on it's own is limited, but it can be used as a feature extractor for other tasks such as homography estimation and image matching.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/superpoint_architecture.png"
alt="drawing" width="500"/>

You can find all the original SuperPoint checkpoints under the [Magic Leap Community](https://huggingface.co/magic-leap-community) organization.

> [!TIP]
> This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
>
> Click on the SuperPoint models in the right sidebar for more examples of how to apply SuperPoint to different computer vision tasks.



The example below demonstrates how to detect interest points in an image with the [`AutoModel`] class.
<hfoptions id="usage">
<hfoption id="AutoModel">

```py
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Post-process to get keypoints, scores, and descriptors
image_size = (image.height, image.width)
processed_outputs = processor.post_process_keypoint_detection(outputs, [image_size])
```

</hfoption>
</hfoptions>

## Notes

- SuperPoint outputs a dynamic number of keypoints per image, which makes it suitable for tasks requiring variable-length feature representations.

    ```py
    from transformers import AutoImageProcessor, SuperPointForKeypointDetection
    import torch
    from PIL import Image
    import requests
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
    url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
    image_2 = Image.open(requests.get(url_image_2, stream=True).raw)
    images = [image_1, image_2]
    inputs = processor(images, return_tensors="pt")
    # Example of handling dynamic keypoint output
    outputs = model(**inputs)
    keypoints = outputs.keypoints  # Shape varies per image
    scores = outputs.scores        # Confidence scores for each keypoint
    descriptors = outputs.descriptors  # 256-dimensional descriptors
    mask = outputs.mask # Value of 1 corresponds to a keypoint detection
    ```

- The model provides both keypoint coordinates and their corresponding descriptors (256-dimensional vectors) in a single forward pass.
- For batch processing with multiple images, you need to use the mask attribute to retrieve the respective information for each image. You can use the `post_process_keypoint_detection` from the `SuperPointImageProcessor` to retrieve the each image information.

    ```py
    # Batch processing example
    images = [image1, image2, image3]
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    image_sizes = [(img.height, img.width) for img in images]
    processed_outputs = processor.post_process_keypoint_detection(outputs, image_sizes)
    ```

- You can then print the keypoints on the image of your choice to visualize the result:
    ```py
    import matplotlib.pyplot as plt
    plt.axis("off")
    plt.imshow(image_1)
    plt.scatter(
        outputs[0]["keypoints"][:, 0],
        outputs[0]["keypoints"][:, 1],
        c=outputs[0]["scores"] * 100,
        s=outputs[0]["scores"] * 50,
        alpha=0.8
    )
    plt.savefig(f"output_image.png")
    ```

<div class="flex justify-center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/ZtFmphEhx8tcbEQqOolyE.png">
</div>

## Resources

- Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SuperPoint/Inference_with_SuperPoint_to_detect_interest_points_in_an_image.ipynb) for an inference and visualization example.

## SuperPointConfig

[[autodoc]] SuperPointConfig

## SuperPointImageProcessor

[[autodoc]] SuperPointImageProcessor

- preprocess

## SuperPointImageProcessorFast

[[autodoc]] SuperPointImageProcessorFast
- preprocess
- post_process_keypoint_detection

## SuperPointForKeypointDetection

[[autodoc]] SuperPointForKeypointDetection
- forward
