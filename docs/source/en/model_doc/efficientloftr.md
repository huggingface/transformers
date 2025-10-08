<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-03-07 and added to Hugging Face Transformers on 2025-07-22 and contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).*

# EfficientLoFTR

[EfficientLoFTR](https://huggingface.co/papers/2403.04765) introduces an improved semi-dense image matching method that enhances both speed and accuracy over LoFTR, a detector-free matcher. The authors propose an aggregated attention mechanism with adaptive token selection to avoid redundant transformer computation, significantly boosting efficiency. To address accuracy issues, they design a two-stage correlation layer that improves subpixel correspondence by mitigating spatial variance in LoFTR’s fine correlation module. The resulting model runs about 2.5× faster than LoFTR while outperforming both LoFTR and the SuperPoint + LightGlue pipeline in accuracy, making it well-suited for large-scale or latency-sensitive tasks like image retrieval and 3D reconstruction.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="keypoint-matching", model="zju-community/efficientloftr", dtype="auto")
url_0 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"

pipeline([url_0, url_1], threshold=0.9)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr", dtype="auto")
inputs = processor(images, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

image_sizes = [[(image.height, image.width) for image in images]]
processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

for i, output in enumerate(processed_outputs):
    print(f"For the image pair {i}")
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(f"Keypoint at {keypoint0.numpy()} matches with keypoint at {keypoint1.numpy()} with score {matching_score}")

visualized_images = processor.visualize_keypoint_matching(images, processed_outputs)
```

</hfoption>
</hfoptions>

## EfficientLoFTRConfig

[[autodoc]] EfficientLoFTRConfig

## EfficientLoFTRImageProcessor

[[autodoc]] EfficientLoFTRImageProcessor

- preprocess
- post_process_keypoint_matching
- visualize_keypoint_matching

## EfficientLoFTRImageProcessorFast

[[autodoc]] EfficientLoFTRImageProcessorFast

- preprocess
- post_process_keypoint_matching
- visualize_keypoint_matching

## EfficientLoFTRModel

[[autodoc]] EfficientLoFTRModel

- forward

## EfficientLoFTRForKeypointMatching

[[autodoc]] EfficientLoFTRForKeypointMatching

- forward
