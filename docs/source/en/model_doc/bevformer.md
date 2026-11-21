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

# BEVFormer

[BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270) was proposed by Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai.

BEVFormer applies spatiotemporal transformers to camera-only 3D object detection by constructing a unified Bird's-Eye-View (BEV) representation from multi-camera image features across consecutive frames:

- **Spatial Cross-Attention**: Lifts 2D BEV query pillars across predefined 3D height anchors, projects them onto multi-view camera inputs using projection matrices (`lidar2img`), and samples visual features via deformable attention across relevant camera views.
- **Temporal Self-Attention**: Fuses historical BEV features (`prev_bev`) aligned with ego-motion (`can_bus`) translation and yaw rotation.
- **Stateless Design**: While upstream frameworks store temporal state inside the model module, the Hugging Face Transformers implementation is completely stateless. The caller passes `prev_bev` into `forward` and receives `outputs.bev_embed` to pass into the subsequent frame.

The abstract from the paper is the following:

*3D visual perception from multiple cameras is attractive for autonomous driving with low cost. However, existing methods often fail to achieve satisfactory results due to the lack of depth information. In this paper, we propose a new framework termed BEVFormer, which learns unified bird's-eye-view (BEV) representations with spatiotemporal transformers to support multiple autonomous driving perception tasks. In a nutshell, BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal domains through predefined grid-shaped BEV queries. To aggregate spatial information, we design spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views. For temporal information, we propose temporal self-attention to recurrently fuse the history BEV information. Our approach delivers impressive results on the nuScenes test set, e.g., 56.9% in terms of NDS, outperforming previous best works by 9.0 points and achieving comparable performance to LiDAR-based baselines.*

## Usage Example

The example below demonstrates how to perform two-frame sequential 3D object detection using `AutoImageProcessor` and `AutoModelFor3DObjectDetection`, passing `prev_bev` across timesteps:

```python
import torch
from transformers import AutoImageProcessor, AutoModelFor3DObjectDetection

checkpoint = "hasanfaesal/bevformer-tiny"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelFor3DObjectDetection.from_pretrained(checkpoint)

# Frame 1: batch of 6 multi-view camera images (batch_size=1, num_cameras=6, height=900, width=1600, channels=3)
# and calibration projection matrix (batch_size=1, num_cameras=6, 4, 4)
pixel_values = torch.randn(1, 6, 3, 480, 800)
lidar2img = torch.rand(1, 6, 4, 4)
can_bus_frame1 = torch.zeros(1, 18)

# Run Frame 1 (no previous BEV history)
outputs_frame1 = model(
    pixel_values=pixel_values,
    lidar2img=lidar2img,
    can_bus=can_bus_frame1,
    prev_bev=None,
)

# Extract predictions and current BEV embedding
bev_embed_frame1 = outputs_frame1.bev_embed
results_frame1 = image_processor.post_process_object_detection(outputs_frame1)

# Frame 2: subsequent timestep with ego-motion delta in can_bus
can_bus_frame2 = torch.zeros(1, 18)
# can_bus translation & rotation deltas populate can_bus_frame2
outputs_frame2 = model(
    pixel_values=pixel_values,
    lidar2img=lidar2img,
    can_bus=can_bus_frame2,
    prev_bev=bev_embed_frame1.detach(),
)

results_frame2 = image_processor.post_process_object_detection(outputs_frame2)
for score, label_id, box in zip(results_frame2[0]["scores"], results_frame2[0]["labels"], results_frame2[0]["boxes"]):
    print(f"{model.config.id2label[label_id.item()]}: {score.item():.2f} {box.tolist()}")
```

## BevformerConfig

[[autodoc]] BevformerConfig

## BevformerImageProcessor

[[autodoc]] BevformerImageProcessor
    - preprocess
    - post_process_object_detection

## BevformerModel

[[autodoc]] BevformerModel
    - forward

## BevformerFor3DObjectDetection

[[autodoc]] BevformerFor3DObjectDetection
    - forward
