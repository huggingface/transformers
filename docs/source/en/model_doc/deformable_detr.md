<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
	<div class="flex flex-wrap space-x-1">
		<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
	</div>
</div>

# Deformable DETR

[Deformable DETR](https://huggingface.co/papers/2010.04159) mitigates the slow convergence issues and limited feature spatial resolution of the original [DETR](detr) by leveraging a new deformable attention module which only attends to a small set of key sampling points around a reference.

You can find all the original DETR checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=deformable_detr) organization.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the Deformable DETR models in the right sidebar for more examples of how to apply Deformable DETR to different object detection and segmentation tasks.

## Notes

- Training Deformable DETR is equivalent to training the original [DETR](detr) model. See the [resources](#resources) section below for demo notebooks.

## Resources

- Demo notebooks regarding inference + fine-tuning on a custom dataset for [`DeformableDetrForObjectDetection`] can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR).
- Scripts for finetuning [`DeformableDetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor
    - preprocess
    - post_process_object_detection

## DeformableDetrImageProcessorFast

[[autodoc]] DeformableDetrImageProcessorFast
    - preprocess
    - post_process_object_detection

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor
    - __call__
    - post_process_object_detection

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel
    - forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection
    - forward
