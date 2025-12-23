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

# DINO DETR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

DINO DETR (DETR with Improved DeNoising Anchor Boxes) is a state-of-the-art end-to-end object detection model introduced in the paper [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605) by Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni and Heung-Yeung Shum.  It builds upon the original DETR framework by addressing key challenges in convergence speed and detection accuracy.

DINO DETR enhances the DETR architecture through three main innovations:

* **Mixed Query Selection for Anchor Initialization**: In DINO DETR decoder queries consist of query locations and query features. The decoder query locations are selected by passing the encoder features through a classification head and selecting the topk locations in terms of max class probability. The decoder query features are learnable weights shared across all samples. In this way, the query locations are initialized close to interesting objects and the query features are made robust to different object classes.

* **Contrastive Denoising Training**: DINO Detr uses denoising queries together with "standard" DETR queries. The denoising queries are strongly perturbed versions of standard queries, assigned to negative labels. This improves the model's robustness and convergence speed.

* **Look Forward Twice Scheme for Box Prediction**: This term means that the bounding boxes are refined iteratively from one decoder layer to the next, by adding corrections to the previous layer bounding boxes. This improves training stability.

These advancements enable DINO DETR to achieve significant performance improvements over previous DETR-like models. For instance, with a ResNet-50 backbone and multi-scale features, DINO attains 49.4 AP in 12 epochs and 51.3 AP in 24 epochs on the COCO dataset, marking a substantial enhancement in detection performance.

The abstract of the paper is the following:

*We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.*

This implementation is contributed by [kostaspitas](https://huggingface.co/kostaspitas) and is based on the official code available at [https://github.com/IDEA-Research/DINO](https://github.com/IDEA-Research/DINO).

### Model Architecture

DINO DETR with Improved DeNoising Anchor Boxes (DINO) follows a similar architecture to the original DETR but introduces enhancements for improved performance and efficiency.

#### 1. Backbone

An input image is processed through a pre-trained convolutional backbone, such as ResNet-50 or ResNet-101. This backbone extracts multi-scale feature maps, which are then projected to match the hidden dimension of the Transformer encoder. For instance, a feature map of shape `(batch_size, 2048, height/32, width/32)` is transformed to `(batch_size, 256, height/32, width/32)` using a convolutional layer. These feature maps are then flattened and transposed to obtain a tensor of shape `(batch_size, seq_len, d_model)`, where `seq_len` is the number of spatial locations and `d_model` is the model dimension (e.g., 256).

#### 2. Transformer Encoder

The flattened feature maps are passed through a multi-layer transformer encoder. The self-attention implementation is deformable attention as first introduced in the [Deformable DETR](https://arxiv.org/abs/2010.04159) paper. This encoder processes the input sequence and outputs `encoder_hidden_states`, which serve as the image features for the subsequent decoder.

#### 3. Object Queries and Decoder

DINO introduces dynamic anchor boxes as object queries. These queries are initialized based on anchor box coordinates and are updated through the decoder layers. The decoder receives these object queries along with the encoder outputs and processes them through multiple self-attention and encoder-decoder cross-attention layers. The decoder cross attention layers perform deformable attention while self-attention is standard multi-head self-attention. The result is `decoder_hidden_states`, which are then used for prediction.

#### 4. Prediction Heads

On top of the decoder outputs, DINO adds two prediction heads:

* **Classification Head**: A linear layer that classifies each object query into one of the object categories or "no object".

* **Bounding Box Head**: A multi-layer perceptron (MLP) that predicts the bounding box coordinates for each object query.

#### 5. Training with Bipartite Matching Loss

DINO employs a bipartite matching loss during training. The predicted classes and bounding boxes for each object query are compared to the ground truth annotations, padded to the same length. The Hungarian algorithm is used to find an optimal one-to-one mapping between the predicted and ground truth annotations. The loss function combines focal loss for classification and a linear combination of L1 loss and generalized Intersection over Union (IoU) loss for bounding box regression. In addition to the original DETR queries, DINO DETR adds a denoising query set which feeds noised groundtruth labels and boxes into the decoder to provide an aixiliary denoising loss. The denoising loss effectively stabilizes and speeds up the DINO DETR training. 

## Usage Tips

- **Object Queries**: DINO utilizes dynamic anchor boxes as object queries to detect objects within an image. The number of queries (`num_queries`) determines the maximum number of objects that can be detected in a single image. By default, this is set to 900 (e.g., 300 queries × 3 patterns) to enhance detection performance.
- **Decoder Parallelism**: Similar to the original DETR, DINO's decoder updates object queries in parallel, differing from autoregressive models like GPT-2. Consequently, no causal attention mask is employed.
- **Position Embeddings**: DINO adds position embeddings to the hidden states at each self-attention and cross-attention layer before projecting to queries and keys. For image position embeddings, you can choose between fixed sinusoidal or learned absolute position embeddings. By default, the `position_embedding_type` parameter in `DinoDetrConfig` is set to `"SineWH"`.
- **Auxiliary Losses**: During training, employing auxiliary losses in the decoder can be beneficial, especially for improving the model's ability to predict the correct number of objects per class. Setting the `auxiliary_loss` parameter in `DinoDetrConfig` to `True` adds an auxiliary_loss after each decoder layer.
- **Distributed Training**: When training the model across multiple nodes, it's important to update the `num_boxes` variable in the `DinoLoss` class to reflect the average number of target boxes across all nodes. This adjustment ensures proper loss computation during distributed training.
- **Backbone Initialization**: `DinoDetrForObjectDetection` can be initialized with any convolutional backbone available in the [timm library](https://github.com/rwightman/pytorch-image-models). For instance, to use a MobileNet backbone, set the `backbone` attribute in `DinoDetrConfig` to `"tf_mobilenetv3_small_075"` and initialize the model with this configuration.
- **Image Preprocessing**: DINO resizes input images such that the shortest side is at least a certain number of pixels, while the longest side is at most 1333 pixels. During training, scale augmentation is applied, randomly setting the shortest side to between 480 and 800 pixels and its longer side to be at most 1333 pixels. At inference time, the shortest side is set to 800 pixels. Use `DinoDetrImageProcessor` to prepare images (and optional annotations in COCO format) for the model. Due to resizing, images in a batch may have different sizes. DinoDetr addresses this by padding images to the largest size in the batch and creating a pixel mask to differentiate real pixels from padding. Alternatively, you can define a custom `collate_fn` to batch images using `DinoDetrImageProcessor.pad_and_create_pixel_mask`.
- **Batch Size Considerations**: The size of input images affects memory usage and, consequently, the `batch_size`.

### Model Initialization Options

There are three ways to instantiate a DinoDetr model:

**Option 1**: Instantiate DinoDetr with pre-trained weights for the entire model
```python
>>> from transformers import DinoDetrForObjectDetection

>>> model = DinoDetrForObjectDetection.from_pretrained("IDEA-Research/dino-resnet-50")
```
**Option 2**: Instantiate DinoDetr with randomly initialized Transformer weights but pre-trained backbone weights
```python
>>>  from transformers import DinoDetrConfig, DinoDetrForObjectDetection

>>> config = DinoDetrConfig()
>>> model = DinoDetrForObjectDetection(config)
```
**Option 3**: Instantiate DinoDetr with randomly initialized weights for both backbone and Transformer
```python
>>> config = DinoDetrConfig(use_pretrained_backbone=False)
>>> model = DinoDetrForObjectDetection(config)
```
One should prepare the data in COCO detection format, then use
[`~transformers.DetrImageProcessor`] to create `pixel_values`, `pixel_mask` and optional
`labels`, which can then be used to train (or fine-tune) a model. For evaluation, one should first convert the
outputs of the model using one of the postprocessing methods of [`~transformers.DetrImageProcessor`]. These can
be provided to either `CocoEvaluator` which allow you to calculate metrics like
mean Average Precision (mAP).

## DinoDetrModel

[[autodoc]] DinoDetrModel

## DinoDetrForObjectDetection

[[autodoc]] DinoDetrForObjectDetection

## DinoDetrConfig

[[autodoc]] DinoDetrConfig

## DinoDetrImageProcessor

[[autodoc]] DinoDetrImageProcessor

## DinoDetrFeatureExtractor

[[autodoc]] DinoDetrFeatureExtractor

