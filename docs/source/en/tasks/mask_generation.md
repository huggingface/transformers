<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mask Generation

Mask generation is the task of generating semantically meaningful masks for an image.
This task is very similar to [image segmentation](semantic_segmentation), but many differences exist. Image segmentation models are trained on labeled datasets and are limited to the classes they have seen during training; they return a set of masks and corresponding classes, given an image.

Mask generation models are trained on large amounts of data and operate in two modes.

- Prompting mode: In this mode, the model takes in an image and a prompt, where a prompt can be a 2D point location (XY coordinates) in the image within an object or a bounding box surrounding an object. In prompting mode, the model only returns the mask over the object
that the prompt is pointing out.
- Segment Everything mode: In segment everything, given an image, the model generates every mask in the image. To do so, a grid of points is generated and overlaid on the image for inference.
- Video Inference: The model accepts a video, and a point or box prompt in a video frame, which is tracked throughout the video. You can get more information on how to do video inference by following [SAM 2 docs](../model_doc/sam2).

Mask generation task is supported by [Segment Anything Model (SAM)](../model_doc/sam) and [Segment Anything Model 2 (SAM2)](../model_doc/sam2), while video inference is supported by [Segment Anything Model 2 (SAM2)](../model_doc/sam2). SAM is a powerful model that consists of a Vision Transformer-based image encoder, a prompt encoder, and a two-way transformer mask decoder. Images and prompts are encoded, and the decoder takes these embeddings and generates valid masks. Meanwhile, SAM 2 extends SAM by adding a memory module to track the masks.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sam.png" alt="SAM Architecture"/>
</div>

SAM serves as a powerful foundation model for segmentation as it has large data coverage. It is trained on
[SA-1B](https://ai.meta.com/datasets/segment-anything/), a dataset with 1 million images and 1.1 billion masks.

In this guide, you will learn how to:

- Infer in segment everything mode with batching,
- Infer in point prompting mode,
- Infer in box prompting mode.

First, let's install `transformers`:

```bash
pip install -q transformers
```

## Mask Generation Pipeline

The easiest way to infer mask generation models is to use the `mask-generation` pipeline.

```python
>>> from transformers import pipeline

>>> checkpoint = "facebook/sam2-hiera-base-plus"
>>> mask_generator = pipeline(model=checkpoint, task="mask-generation")
```

Let's see the image.

```python
from PIL import Image
import requests

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="Example Image"/>
</div>

Let's segment everything. `points-per-batch` enables parallel inference of points in segment everything mode. This enables faster inference, but consumes more memory. Moreover, SAM only enables batching over points and not the images. `pred_iou_thresh` is the IoU confidence threshold where only the masks above that certain threshold are returned.

```python
masks = mask_generator(image, points_per_batch=128, pred_iou_thresh=0.88)
```

The `masks` looks like the following:

```bash
{'masks': [tensor([[False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ..., False, False, False], .. 
 'scores': tensor([0.9874, 0.9793, 0.9780, 0.9776, ... 0.9016])}
```

We can visualize them like this:

```python
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')

for i, mask in enumerate(masks["masks"]):
    plt.imshow(mask, cmap='viridis', alpha=0.1, vmin=0, vmax=1)

plt.axis('off')
plt.show()
```

Below is the original image in grayscale with colorful maps overlaid. Very impressive.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_segmented.png" alt="Visualized"/>
</div>

## Model Inference

### Point Prompting

You can also use the model without the pipeline. To do so, initialize the model and
the processor.

```python
from transformers import SamModel, SamProcessor
from accelerate import Accelerator
import torch
device = Accelerator().device
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
```

To do point prompting, pass the input point to the processor, then take the processor output
and pass it to the model for inference. To post-process the model output, pass the outputs and
`original_sizes` are taken from the processor's initial output. We need to pass these
since the processor resizes the image, and the output needs to be extrapolated.

```python
input_points = [[[2592, 1728]]] # point location of the bee

inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
```

We can visualize the three masks in the `masks` output.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
mask_list = [masks[0][0][0].numpy(), masks[0][0][1].numpy(), masks[0][0][2].numpy()]

for i, mask in enumerate(mask_list, start=1):
    overlayed_image = np.array(image).copy()

    overlayed_image[:,:,0] = np.where(mask == 1, 255, overlayed_image[:,:,0])
    overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
    overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])
    
    axes[i].imshow(overlayed_image)
    axes[i].set_title(f'Mask {i}')
for ax in axes:
    ax.axis('off')

plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/masks.png" alt="Visualized"/>
</div>

### Box Prompting

You can also do box prompting in a similar fashion to point prompting. You can simply pass the input box in the format of a list
`[x_min, y_min, x_max, y_max]` format along with the image to the `processor`. Take the processor output and directly pass it
to the model, then post-process the output again.

```python
# bounding box around the bee
box = [2350, 1600, 2850, 2100]

inputs = processor(
        image,
        input_boxes=[[[box]]],
        return_tensors="pt"
    ).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

mask = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
)[0][0][0].numpy()
```

You can visualize the bounding box around the bee as shown below.

```python
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(image)

rectangle = patches.Rectangle((2350, 1600), 500, 500, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rectangle)
ax.axis("off")
plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/bbox.png" alt="Visualized Bbox"/>
</div>

You can see the inference output below.

```python
fig, ax = plt.subplots()
ax.imshow(image)
ax.imshow(mask, cmap='viridis', alpha=0.4)

ax.axis("off")
plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/box_inference.png" alt="Visualized Inference"/>
</div>

## Fine-tuning for Mask Generation

We will fine-tune SAM2.1 on small part of MicroMat dataset for image matting. We need to install the [monai](https://github.com/Project-MONAI/MONAI) library to use DICE loss, and [trackio](https://huggingface.co/docs/trackio/index) (>=0.14.0) for logging the masks during training.

```bash
pip install -q datasets monai "trackio[gpu]>=0.14.0"
``` 
We can now load our dataset and take a look.

```python
from datasets import load_dataset

dataset = load_dataset("merve/MicroMat-mini", split="train")
dataset
# Dataset({
#    features: ['image', 'mask', 'prompt', 'image_id', 'object_id', 'sample_idx', 'granularity', 
# 'image_path', 'mask_path', 'prompt_path'],  num_rows: 94
#})
```

We need image, mask and prompt columns. We split for train and test.

```python
dataset = dataset.train_test_split(test_size=0.1)
train_ds = dataset["train"]
val_ds = dataset["test"]
```

Let's take a look at a sample.

```python
train_ds[0]
```

```
 {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2040x1356>,
 'mask': <PIL.PngImagePlugin.PngImageFile image mode=L size=2040x1356>,
 'prompt': '{"point": [[137, 1165, 1], [77, 1273, 0], [58, 1351, 0]], "bbox": [0, 701, 251, 1356]}',
 'image_id': '0034',
 'object_id': '34',
 'sample_idx': 1,
 'granularity': 'fine',
 'image_path': '/content/MicroMat-mini/img/0034.png',
 'mask_path': '/content/MicroMat-mini/mask/0034_34.png',
 'prompt_path': '/content/MicroMat-mini/prompt/0034_34.json'}
```

Prompts are string of dictionaries, so you can get the bounding boxes as shown below.

```python
import json

json.loads(train_ds["prompt"][0])["bbox"]
# [0, 701, 251, 1356]
```

Visualize an example image, prompt and mask.

```python
import matplotlib.pyplot as plt
import numpy as np

def show_mask(mask, ax):
    color = np.array([0.12, 0.56, 1.0, 0.6])
    mask = np.array(mask)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, 4)
    ax.imshow(mask_image)
    x0, y0, x1, y1 = eval(train_ds["prompt"][0])["bbox"]
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                      fill=False, edgecolor="lime", linewidth=2))

example = train_ds[0]
image = np.array(example["image"])
ground_truth_mask = np.array(example["mask"])

fig, ax = plt.subplots()
ax.imshow(image)
show_mask(ground_truth_mask, ax)
ax.set_title("Ground truth mask")
ax.set_axis_off()

plt.show() 
```

Now we can define our dataset for loading the data. SAMDataset wraps our dataset and formats each sample the way the SAM processor expects. So instead of raw images and masks, you get processed images, bounding boxes, and ground-truth masks ready for training.

By default, processor resizes images, so on top of images and masks, it also returns original sizes. We also need to binarize the mask as it has values [0, 255].

```python
from torch.utils.data import Dataset
import torch

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    prompt = eval(item["prompt"])["bbox"]
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    inputs["ground_truth_mask"] = (np.array(item["mask"]) > 0).astype(np.float32)
    inputs["original_image_size"] = torch.tensor(image.size[::-1])


    return inputs
```

We can initialize the processor and the dataset with it.

```python
from transformers import Sam2Processor

processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
train_dataset = SAMDataset(dataset=train_ds, processor=processor)
``` 

We need to define a data collator that will turn varying size of ground truth masks to batches of reshaped masks in same shape. We reshape them using nearest neighbor interpolation. We also make batched tensors for rest of the elements in the batch. If your masks are all of same size, feel free to skip this step.

```python
import torch.nn.functional as F

def collate_fn(batch, target_hw=(256, 256)):

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    original_sizes = torch.stack([item["original_sizes"] for item in batch])
    input_boxes = torch.cat([item["input_boxes"] for item in batch], dim=0)
    ground_truth_masks = torch.cat([
        F.interpolate(
            torch.as_tensor(x["ground_truth_mask"]).unsqueeze(0).unsqueeze(0).float(),
            size=(256, 256),
            mode="nearest"
        )
        for x in batch
    ], dim=0).long()

    return {
        "pixel_values": pixel_values,
        "original_sizes": original_sizes,
        "input_boxes": input_boxes,
        "ground_truth_mask": ground_truth_masks,
        "original_image_size": torch.stack([item["original_image_size"] for item in batch]),
    }

from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
)
```

Let's take a look at what the data loader yields.

```python
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

# pixel_values torch.Size([4, 3, 1024, 1024])
# original_sizes torch.Size([4, 1, 2])
# input_boxes torch.Size([4, 1, 4])
# ground_truth_mask torch.Size([4, 1, 256, 256])
#original_image_size torch.Size([4, 2])
```

We will now load the model and freeze the vision and the prompt encoder to only train the mask decoder.

```python
from transformers import Sam2Model

model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small")

for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
``` 

We can now define the optimizer and the loss function.
```python 
from torch.optim import Adam
import monai

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
```

Let's see how the model performs before training.

```python
import matplotlib.pyplot as plt

item = val_ds[1]
img = item["image"]
bbox = json.loads(item["prompt"])["bbox"]
inputs = processor(images=img, input_boxes=[[bbox]], return_tensors="pt").to(model.device)

with torch.no_grad():
  outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
preds = masks.squeeze(0)
mask = (preds[0] > 0).cpu().numpy()

overlay = np.asarray(img, dtype=np.uint8).copy()
overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0], dtype=np.float32)

plt.imshow(overlay)
plt.axis("off")
plt.show()
```

![SAM2 result after training](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sam2_before_training.png)

We need to log our predictions to trackio so we can monitor the model improvement in the middle of the training.

```python
from PIL import Image
import trackio
import json


@torch.no_grad()
def predict_fn(img, bbox):

  inputs = processor(images=img, input_boxes=[[bbox]], return_tensors="pt").to(model.device)

  with torch.no_grad():
      outputs = model(**inputs)

  masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
  return masks

def log_eval_masks_trackio(dataset, indices, step, predict_fn,  project=None, sample_cap=8):
    logs = {"eval/step": int(step)}
    for idx in indices[:sample_cap]:
        item = dataset[idx] 
        img = item["image"]
        bbox = json.loads(item["prompt"])["bbox"]
        preds = predict_fn(img, bbox)
        preds = preds.squeeze(0)
        mask = (preds[0] > 0).cpu().numpy()  

        overlay = np.asarray(img, dtype=np.uint8).copy()
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
        logs[f"{idx}/overlay"] = trackio.Image(overlay, caption="overlay")
        
    trackio.log(logs)
```

We can now write our training loop and train!

Notice how we log our loss and evaluation masks with trackio.

```python
from tqdm import tqdm
from statistics import mean
import trackio
import torch

num_epochs = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
trackio.init(project="mask-eval")
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks)

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
      epoch_losses.append(loss.item())
      
    log_eval_masks_trackio(dataset=val_ds, indices=[0, 3, 6, 9], step=epoch, predict_fn=predict_fn, project="mask-eval")
    print(f'Epoch: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    trackio.log({"loss": mean(epoch_losses)})

trackio.finish()
```

Let's put the trained model to test.

```python
import matplotlib.pyplot as plt

item = val_ds[1]
img = item["image"]
bbox = json.loads(item["prompt"])["bbox"]

inputs = processor(images=img, input_boxes=[[bbox]], return_tensors="pt").to(model.device)

with torch.no_grad():
  outputs = model(**inputs)

preds = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

preds = preds.squeeze(0)
mask = (preds[0] > 0).cpu().numpy()

overlay = np.asarray(img, dtype=np.uint8).copy()
overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0], dtype=np.float32)

plt.imshow(overlay)
plt.axis("off")
plt.show()
```

Great improvement after only training for 20 epochs on a small dataset!

![SAM2 result after training](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sam2_after_training.png)
