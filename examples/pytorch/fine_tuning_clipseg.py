from coco_segmentation_dataset import CocoSegmentation
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPSegConfig
from PIL import Image
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch 
img_folder='/data/coco2017/val2017/'
ann_file='/data/coco2017/annotations/instances_val2017.json'

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

def collate_fn(batch):
    # Image.fromarray(batch["masked_img"][0]).save("test.jpg")
    images = [sample["image"] for sample in batch]
    simple_prompts = [sample["category"] for sample in batch]
    processed = processor(text=simple_prompts, images=images, padding="max_length", return_tensors="pt")
    return processed

# Dataset and dataloader
dataset = CocoSegmentation(img_folder, ann_file)
loader = DataLoader(dataset, batch_size=5, shuffle=True,collate_fn=collate_fn)
epochs = 10

# Load pretrained model
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
# Load a randomly initialized model so we can tune it
model = CLIPSegForImageSegmentation(CLIPSegConfig())
# model.decoder.layers[2].mlp.fc1.weight

# Parameters obtained from: https://github.com/timojl/clipseg/blob/master/experiments/coco.yaml
lr=0.001
T_max = 20000
eta_min = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        
        outputs = model(batch)
        # TODO
        loss = loss_fn(pred, data_y[0].cuda())
        loss.backward()
        optimizer.step()

    
# ####### Inference on CLIPSeg
# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
# from PIL import Image
# import requests
# import torch
# import numpy as np

# # Load image
# url = "https://datasets-server.huggingface.co/assets/huggingface/cats-image/--/image/test/0/image/image.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # Creates the processor and the model
# processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# prompts = ["a cat", "a remote control"]
# inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# # predict
# with torch.no_grad():
#   outputs = model(**inputs)

# # for each prompt
# for idx, prompt in enumerate(prompts):
#     logits = outputs.logits[idx]
#     # pass through a sigmoid to obtain values [0, 1]
#     norm_logits = torch.sigmoid(logits)

#     threshold = 0.5
#     segmented = (norm_logits > threshold) * 255
#     Image.fromarray(segmented.numpy().astype(np.uint8)).save(f"{prompt}.jpg")
# ############################