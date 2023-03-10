
import torch
import requests

from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from datasets import load_dataset 

dataset = load_dataset("ybelkada/football-dataset", split="train")

torch_device = 0

def prepare_img():
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    url2 = "https://www.connollycove.com/wp-content/uploads/2019/06/temple-bar-dublin-world-famous-irish-pub.jpg"
    im2 = Image.open(requests.get(url2, stream=True).raw)
    return [im, im2]


model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-textcaps-base").to(
    torch_device
)
processor = Pix2StructProcessor.from_pretrained("ybelkada/pix2struct-textcaps-base")
images = prepare_img()

texts = ["A picture of", "An photography of"]

# image only
inputs = processor(images=images, text=texts, return_tensors="pt").to(torch_device)

predictions = model.generate(**inputs)

from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], return_tensors="pt", max_patches=512)
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collator(batch):
    new_batch = {}
    texts = [item["text"] for item in batch]

    text_input = processor.tokenizer(texts, padding=True, return_tensors="pt")

    new_batch["input_ids"] = text_input["input_ids"]
    new_batch["decoder_attention_mask"] = text_input["attention_mask"]
    
    new_batch["attention_mask"] = torch.cat([item["attention_mask"].unsqueeze(0) for item in batch], dim=0)
    new_batch["pixel_embeds"] = torch.cat([item["pixel_embeds"].unsqueeze(0) for item in batch], dim=0)

    return new_batch


    
train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

import torch

optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):

        input_ids = batch.pop("input_ids").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        pixel_values = batch.pop("pixel_embeds").to(device)
        decoder_attention_mask = batch.pop("decoder_attention_mask").to(device)

        outputs = model(labels=input_ids, pixel_embeds=pixel_values, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)

        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if epoch % 10 == 0:
        model.eval()

        with torch.no_grad():
            predictions = model.generate(**inputs)
            print(processor.batch_decode(predictions))
        
        model.train()
