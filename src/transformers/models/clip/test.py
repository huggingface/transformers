from transformers import CLIPForImageClassification


model = CLIPForImageClassification.from_pretrained("openai/clip-vit-base-patch16")
