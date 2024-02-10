from transformers import SiglipForImageClassification


model = SiglipForImageClassification.from_pretrained("google/siglip-base-patch16-256-multilingual")
