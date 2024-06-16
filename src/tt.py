from transformers import AutoModel, AutoProcessor, Kosmos2Config, Kosmos2_5Config, AutoConfig

device = "cuda:0"
repo = "kirp/kosmos2_5" # local
model = AutoModel.from_pretrained(repo, device_map = device)
processor = AutoProcessor.from_pretrained(repo)