from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import UdopImageProcessor, UdopProcessor, UdopTokenizer


processor = UdopImageProcessor()
tokenizer = UdopTokenizer.from_pretrained("/Users/nielsrogge/Downloads/udop-unimodel-large-224/")

processor = UdopProcessor(image_processor=processor, tokenizer=tokenizer)

filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
)
image = Image.open(filepath).convert("RGB")

encoding = processor(images=image, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

print(processor.batch_decode(encoding["input_ids"]))
