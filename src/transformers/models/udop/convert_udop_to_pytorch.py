import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

from transformers import T5Tokenizer, UdopConfig, UdopForConditionalGeneration


def transform(image, image_size=224):
    trans = T.Compose(
        [
            T.Resize([image_size, image_size]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = trans(image)  # copy to make it writeable
    return image


state_dict = torch.load("/Users/nielsrogge/Downloads/udop-unimodel-large-224/pytorch_model.bin", map_location="cpu")

print("Original state dict:")
# for name, param in state_dict.items():
#     print(name, param.shape)

# rename keys
for key, value in state_dict.copy().items():
    val = state_dict.pop(key)
    if "lm_head" not in key:
        key = "udop." + key
    state_dict[key] = val

# create HF model
config = UdopConfig()
model = UdopForConditionalGeneration(config)

# load weights
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
assert missing_keys == ["udop.encoder.embed_patches.proj.weight", "udop.encoder.embed_patches.proj.bias"]
assert unexpected_keys == ["udop.pos_embed"]
print("Looks ok!")

# single forward pass
print("Testing single forward pass..")

question = "Question answering. What is the title?"
tokenizer = T5Tokenizer.from_pretrained("t5-base")
input_ids = tokenizer(question, return_tensors="pt").input_ids
# input_ids = torch.tensor([[101, 102]])

seg_data = torch.tensor([[[0, 0, 0, 0] for _ in range(input_ids.shape[1])]]).float()
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
)
image = Image.open(filepath).convert("RGB")
image = transform(image).unsqueeze(0)
decoder_input_ids = torch.tensor([[101]])

with torch.no_grad():
    outputs = model(input_ids=input_ids, seg_data=seg_data, image=image, decoder_input_ids=decoder_input_ids)

# autoregressive decoding
print("Testing generation...")
model_kwargs = {"seg_data": seg_data, "image": image}
outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)

print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
