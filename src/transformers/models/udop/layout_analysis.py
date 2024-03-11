import torch
from huggingface_hub import hf_hub_download

from transformers import UdopForConditionalGeneration, UdopTokenizer


tokenizer = UdopTokenizer.from_pretrained("/Users/nielsrogge/Documents/UDOP/test")
model = UdopForConditionalGeneration.from_pretrained("/Users/nielsrogge/Documents/UDOP/test")

filepath = hf_hub_download(
    repo_id="nielsr/test-image", filename="input_ids_udop_512_layout_analysis.pt", repo_type="dataset"
)
input_ids = torch.load(filepath)

filepath = hf_hub_download(
    repo_id="nielsr/test-image", filename="seg_data_udop_512_layout_analysis.pt", repo_type="dataset"
)
bbox = torch.load(filepath)

filepath = hf_hub_download(repo_id="nielsr/test-image", filename="pixel_values_udop_512.pt", repo_type="dataset")
pixel_values = torch.load(filepath)

print("Special tokens:", tokenizer.additional_special_tokens)


# Inference from the inputs
output_ids = model.generate(
    input_ids,
    bbox=bbox,
    pixel_values=pixel_values,
    use_cache=True,
    decoder_start_token_id=None,
    num_beams=1,
    max_length=10,
)

print("Output ids:", output_ids)

output_text = tokenizer.decode(output_ids[0][1:-1])

print(output_text)

print(tokenizer.convert_tokens_to_ids("<loc_130>"))
