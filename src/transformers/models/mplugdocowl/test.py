from PIL import Image

# image = Image.open("/raid/dana/test_image.png")
#image = Image.open("/raid/dana/examples_Rebecca_(1939_poster)_Small.jpeg")
image = Image.open('/raid/dana/fflw0023_1.png')
# query = "<image>Recognize text in the image."
# query = "<image>What's the value of the Very well bar in the 65+ age group? Answer the question with detailed explanation."
query = "<image>Parse texts in the image."
#query = "<image>What is the name of the movie in the poster? Provide detailed explanation."
output = processor(images=image, text=query)
breakpoint()
device = torch.device("cuda:0")
output.to(device)
model.to(device)
torch.set_default_dtype(torch.float16)
# with torch.inference_mode():
# outputs = model(input_ids=output['input_ids'], pixel_values = output['pixel_values'],attention_mask=output['attention_mask'], patch_positions=output['patch_positions'])
try:
    tokens = model.generate(output["input_ids"], pixel_values=output["pixel_values"], max_new_tokens=512)
except AttributeError as e:
    raise (e)

breakpoint()