from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
device = "cuda:0"
repo = "kirp/kosmos2_5"
dtype = torch.bfloat16
model = AutoModelForVision2Seq.from_pretrained(repo, device_map = device, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained(repo)

path = "/home/yilinjia/MambaOCR/kosmos2_5-convertor/receipt_00008.png"
image = Image.open(path)
prompt = "<ocr>" 
# prompt = "<md>"
inputs = processor(text=prompt, images=image, return_tensors="pt", max_patches=4096)

raw_width, raw_height = image.size
height, width = inputs.pop("height"), inputs.pop("width")
scale_height = raw_height / height
scale_width = raw_width / width

inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
with torch.no_grad():
    generated_text = model.generate(**inputs, max_new_tokens=256)


import re, os
def postprocess(y, scale_height, scale_width, result_path=None):
    y = (
        y.replace("<s>", "")
        .replace("</s>", "")
        .replace("<image>", "")
        .replace("</image>", "")
        .replace(prompt, "")
    )
    info = y
    # pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    # bboxs_raw = re.findall(pattern, y)
    # lines = re.split(pattern, y)[1:]
    # bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    # bboxs = [[int(j) for j in i] for i in bboxs]
    # info = ""
    # for i in range(len(lines)):
    #     box = bboxs[i]
    #     # do we need to convert the size of the box?
    #     # maybe yes
    #     x0, y0, x1, y1 = box
    #     # maybe modify the order
    #     if not (x0 >= x1 or y0 >= y1):
    #         x0 = int(x0 * scale_width)
    #         y0 = int(y0 * scale_height)
    #         x1 = int(x1 * scale_width)
    #         y1 = int(y1 * scale_height)
    #         info += f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{lines[i]}"
    
    if result_path is not None:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        # create and write in utf-8
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(info)
    else:
        print(info)

postprocess(processor.batch_decode(generated_text)[0],scale_height, scale_width)
