# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert UDOP checkpoints from the original repository. URL: https://github.com/microsoft/i-Code/tree/main/i-Code-Doc"""


import argparse

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

from transformers import UdopConfig, UdopForConditionalGeneration, UdopImageProcessor, UdopProcessor, UdopTokenizer


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


def get_image():
    filepath = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
    )
    image = Image.open(filepath).convert("RGB")

    return image


def prepare_dummy_inputs(tokenizer, image_processor):
    prompt = "Question answering. What is the name of the company?"
    prompt = "Question answering. In which year is the report made?"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    image = get_image()
    # words, boxes = apply_tesseract(image, lang=None)
    # fmt: off
    words = ['7', 'ITC', 'Limited', 'REPORT', 'AND', 'ACCOUNTS', '2013', 'ITC’s', 'Brands:', 'An', 'Asset', 'for', 'the', 'Nation', 'The', 'consumer', 'needs', 'and', 'aspirations', 'they', 'fulfil,', 'the', 'benefit', 'they', 'generate', 'for', 'millions', 'across', 'ITC’s', 'value', 'chains,', 'the', 'future-ready', 'capabilities', 'that', 'support', 'them,', 'and', 'the', 'value', 'that', 'they', 'create', 'for', 'the', 'country,', 'have', 'made', 'ITC’s', 'brands', 'national', 'assets,', 'adding', 'to', 'India’s', 'competitiveness.', 'It', 'is', 'ITC’s', 'aspiration', 'to', 'be', 'the', 'No', '1', 'FMCG', 'player', 'in', 'the', 'country,', 'driven', 'by', 'its', 'new', 'FMCG', 'businesses.', 'A', 'recent', 'Nielsen', 'report', 'has', 'highlighted', 'that', "ITC's", 'new', 'FMCG', 'businesses', 'are', 'the', 'fastest', 'growing', 'among', 'the', 'top', 'consumer', 'goods', 'companies', 'operating', 'in', 'India.', 'ITC', 'takes', 'justifiable', 'pride', 'that,', 'along', 'with', 'generating', 'economic', 'value,', 'these', 'celebrated', 'Indian', 'brands', 'also', 'drive', 'the', 'creation', 'of', 'larger', 'societal', 'capital', 'through', 'the', 'virtuous', 'cycle', 'of', 'sustainable', 'and', 'inclusive', 'growth.', 'DI', 'WILLS', '*', ';', 'LOVE', 'DELIGHTFULLY', 'SOFT', 'SKIN?', 'aia', 'Ans', 'Source:', 'https://www.industrydocuments.ucsf.edu/docs/snbx0223']
    boxes = [[0, 45, 67, 80], [72, 56, 109, 67], [116, 56, 189, 67], [198, 59, 253, 66], [257, 59, 285, 66], [289, 59, 365, 66], [372, 59, 407, 66], [74, 136, 161, 158], [175, 137, 306, 158], [318, 137, 363, 158], [374, 137, 472, 158], [483, 136, 529, 158], [540, 137, 593, 158], [608, 137, 717, 158], [73, 194, 100, 203], [106, 196, 177, 203], [183, 194, 227, 203], [233, 194, 259, 203], [265, 194, 344, 205], [74, 211, 104, 222], [109, 210, 141, 221], [147, 211, 169, 220], [175, 210, 223, 220], [229, 211, 259, 222], [265, 211, 329, 222], [334, 210, 352, 220], [74, 227, 127, 236], [133, 229, 180, 236], [187, 227, 221, 236], [226, 227, 264, 236], [270, 227, 320, 237], [327, 227, 349, 236], [74, 243, 161, 254], [166, 243, 249, 254], [254, 243, 281, 252], [286, 244, 342, 254], [74, 260, 112, 270], [119, 260, 145, 269], [151, 260, 174, 269], [179, 260, 217, 269], [222, 260, 249, 269], [254, 260, 285, 271], [290, 260, 335, 269], [340, 259, 359, 269], [74, 276, 95, 284], [101, 276, 156, 287], [164, 276, 198, 284], [203, 276, 244, 284], [251, 275, 285, 284], [291, 276, 340, 284], [74, 292, 129, 301], [135, 292, 185, 302], [192, 292, 242, 303], [248, 292, 261, 301], [267, 292, 312, 301], [74, 308, 195, 319], [75, 335, 82, 344], [88, 335, 98, 344], [105, 335, 138, 344], [144, 335, 214, 346], [220, 336, 233, 344], [239, 335, 256, 344], [262, 335, 283, 344], [290, 335, 309, 344], [316, 335, 320, 344], [74, 351, 119, 360], [126, 352, 170, 362], [176, 352, 186, 360], [192, 352, 214, 360], [220, 352, 276, 362], [282, 352, 326, 360], [333, 352, 349, 362], [74, 368, 89, 377], [95, 370, 124, 377], [129, 367, 175, 377], [181, 368, 266, 377], [272, 368, 283, 376], [289, 368, 333, 377], [74, 384, 126, 393], [134, 385, 175, 395], [181, 384, 206, 393], [212, 384, 292, 395], [298, 384, 325, 393], [330, 384, 366, 393], [74, 403, 103, 409], [109, 400, 154, 409], [161, 401, 241, 409], [247, 403, 269, 409], [275, 401, 296, 409], [302, 400, 349, 409], [74, 417, 131, 428], [137, 419, 186, 428], [192, 417, 214, 426], [219, 417, 242, 428], [248, 419, 319, 426], [74, 433, 119, 444], [125, 433, 204, 444], [210, 433, 278, 444], [285, 433, 295, 441], [302, 433, 340, 442], [75, 449, 98, 458], [104, 449, 142, 458], [146, 449, 215, 460], [221, 449, 258, 460], [263, 449, 293, 459], [300, 449, 339, 460], [74, 466, 101, 474], [108, 466, 185, 476], [191, 466, 261, 474], [267, 466, 309, 476], [315, 466, 354, 474], [74, 482, 151, 491], [158, 482, 201, 491], [208, 482, 258, 491], [263, 482, 292, 491], [298, 482, 333, 491], [338, 482, 360, 491], [74, 498, 131, 507], [137, 498, 150, 507], [156, 498, 197, 509], [202, 498, 257, 507], [263, 498, 310, 509], [74, 515, 128, 525], [134, 515, 156, 523], [161, 515, 218, 523], [223, 515, 261, 525], [267, 514, 280, 523], [74, 531, 156, 540], [162, 531, 188, 540], [195, 531, 257, 540], [263, 531, 315, 542], [871, 199, 878, 202], [883, 199, 908, 202], [894, 251, 904, 257], [841, 268, 841, 270], [784, 373, 811, 378], [816, 373, 896, 378], [784, 381, 811, 387], [815, 381, 847, 387], [645, 908, 670, 915], [692, 908, 712, 915], [220, 984, 285, 993], [293, 983, 779, 996]]
    # fmt: on
    text_list = []
    bbox_list = []
    for text, box in zip(words, boxes):
        if text == "":
            continue
        sub_tokens = tokenizer.tokenize(text)
        for sub_token in sub_tokens:
            text_list.append(sub_token)
            bbox_list.append(box)

    input_ids = tokenizer.convert_tokens_to_ids(text_list)

    input_ids = prompt_ids + input_ids
    bbox = [[0, 0, 0, 0]] * len(prompt_ids) + bbox_list

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    original_image = transform(image).unsqueeze(0)
    # verify pixel values
    assert torch.allclose(original_image, pixel_values)

    return torch.tensor(input_ids).unsqueeze(0), torch.tensor(bbox).unsqueeze(0).float(), pixel_values


def convert_udop_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    model_name_to_checkpoint_path = {
        "udop-large": "/Users/nielsrogge/Downloads/udop-unimodel-large-224/pytorch_model.bin",
        "udop-dual-large": "",
    }

    # load original state dict
    checkpoint_path = model_name_to_checkpoint_path[model_name]
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # create HF model
    config = UdopConfig(decoder_start_token_id=0)
    model = UdopForConditionalGeneration(config)
    model.eval()

    # load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    assert missing_keys == ["encoder.embed_patches.proj.weight", "encoder.embed_patches.proj.bias"]
    assert unexpected_keys == ["pos_embed"]

    # prepare dummy inputs
    tokenizer = UdopTokenizer.from_pretrained("t5-base", legacy=True)
    image_processor = UdopImageProcessor()
    processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # input_ids, bbox, image = prepare_dummy_inputs(tokenizer, image_processor)
    prompt = "Question answering. In which year is the report made?"
    encoding = processor(images=get_image(), text=prompt, return_tensors="pt")

    input_ids = encoding.input_ids
    bbox = encoding.bbox.float()
    pixel_values = encoding.pixel_values

    # single forward pass
    print("Testing single forward pass..")
    with torch.no_grad():
        decoder_input_ids = torch.tensor([[101]])
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])
    assert torch.allclose(outputs.logits[0, :3, :3], torch.tensor([[-20.0254, 0.8438, -17.1796]]), atol=1e-4)
    print("Looks ok!")

    # autoregressive decoding
    print("Testing generation...")
    model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)

    print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        tokenizer.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"nielsr/{model_name}")
        processor.push_to_hub(f"nielsr/{model_name}")
        # BIG note here: to save the fast tokenizer files in the repo on the hub, you need to do the following:
        # see https://discuss.huggingface.co/t/convert-slow-xlmrobertatokenizer-to-fast-one/20876


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="udop-large",
        type=str,
        choices=["udop-large", "udop-dual-large"],
        help=("Name of the UDOP model you'd like to convert."),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_udop_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
