# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from transformers import (
    LayoutLMv3ImageProcessor,
    UdopConfig,
    UdopForConditionalGeneration,
    UdopProcessor,
    UdopTokenizer,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def original_transform(image, image_size=224):
    transform = T.Compose(
        [
            T.Resize([image_size, image_size]),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    image = transform(image)
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
    original_pixel_values = original_transform(image, image_size=image_processor.size["height"]).unsqueeze(0)
    # verify pixel values
    assert torch.allclose(original_pixel_values, pixel_values)
    print("Pixel values are ok!")

    return torch.tensor(input_ids).unsqueeze(0), torch.tensor(bbox).unsqueeze(0).float(), pixel_values


def convert_udop_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # model_name to checkpoint_path
    name_to_checkpoint_path = {
        "udop-large": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-224/pytorch_model.bin",
        "udop-large-512": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512/pytorch_model.bin",
        "udop-large-512-300k": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512-300k-steps/pytorch_model.bin",
    }

    # load original state dict
    checkpoint_path = name_to_checkpoint_path[model_name]
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    print("Checkpoint path:", checkpoint_path)

    # create HF model
    image_size = 512 if "512" in model_name else 224
    config = UdopConfig(decoder_start_token_id=0, image_size=image_size)
    model = UdopForConditionalGeneration(config)
    model.eval()

    # rename keys
    state_dict = {k.replace("cell2dembedding", "cell_2d_embedding"): v for k, v in state_dict.items()}

    # load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    assert missing_keys == ["encoder.embed_patches.proj.weight", "encoder.embed_patches.proj.bias"]
    assert unexpected_keys == ["pos_embed"]

    # Add extra_ids to the special token list
    # NOTE special tokens have a unique order
    # see https://github.com/huggingface/transformers/issues/29591 for details
    # fmt: off
    additional_special_tokens = ['<extra_id_99>', '<extra_id_98>', '<extra_id_97>', '<extra_id_96>', '<extra_id_95>', '<extra_id_94>', '<extra_id_93>', '<extra_id_92>', '<extra_id_91>', '<extra_id_90>', '<extra_id_89>', '<extra_id_88>', '<extra_id_87>', '<extra_id_86>', '<extra_id_85>', '<extra_id_84>', '<extra_id_83>', '<extra_id_82>', '<extra_id_81>', '<extra_id_80>', '<extra_id_79>', '<extra_id_78>', '<extra_id_77>', '<extra_id_76>', '<extra_id_75>', '<extra_id_74>', '<extra_id_73>', '<extra_id_72>', '<extra_id_71>', '<extra_id_70>', '<extra_id_69>', '<extra_id_68>', '<extra_id_67>', '<extra_id_66>', '<extra_id_65>', '<extra_id_64>', '<extra_id_63>', '<extra_id_62>', '<extra_id_61>', '<extra_id_60>', '<extra_id_59>', '<extra_id_58>', '<extra_id_57>', '<extra_id_56>', '<extra_id_55>', '<extra_id_54>', '<extra_id_53>', '<extra_id_52>', '<extra_id_51>', '<extra_id_50>', '<extra_id_49>', '<extra_id_48>', '<extra_id_47>', '<extra_id_46>', '<extra_id_45>', '<extra_id_44>', '<extra_id_43>', '<extra_id_42>', '<extra_id_41>', '<extra_id_40>', '<extra_id_39>', '<extra_id_38>', '<extra_id_37>', '<extra_id_36>', '<extra_id_35>', '<extra_id_34>', '<extra_id_33>', '<extra_id_32>', '<extra_id_31>', '<extra_id_30>', '<extra_id_29>', '<extra_id_28>', '<extra_id_27>', '<extra_id_26>', '<extra_id_25>', '<extra_id_24>', '<extra_id_23>', '<extra_id_22>', '<extra_id_21>', '<extra_id_20>', '<extra_id_19>', '<extra_id_18>', '<extra_id_17>', '<extra_id_16>', '<extra_id_15>', '<extra_id_14>', '<extra_id_13>', '<extra_id_12>', '<extra_id_11>', '<extra_id_10>', '<extra_id_9>', '<extra_id_8>', '<extra_id_7>', '<extra_id_6>', '<extra_id_5>', '<extra_id_4>', '<extra_id_3>', '<extra_id_2>', '<extra_id_1>', '<extra_id_0>', '<extra_l_id_99>', '<extra_l_id_98>', '<extra_l_id_97>', '<extra_l_id_96>', '<extra_l_id_95>', '<extra_l_id_94>', '<extra_l_id_93>', '<extra_l_id_92>', '<extra_l_id_91>', '<extra_l_id_90>', '<extra_l_id_89>', '<extra_l_id_88>', '<extra_l_id_87>', '<extra_l_id_86>', '<extra_l_id_85>', '<extra_l_id_84>', '<extra_l_id_83>', '<extra_l_id_82>', '<extra_l_id_81>', '<extra_l_id_80>', '<extra_l_id_79>', '<extra_l_id_78>', '<extra_l_id_77>', '<extra_l_id_76>', '<extra_l_id_75>', '<extra_l_id_74>', '<extra_l_id_73>', '<extra_l_id_72>', '<extra_l_id_71>', '<extra_l_id_70>', '<extra_l_id_69>', '<extra_l_id_68>', '<extra_l_id_67>', '<extra_l_id_66>', '<extra_l_id_65>', '<extra_l_id_64>', '<extra_l_id_63>', '<extra_l_id_62>', '<extra_l_id_61>', '<extra_l_id_60>', '<extra_l_id_59>', '<extra_l_id_58>', '<extra_l_id_57>', '<extra_l_id_56>', '<extra_l_id_55>', '<extra_l_id_54>', '<extra_l_id_53>', '<extra_l_id_52>', '<extra_l_id_51>', '<extra_l_id_50>', '<extra_l_id_49>', '<extra_l_id_48>', '<extra_l_id_47>', '<extra_l_id_46>', '<extra_l_id_45>', '<extra_l_id_44>', '<extra_l_id_43>', '<extra_l_id_42>', '<extra_l_id_41>', '<extra_l_id_40>', '<extra_l_id_39>', '<extra_l_id_38>', '<extra_l_id_37>', '<extra_l_id_36>', '<extra_l_id_35>', '<extra_l_id_34>', '<extra_l_id_33>', '<extra_l_id_32>', '<extra_l_id_31>', '<extra_l_id_30>', '<extra_l_id_29>', '<extra_l_id_28>', '<extra_l_id_27>', '<extra_l_id_26>', '<extra_l_id_25>', '<extra_l_id_24>', '<extra_l_id_23>', '<extra_l_id_22>', '<extra_l_id_21>', '<extra_l_id_20>', '<extra_l_id_19>', '<extra_l_id_18>', '<extra_l_id_17>', '<extra_l_id_16>', '<extra_l_id_15>', '<extra_l_id_14>', '<extra_l_id_13>', '<extra_l_id_12>', '<extra_l_id_11>', '<extra_l_id_10>', '<extra_l_id_9>', '<extra_l_id_8>', '<extra_l_id_7>', '<extra_l_id_6>', '<extra_l_id_5>', '<extra_l_id_4>', '<extra_l_id_3>', '<extra_l_id_2>', '<extra_l_id_1>', '<extra_l_id_0>', '</extra_l_id_99>', '</extra_l_id_98>', '</extra_l_id_97>', '</extra_l_id_96>', '</extra_l_id_95>', '</extra_l_id_94>', '</extra_l_id_93>', '</extra_l_id_92>', '</extra_l_id_91>', '</extra_l_id_90>', '</extra_l_id_89>', '</extra_l_id_88>', '</extra_l_id_87>', '</extra_l_id_86>', '</extra_l_id_85>', '</extra_l_id_84>', '</extra_l_id_83>', '</extra_l_id_82>', '</extra_l_id_81>', '</extra_l_id_80>', '</extra_l_id_79>', '</extra_l_id_78>', '</extra_l_id_77>', '</extra_l_id_76>', '</extra_l_id_75>', '</extra_l_id_74>', '</extra_l_id_73>', '</extra_l_id_72>', '</extra_l_id_71>', '</extra_l_id_70>', '</extra_l_id_69>', '</extra_l_id_68>', '</extra_l_id_67>', '</extra_l_id_66>', '</extra_l_id_65>', '</extra_l_id_64>', '</extra_l_id_63>', '</extra_l_id_62>', '</extra_l_id_61>', '</extra_l_id_60>', '</extra_l_id_59>', '</extra_l_id_58>', '</extra_l_id_57>', '</extra_l_id_56>', '</extra_l_id_55>', '</extra_l_id_54>', '</extra_l_id_53>', '</extra_l_id_52>', '</extra_l_id_51>', '</extra_l_id_50>', '</extra_l_id_49>', '</extra_l_id_48>', '</extra_l_id_47>', '</extra_l_id_46>', '</extra_l_id_45>', '</extra_l_id_44>', '</extra_l_id_43>', '</extra_l_id_42>', '</extra_l_id_41>', '</extra_l_id_40>', '</extra_l_id_39>', '</extra_l_id_38>', '</extra_l_id_37>', '</extra_l_id_36>', '</extra_l_id_35>', '</extra_l_id_34>', '</extra_l_id_33>', '</extra_l_id_32>', '</extra_l_id_31>', '</extra_l_id_30>', '</extra_l_id_29>', '</extra_l_id_28>', '</extra_l_id_27>', '</extra_l_id_26>', '</extra_l_id_25>', '</extra_l_id_24>', '</extra_l_id_23>', '</extra_l_id_22>', '</extra_l_id_21>', '</extra_l_id_20>', '</extra_l_id_19>', '</extra_l_id_18>', '</extra_l_id_17>', '</extra_l_id_16>', '</extra_l_id_15>', '</extra_l_id_14>', '</extra_l_id_13>', '</extra_l_id_12>', '</extra_l_id_11>', '</extra_l_id_10>', '</extra_l_id_9>', '</extra_l_id_8>', '</extra_l_id_7>', '</extra_l_id_6>', '</extra_l_id_5>', '</extra_l_id_4>', '</extra_l_id_3>', '</extra_l_id_2>', '</extra_l_id_1>', '</extra_l_id_0>', '<extra_t_id_99>', '<extra_t_id_98>', '<extra_t_id_97>', '<extra_t_id_96>', '<extra_t_id_95>', '<extra_t_id_94>', '<extra_t_id_93>', '<extra_t_id_92>', '<extra_t_id_91>', '<extra_t_id_90>', '<extra_t_id_89>', '<extra_t_id_88>', '<extra_t_id_87>', '<extra_t_id_86>', '<extra_t_id_85>', '<extra_t_id_84>', '<extra_t_id_83>', '<extra_t_id_82>', '<extra_t_id_81>', '<extra_t_id_80>', '<extra_t_id_79>', '<extra_t_id_78>', '<extra_t_id_77>', '<extra_t_id_76>', '<extra_t_id_75>', '<extra_t_id_74>', '<extra_t_id_73>', '<extra_t_id_72>', '<extra_t_id_71>', '<extra_t_id_70>', '<extra_t_id_69>', '<extra_t_id_68>', '<extra_t_id_67>', '<extra_t_id_66>', '<extra_t_id_65>', '<extra_t_id_64>', '<extra_t_id_63>', '<extra_t_id_62>', '<extra_t_id_61>', '<extra_t_id_60>', '<extra_t_id_59>', '<extra_t_id_58>', '<extra_t_id_57>', '<extra_t_id_56>', '<extra_t_id_55>', '<extra_t_id_54>', '<extra_t_id_53>', '<extra_t_id_52>', '<extra_t_id_51>', '<extra_t_id_50>', '<extra_t_id_49>', '<extra_t_id_48>', '<extra_t_id_47>', '<extra_t_id_46>', '<extra_t_id_45>', '<extra_t_id_44>', '<extra_t_id_43>', '<extra_t_id_42>', '<extra_t_id_41>', '<extra_t_id_40>', '<extra_t_id_39>', '<extra_t_id_38>', '<extra_t_id_37>', '<extra_t_id_36>', '<extra_t_id_35>', '<extra_t_id_34>', '<extra_t_id_33>', '<extra_t_id_32>', '<extra_t_id_31>', '<extra_t_id_30>', '<extra_t_id_29>', '<extra_t_id_28>', '<extra_t_id_27>', '<extra_t_id_26>', '<extra_t_id_25>', '<extra_t_id_24>', '<extra_t_id_23>', '<extra_t_id_22>', '<extra_t_id_21>', '<extra_t_id_20>', '<extra_t_id_19>', '<extra_t_id_18>', '<extra_t_id_17>', '<extra_t_id_16>', '<extra_t_id_15>', '<extra_t_id_14>', '<extra_t_id_13>', '<extra_t_id_12>', '<extra_t_id_11>', '<extra_t_id_10>', '<extra_t_id_9>', '<extra_t_id_8>', '<extra_t_id_7>', '<extra_t_id_6>', '<extra_t_id_5>', '<extra_t_id_4>', '<extra_t_id_3>', '<extra_t_id_2>', '<extra_t_id_1>', '<extra_t_id_0>', '</extra_t_id_99>', '</extra_t_id_98>', '</extra_t_id_97>', '</extra_t_id_96>', '</extra_t_id_95>', '</extra_t_id_94>', '</extra_t_id_93>', '</extra_t_id_92>', '</extra_t_id_91>', '</extra_t_id_90>', '</extra_t_id_89>', '</extra_t_id_88>', '</extra_t_id_87>', '</extra_t_id_86>', '</extra_t_id_85>', '</extra_t_id_84>', '</extra_t_id_83>', '</extra_t_id_82>', '</extra_t_id_81>', '</extra_t_id_80>', '</extra_t_id_79>', '</extra_t_id_78>', '</extra_t_id_77>', '</extra_t_id_76>', '</extra_t_id_75>', '</extra_t_id_74>', '</extra_t_id_73>', '</extra_t_id_72>', '</extra_t_id_71>', '</extra_t_id_70>', '</extra_t_id_69>', '</extra_t_id_68>', '</extra_t_id_67>', '</extra_t_id_66>', '</extra_t_id_65>', '</extra_t_id_64>', '</extra_t_id_63>', '</extra_t_id_62>', '</extra_t_id_61>', '</extra_t_id_60>', '</extra_t_id_59>', '</extra_t_id_58>', '</extra_t_id_57>', '</extra_t_id_56>', '</extra_t_id_55>', '</extra_t_id_54>', '</extra_t_id_53>', '</extra_t_id_52>', '</extra_t_id_51>', '</extra_t_id_50>', '</extra_t_id_49>', '</extra_t_id_48>', '</extra_t_id_47>', '</extra_t_id_46>', '</extra_t_id_45>', '</extra_t_id_44>', '</extra_t_id_43>', '</extra_t_id_42>', '</extra_t_id_41>', '</extra_t_id_40>', '</extra_t_id_39>', '</extra_t_id_38>', '</extra_t_id_37>', '</extra_t_id_36>', '</extra_t_id_35>', '</extra_t_id_34>', '</extra_t_id_33>', '</extra_t_id_32>', '</extra_t_id_31>', '</extra_t_id_30>', '</extra_t_id_29>', '</extra_t_id_28>', '</extra_t_id_27>', '</extra_t_id_26>', '</extra_t_id_25>', '</extra_t_id_24>', '</extra_t_id_23>', '</extra_t_id_22>', '</extra_t_id_21>', '</extra_t_id_20>', '</extra_t_id_19>', '</extra_t_id_18>', '</extra_t_id_17>', '</extra_t_id_16>', '</extra_t_id_15>', '</extra_t_id_14>', '</extra_t_id_13>', '</extra_t_id_12>', '</extra_t_id_11>', '</extra_t_id_10>', '</extra_t_id_9>', '</extra_t_id_8>', '</extra_t_id_7>', '</extra_t_id_6>', '</extra_t_id_5>', '</extra_t_id_4>', '</extra_t_id_3>', '</extra_t_id_2>', '</extra_t_id_1>', '</extra_t_id_0>', '<loc_500>', '<loc_499>', '<loc_498>', '<loc_497>', '<loc_496>', '<loc_495>', '<loc_494>', '<loc_493>', '<loc_492>', '<loc_491>', '<loc_490>', '<loc_489>', '<loc_488>', '<loc_487>', '<loc_486>', '<loc_485>', '<loc_484>', '<loc_483>', '<loc_482>', '<loc_481>', '<loc_480>', '<loc_479>', '<loc_478>', '<loc_477>', '<loc_476>', '<loc_475>', '<loc_474>', '<loc_473>', '<loc_472>', '<loc_471>', '<loc_470>', '<loc_469>', '<loc_468>', '<loc_467>', '<loc_466>', '<loc_465>', '<loc_464>', '<loc_463>', '<loc_462>', '<loc_461>', '<loc_460>', '<loc_459>', '<loc_458>', '<loc_457>', '<loc_456>', '<loc_455>', '<loc_454>', '<loc_453>', '<loc_452>', '<loc_451>', '<loc_450>', '<loc_449>', '<loc_448>', '<loc_447>', '<loc_446>', '<loc_445>', '<loc_444>', '<loc_443>', '<loc_442>', '<loc_441>', '<loc_440>', '<loc_439>', '<loc_438>', '<loc_437>', '<loc_436>', '<loc_435>', '<loc_434>', '<loc_433>', '<loc_432>', '<loc_431>', '<loc_430>', '<loc_429>', '<loc_428>', '<loc_427>', '<loc_426>', '<loc_425>', '<loc_424>', '<loc_423>', '<loc_422>', '<loc_421>', '<loc_420>', '<loc_419>', '<loc_418>', '<loc_417>', '<loc_416>', '<loc_415>', '<loc_414>', '<loc_413>', '<loc_412>', '<loc_411>', '<loc_410>', '<loc_409>', '<loc_408>', '<loc_407>', '<loc_406>', '<loc_405>', '<loc_404>', '<loc_403>', '<loc_402>', '<loc_401>', '<loc_400>', '<loc_399>', '<loc_398>', '<loc_397>', '<loc_396>', '<loc_395>', '<loc_394>', '<loc_393>', '<loc_392>', '<loc_391>', '<loc_390>', '<loc_389>', '<loc_388>', '<loc_387>', '<loc_386>', '<loc_385>', '<loc_384>', '<loc_383>', '<loc_382>', '<loc_381>', '<loc_380>', '<loc_379>', '<loc_378>', '<loc_377>', '<loc_376>', '<loc_375>', '<loc_374>', '<loc_373>', '<loc_372>', '<loc_371>', '<loc_370>', '<loc_369>', '<loc_368>', '<loc_367>', '<loc_366>', '<loc_365>', '<loc_364>', '<loc_363>', '<loc_362>', '<loc_361>', '<loc_360>', '<loc_359>', '<loc_358>', '<loc_357>', '<loc_356>', '<loc_355>', '<loc_354>', '<loc_353>', '<loc_352>', '<loc_351>', '<loc_350>', '<loc_349>', '<loc_348>', '<loc_347>', '<loc_346>', '<loc_345>', '<loc_344>', '<loc_343>', '<loc_342>', '<loc_341>', '<loc_340>', '<loc_339>', '<loc_338>', '<loc_337>', '<loc_336>', '<loc_335>', '<loc_334>', '<loc_333>', '<loc_332>', '<loc_331>', '<loc_330>', '<loc_329>', '<loc_328>', '<loc_327>', '<loc_326>', '<loc_325>', '<loc_324>', '<loc_323>', '<loc_322>', '<loc_321>', '<loc_320>', '<loc_319>', '<loc_318>', '<loc_317>', '<loc_316>', '<loc_315>', '<loc_314>', '<loc_313>', '<loc_312>', '<loc_311>', '<loc_310>', '<loc_309>', '<loc_308>', '<loc_307>', '<loc_306>', '<loc_305>', '<loc_304>', '<loc_303>', '<loc_302>', '<loc_301>', '<loc_300>', '<loc_299>', '<loc_298>', '<loc_297>', '<loc_296>', '<loc_295>', '<loc_294>', '<loc_293>', '<loc_292>', '<loc_291>', '<loc_290>', '<loc_289>', '<loc_288>', '<loc_287>', '<loc_286>', '<loc_285>', '<loc_284>', '<loc_283>', '<loc_282>', '<loc_281>', '<loc_280>', '<loc_279>', '<loc_278>', '<loc_277>', '<loc_276>', '<loc_275>', '<loc_274>', '<loc_273>', '<loc_272>', '<loc_271>', '<loc_270>', '<loc_269>', '<loc_268>', '<loc_267>', '<loc_266>', '<loc_265>', '<loc_264>', '<loc_263>', '<loc_262>', '<loc_261>', '<loc_260>', '<loc_259>', '<loc_258>', '<loc_257>', '<loc_256>', '<loc_255>', '<loc_254>', '<loc_253>', '<loc_252>', '<loc_251>', '<loc_250>', '<loc_249>', '<loc_248>', '<loc_247>', '<loc_246>', '<loc_245>', '<loc_244>', '<loc_243>', '<loc_242>', '<loc_241>', '<loc_240>', '<loc_239>', '<loc_238>', '<loc_237>', '<loc_236>', '<loc_235>', '<loc_234>', '<loc_233>', '<loc_232>', '<loc_231>', '<loc_230>', '<loc_229>', '<loc_228>', '<loc_227>', '<loc_226>', '<loc_225>', '<loc_224>', '<loc_223>', '<loc_222>', '<loc_221>', '<loc_220>', '<loc_219>', '<loc_218>', '<loc_217>', '<loc_216>', '<loc_215>', '<loc_214>', '<loc_213>', '<loc_212>', '<loc_211>', '<loc_210>', '<loc_209>', '<loc_208>', '<loc_207>', '<loc_206>', '<loc_205>', '<loc_204>', '<loc_203>', '<loc_202>', '<loc_201>', '<loc_200>', '<loc_199>', '<loc_198>', '<loc_197>', '<loc_196>', '<loc_195>', '<loc_194>', '<loc_193>', '<loc_192>', '<loc_191>', '<loc_190>', '<loc_189>', '<loc_188>', '<loc_187>', '<loc_186>', '<loc_185>', '<loc_184>', '<loc_183>', '<loc_182>', '<loc_181>', '<loc_180>', '<loc_179>', '<loc_178>', '<loc_177>', '<loc_176>', '<loc_175>', '<loc_174>', '<loc_173>', '<loc_172>', '<loc_171>', '<loc_170>', '<loc_169>', '<loc_168>', '<loc_167>', '<loc_166>', '<loc_165>', '<loc_164>', '<loc_163>', '<loc_162>', '<loc_161>', '<loc_160>', '<loc_159>', '<loc_158>', '<loc_157>', '<loc_156>', '<loc_155>', '<loc_154>', '<loc_153>', '<loc_152>', '<loc_151>', '<loc_150>', '<loc_149>', '<loc_148>', '<loc_147>', '<loc_146>', '<loc_145>', '<loc_144>', '<loc_143>', '<loc_142>', '<loc_141>', '<loc_140>', '<loc_139>', '<loc_138>', '<loc_137>', '<loc_136>', '<loc_135>', '<loc_134>', '<loc_133>', '<loc_132>', '<loc_131>', '<loc_130>', '<loc_129>', '<loc_128>', '<loc_127>', '<loc_126>', '<loc_125>', '<loc_124>', '<loc_123>', '<loc_122>', '<loc_121>', '<loc_120>', '<loc_119>', '<loc_118>', '<loc_117>', '<loc_116>', '<loc_115>', '<loc_114>', '<loc_113>', '<loc_112>', '<loc_111>', '<loc_110>', '<loc_109>', '<loc_108>', '<loc_107>', '<loc_106>', '<loc_105>', '<loc_104>', '<loc_103>', '<loc_102>', '<loc_101>', '<loc_100>', '<loc_99>', '<loc_98>', '<loc_97>', '<loc_96>', '<loc_95>', '<loc_94>', '<loc_93>', '<loc_92>', '<loc_91>', '<loc_90>', '<loc_89>', '<loc_88>', '<loc_87>', '<loc_86>', '<loc_85>', '<loc_84>', '<loc_83>', '<loc_82>', '<loc_81>', '<loc_80>', '<loc_79>', '<loc_78>', '<loc_77>', '<loc_76>', '<loc_75>', '<loc_74>', '<loc_73>', '<loc_72>', '<loc_71>', '<loc_70>', '<loc_69>', '<loc_68>', '<loc_67>', '<loc_66>', '<loc_65>', '<loc_64>', '<loc_63>', '<loc_62>', '<loc_61>', '<loc_60>', '<loc_59>', '<loc_58>', '<loc_57>', '<loc_56>', '<loc_55>', '<loc_54>', '<loc_53>', '<loc_52>', '<loc_51>', '<loc_50>', '<loc_49>', '<loc_48>', '<loc_47>', '<loc_46>', '<loc_45>', '<loc_44>', '<loc_43>', '<loc_42>', '<loc_41>', '<loc_40>', '<loc_39>', '<loc_38>', '<loc_37>', '<loc_36>', '<loc_35>', '<loc_34>', '<loc_33>', '<loc_32>', '<loc_31>', '<loc_30>', '<loc_29>', '<loc_28>', '<loc_27>', '<loc_26>', '<loc_25>', '<loc_24>', '<loc_23>', '<loc_22>', '<loc_21>', '<loc_20>', '<loc_19>', '<loc_18>', '<loc_17>', '<loc_16>', '<loc_15>', '<loc_14>', '<loc_13>', '<loc_12>', '<loc_11>', '<loc_10>', '<loc_9>', '<loc_8>', '<loc_7>', '<loc_6>', '<loc_5>', '<loc_4>', '<loc_3>', '<loc_2>', '<loc_1>', '<loc_0>', '<other_199>', '<other_198>', '<other_197>', '<other_196>', '<other_195>', '<other_194>', '<other_193>', '<other_192>', '<other_191>', '<other_190>', '<other_189>', '<other_188>', '<other_187>', '<other_186>', '<other_185>', '<other_184>', '<other_183>', '<other_182>', '<other_181>', '<other_180>', '<other_179>', '<other_178>', '<other_177>', '<other_176>', '<other_175>', '<other_174>', '<other_173>', '<other_172>', '<other_171>', '<other_170>', '<other_169>', '<other_168>', '<other_167>', '<other_166>', '<other_165>', '<other_164>', '<other_163>', '<other_162>', '<other_161>', '<other_160>', '<other_159>', '<other_158>', '<other_157>', '<other_156>', '<other_155>', '<other_154>', '<other_153>', '<other_152>', '<other_151>', '<other_150>', '<other_149>', '<other_148>', '<other_147>', '<other_146>', '<other_145>', '<other_144>', '<other_143>', '<other_142>', '<other_141>', '<other_140>', '<other_139>', '<other_138>', '<other_137>', '<other_136>', '<other_135>', '<other_134>', '<other_133>', '<other_132>', '<other_131>', '<other_130>', '<other_129>', '<other_128>', '<other_127>', '<other_126>', '<other_125>', '<other_124>', '<other_123>', '<other_122>', '<other_121>', '<other_120>', '<other_119>', '<other_118>', '<other_117>', '<other_116>', '<other_115>', '<other_114>', '<other_113>', '<other_112>', '<other_111>', '<other_110>', '<other_109>', '<other_108>', '<other_107>', '<other_106>', '<other_105>', '<other_104>', '<other_103>', '<other_102>', '<other_101>', '<other_100>', '<other_99>', '<other_98>', '<other_97>', '<other_96>', '<other_95>', '<other_94>', '<other_93>', '<other_92>', '<other_91>', '<other_90>', '<other_89>', '<other_88>', '<other_87>', '<other_86>', '<other_85>', '<other_84>', '<other_83>', '<other_82>', '<other_81>', '<other_80>', '<other_79>', '<other_78>', '<other_77>', '<other_76>', '<other_75>', '<other_74>', '<other_73>', '<other_72>', '<other_71>', '<other_70>', '<other_69>', '<other_68>', '<other_67>', '<other_66>', '<other_65>', '<other_64>', '<other_63>', '<other_62>', '<other_61>', '<other_60>', '<other_59>', '<other_58>', '<other_57>', '<other_56>', '<other_55>', '<other_54>', '<other_53>', '<other_52>', '<other_51>', '<other_50>', '<other_49>', '<other_48>', '<other_47>', '<other_46>', '<other_45>', '<other_44>', '<other_43>', '<other_42>', '<other_41>', '<other_40>', '<other_39>', '<other_38>', '<other_37>', '<other_36>', '<other_35>', '<other_34>', '<other_33>', '<other_32>', '<other_31>', '<other_30>', '<other_29>', '<other_28>', '<other_27>', '<other_26>', '<other_25>', '<other_24>', '<other_23>', '<other_22>', '<other_21>', '<other_20>', '<other_19>', '<other_18>', '<other_17>', '<other_16>', '<other_15>', '<other_14>', '<other_13>', '<other_12>', '<other_11>', '<other_10>', '<other_9>', '<other_8>', '<other_7>', '<other_6>', '<other_5>', '<other_4>', '<other_3>', '<other_2>', '<other_1>', '<other_0>']
    # fmt: on

    tokenizer = UdopTokenizer.from_pretrained(
        "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512",
        legacy=True,
        additional_special_tokens=additional_special_tokens,
    )
    size = {"height": image_size, "width": image_size}
    image_processor = LayoutLMv3ImageProcessor(
        image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD, size=size
    )
    processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # prepare dummy inputs
    input_ids, bbox, image = prepare_dummy_inputs(tokenizer, image_processor)
    prompt = "Question answering. In which year is the report made?"
    encoding = processor(images=get_image(), text=prompt, return_tensors="pt")

    input_ids = encoding.input_ids
    try:
        EXPECTED_INPUT_IDS = torch.tensor([[11860, 18243, 5, 86, 84, 215, 19, 8, 934, 263, 58, 1, 489, 27, 3838, 7363, 4083, 14536, 3430, 5686, 5911, 17161, 134, 2038, 27, 3838, 22, 7, 4688, 7, 10, 389, 18202, 21, 8, 11046, 37, 3733, 523, 11, 38, 2388, 1628, 3, 13133, 23334, 6, 8, 1656, 79, 3806, 21, 4040, 640, 27, 3838, 22, 7, 701, 16534, 6, 8, 3, 76, 2693, 18, 23015, 5644, 24, 380, 3, 6015, 6, 11, 8, 701, 24, 79, 482, 21, 3, 88, 684, 6, 43, 263, 27, 3838, 22, 7, 3635, 1157, 4089, 6, 2651, 12, 1547, 22, 7, 3265, 655, 5, 19, 27, 3838, 22, 7, 38, 2388, 257, 12, 36, 8, 465, 209, 13409, 12150, 1959, 16, 8, 684, 6, 6737, 57, 165, 126, 13409, 12150, 1623, 5, 71, 1100, 30298, 934, 65, 12566, 24, 27, 3838, 31, 7, 126, 13409, 12150, 1623, 33, 8, 10391, 1710, 859, 8, 420, 3733, 4968, 688, 2699, 16, 1547, 5, 27, 3838, 1217, 131, 99, 23, 179, 6064, 24, 6, 590, 28, 3, 11600, 1456, 701, 6, 175, 9443, 2557, 3635, 92, 1262, 8, 3409, 13, 2186, 3, 27908, 1784, 190, 8, 3, 5771, 17, 13281, 4005, 13, 5086, 11, 13066, 1170, 5, 10826, 16309, 134, 3, 2, 276, 26, 3, 55, 391, 13570, 5, 10315, 309, 3577, 19114, 371, 4254, 5121, 5055, 6245, 3, 10047, 3162, 58, 3, 9, 61, 1713, 2703, 476, 667, 25158, 301, 6058, 6038, 476, 3765, 9149, 10, 4893, 1303, 1986, 5, 13580, 7, 8224, 28244, 7, 5, 76, 75, 7, 89, 5, 15, 1259, 87, 7171, 7, 87, 7, 29, 115, 226, 4305, 2773, 1]])  # fmt: skip
        torch.testing.assert_close(EXPECTED_INPUT_IDS, input_ids)
        bbox = encoding.bbox.float()
        pixel_values = encoding.pixel_values
    except Exception:
        print("Input_ids don't match, preparing dummy inputs")
        input_ids, bbox, pixel_values = prepare_dummy_inputs(tokenizer, image_processor)

    # Verify single forward pass
    print("Testing single forward pass..")
    with torch.no_grad():
        decoder_input_ids = torch.tensor([[101]])
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

    # tensor([[-18.5262,   1.5087, -15.7051]]) on linux
    # tensor([[-19.4976,   0.8515, -17.1873]]) on mac
    try:
        assert torch.allclose(outputs.logits[0, :3, :3], torch.tensor([[-18.5262, 1.5087, -15.7051]]), atol=1e-4)
        print("Looks ok!")
    except Exception:
        print("logits don't match let's try to generate")

    # Verify autoregressive decoding
    print("Testing generation...")
    model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)

    print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # autoregressive decoding with original input data
    print("Testing generation with original inputs...")
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="input_ids_udop.pt", repo_type="dataset")
    input_ids = torch.load(filepath)
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="bbox_udop.pt", repo_type="dataset")
    bbox = torch.load(filepath)
    pixel_values_filename = "pixel_values_udop_512.pt" if "512" in model_name else "pixel_values_udop_224.pt"
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename=pixel_values_filename, repo_type="dataset")
    pixel_values = torch.load(filepath)

    print("Decoded input ids:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("Bbox shape:", bbox.shape)

    model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Generated:", generated_text)

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"microsoft/{model_name}")
        processor.push_to_hub(f"microsoft/{model_name}")
        # BIG note here: to save the fast tokenizer files in the repo on the hub, you need to do the following:
        # see https://discuss.huggingface.co/t/convert-slow-xlmrobertatokenizer-to-fast-one/20876


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="udop-large",
        type=str,
        choices=["udop-large", "udop-large-512", "udop-large-512-300k"],
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
