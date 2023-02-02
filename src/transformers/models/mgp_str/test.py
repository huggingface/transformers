import torch
import requests
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from transformers import (
    BertTokenizer,
    GPT2Tokenizer,
    MGPSTRTokenizer,
    MGPSTRConfig,
    MGPSTRModel,
    MGPSTRProcessor
)

from transformers import MGPSTRTokenizer

# tokenizer = MGPSTRTokenizer.from_pretrained('alibaba-damo/mgp-str-base')


# url = 'https://i.postimg.cc/ZKwLg2Gw/367-14.png'
# url = 'https://i.postimg.cc/ZKwLg2Gw/367-14.png'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# # image = image.resize((128, 32), Image.BICUBIC)
# # img_tensor = transforms.ToTensor()(image)
# # image_tensor = img_tensor.unsqueeze(0)

# processor = MGPSTRProcessor.from_pretrained('alibaba-damo/mgp-str-base')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values
# print(pixel_values.shape)

# mgp_str_config = MGPSTRConfig()
# model = MGPSTRModel.from_pretrained('alibaba-damo/mgp-str-base')
# outs, attens = model(pixel_values)
# print(outs[0].shape)
# out_strs = processor.batch_decode(outs)
# print(out_strs['generated_text'])




# 从本地load模型
# from collections import OrderedDict
# model_path = '/mnt/workspace/workgroup/wangpeng/wangpeng/research/model/mgpstr/mgp_str_base_patch4_32_128.pth'
# state_dict = torch.load(model_path, map_location='cpu')
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k
#     if k.startswith('module.mgp_str.'):
#         name = k[15:]
#     if name.startswith('blocks.'):
#         name = 'encoder.' + name
#     if name.startswith('char_tokenLearner.'):
#         name = name.replace('char_tokenLearner', 'char_a3_module')
#     if name.startswith('wp_tokenLearner.'):
#         name = name.replace('wp_tokenLearner', 'wp_a3_module')
#     if name.startswith('bpe_tokenLearner.'):
#         name = name.replace('bpe_tokenLearner', 'bpe_a3_module')
#     new_state_dict[name] = v
# del new_state_dict["norm.weight"]
# del new_state_dict["norm.bias"]
# del new_state_dict["head.weight"]
# del new_state_dict["head.bias"]

# mgp_str_config = MGPSTRConfig()
# model = MGPSTRModel(mgp_str_config)
# model.load_state_dict(new_state_dict)
# torch.save(new_state_dict, '/mnt/workspace/workgroup/wangpeng/wangpeng/research/model/mgpstr/huggingface/pytorch_model.bin')
# model.eval()

# vit = '/mnt/workspace/workgroup/wangpeng/wangpeng/research/model/vit/pytorch_model.bin'
# vit_state_dict = torch.load(vit, map_location='cpu')



# test for IIT5K dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset


class ImgDataset(Dataset):

    def __init__(self, root, batch_size):

        with open(root) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.data_list = [x.strip() for x in content] 

        self.leng_index = [0] * 26
        self.leng_index.append(0)
        text_length = 1
        for i, line in enumerate(self.data_list):
            label_text = line.split(' ')[0]
            if i > 0 and len(label_text) != text_length:
                self.leng_index[text_length] = i
            text_length = len(label_text)
        
        self.nSamples = len(self.data_list)
        self.batch_size = batch_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        sample_ = self.data_list[index].split(' ')
        label = sample_[0]
        img_path = sample_[1]

        try:
            img = Image.open(img_path).convert('RGB')  # for color image
        except IOError:
            print('Corrupted read image for ', img_path)
            randi = random.randint(0, self.nSamples-1)
            return self.__getitem__(randi)

        label = label.lower() 

        return img, label, img_path

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=128, opt=None):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels, img_paths = zip(*batch)

        # pil = transforms.ToPILImage()
        # image_tensors = [transforms.ToTensor()((image.resize((128, 32), Image.BICUBIC))) for image in images]
        # image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        # return image_tensors, labels, img_paths
        return images, labels, img_paths

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

mgp_str_config = MGPSTRConfig()
model = MGPSTRModel.from_pretrained('alibaba-damo/mgp-str-base')
# outs, attens = model(pixel_values)
processor = MGPSTRProcessor.from_pretrained('alibaba-damo/mgp-str-base')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# dataloader
eval_data_path = '/mnt/workspace/workgroup/wangpeng/wangpeng/research/STR/datasets/imgs/evaluation/A100/IIIT5k_3000.txt'
eval_data = ImgDataset(root=eval_data_path, batch_size=8)
AlignCollate_evaluation = AlignCollate(imgH=32, imgW=128)
evaluation_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=8,
    shuffle=False,
    num_workers=int(12),
    collate_fn=AlignCollate_evaluation, pin_memory=True)

char_n_correct = 0
bpe_n_correct = 0
wp_n_correct = 0
out_n_correct = 0
length_of_data = 0
for i, (image_tensors, labels, imgs_path) in enumerate(evaluation_loader):
    batch_size = len(labels)
    length_of_data = length_of_data + batch_size
    
    pixel_values = processor(images=image_tensors, return_tensors="pt").pixel_values
    outs, attens = model(pixel_values)
    
    # outs, attens = model(image_tensors)
    out_strs = processor.batch_decode(outs)
    fianl_texts = out_strs['generated_text']
    final_scores = out_strs['scores']
    char_strs = out_strs['char_preds']
    bpe_strs = out_strs['bpe_preds']
    wp_strs = out_strs['wp_preds']
    for index,gt in enumerate(labels):
        if char_strs[index] == gt:
            char_n_correct += 1
        if bpe_strs[index] == gt:
            bpe_n_correct += 1
        if wp_strs[index] == gt:
            wp_n_correct += 1
        if fianl_texts[index] == gt:
            out_n_correct += 1

char_accuracy = char_n_correct/float(length_of_data) * 100
bpe_accuracy = bpe_n_correct / float(length_of_data) * 100
wp_accuracy = wp_n_correct / float(length_of_data) * 100
out_accuracy = out_n_correct / float(length_of_data) * 100
print(f'char_Acc {char_accuracy:0.3f}\t bpe_Acc {bpe_accuracy:0.3f}\t wp_Acc {wp_accuracy:0.3f}\t  fused_Acc {out_accuracy:0.3f}')


