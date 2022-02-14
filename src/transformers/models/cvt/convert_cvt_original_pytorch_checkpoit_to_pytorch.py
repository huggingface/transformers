# Run this file inside CVT Repo

from transformers import CvtConfig, CvtForImageClassification
from lib.models.cls_cvt import ConvolutionalVisionTransformer as CVT
import torch
import yaml

# Repo link: https://github.com/microsoft/CvT
# Model Zoo: https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al
# yaml is in experiments folder

with open('C:\\Users\AH87766\Documents\CvT\experiments\imagenet\cvt\cvt-13-224x224.yaml', 'r') as f:
    original_config = yaml.load(f)

cvt_hugging_config = CvtConfig()
cvt_hugging_model = CvtForImageClassification(cvt_hugging_config)

original_model = CVT(spec=original_config['MODEL']['SPEC'])

print(original_model)
print(cvt_hugging_model)

