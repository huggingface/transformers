from transformers.models.visual_bert.convert_visual_bert_original_pytorch_checkpoint_to_pytorch import convert_visual_bert_checkpoint
import os

state_dir = "/media/crocoder/New Volume/Weights"

ACCEPTABLE_CHECKPOINTS = [
    'nlvr2_coco_pre_trained.th',
    'nlvr2_fine_tuned.th',
    'nlvr2_pre_trained.th',
    'vcr_coco_pre_train.th',
    'vcr_fine_tune.th',
    'vcr_pre_train.th',
    'vqa_coco_pre_trained.th',
    'vqa_fine_tuned.th',
    'vqa_pre_trained.th',
]

DUMP_NAMES = [
    'visualbert-nlvr2-coco-pre',
    'visualbert-nlvr2',
    'visualbert-nlvr2-pre',
    'visualbert-vcr-coco-pre',
    'visualbert-vcr',
    'visualbert-vcr-pre',
    'visualbert-vqa-coco-pre',
    'visualbert-vqa',
    'visualbert-vqa-pre',
]

file_to_dump = dict(zip(ACCEPTABLE_CHECKPOINTS, DUMP_NAMES))

for file_name in ACCEPTABLE_CHECKPOINTS:
    print(file_name)
    file_path = os.path.join(state_dir, file_name)
    convert_visual_bert_checkpoint(file_path, os.path.join(state_dir, 'saved', 'hf-repo', file_to_dump[file_name]))
