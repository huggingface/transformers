import os
import shutil
import argparse
import torch
from x2paddle.convert import pytorch2paddle

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        help="The name of the glue task to train on.",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save x2paddle model.",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    args = parser.parse_args()
    return args

args = parse_args()

if args.task_name is not None:
   raw_datasets = load_dataset("glue", args.task_name)
   is_regression = args.task_name == "stsb"
   if not is_regression:
       label_list = raw_datasets["train"].features["label"].names
       num_labels = len(label_list)
   else:
       num_labels = 1

config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config
)
model.eval()

input_ids = torch.zeros([1, args.max_length]).long()
token_type_ids = torch.zeros([1, args.max_length]).long()
attention_msk = torch.zeros([1, args.max_length]).long()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir) 

# start convert
pytorch2paddle(model,
               save_dir=args.save_dir,
               jit_type="trace",  
               input_examples = [input_ids, attention_msk, token_type_ids])

for file_name in os.listdir(args.model_name_or_path):
    if 'json' in file_name or 'txt' in file_name:
        shutil.copy(
            os.path.join(args.model_name_or_path, file_name),
            args.save_dir)

