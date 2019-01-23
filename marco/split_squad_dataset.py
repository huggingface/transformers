
import torch
from torch.autograd import Variable
import numpy as np

from tqdm import tqdm

import json


input_file="/home/david/work/datasets/train-v1.1.json"
output ={}

split_count = 1000

import io, json
def save_to_file(file_name, data):
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

with open(input_file, "r", encoding='utf-8') as reader:
    dataset_json = json.load(reader)
    input_data = dataset_json["data"]
    
    examples = []
    total_count = 0
    for entry_index, entry in enumerate(input_data):
        print('title', entry['title'])
        for paragraph_index, paragraph in enumerate(entry["paragraphs"]):
            context = paragraph["context"]
            #print(' ', 'context', context)
            for qa_index, qa in enumerate(paragraph["qas"]):
                qas_id = qa["id"]
                question_text = qa["question"]

                answer0 = qa["answers"][0]
                answer_text = answer0["text"]
                answer_start = answer0["answer_start"]

                # print(' ', ' ', 'id', qas_id)
                # print(' ', ' ', 'question', question_text)
                # print(' ', ' ', 'answer', answer0)
                # print(' ', ' ', 'answer_text', answer_text)
                # print(' ', ' ', 'answer_start', answer_start)
            print(entry_index, paragraph_index, 'total_qa', qa_index)
        print(entry_index, 'total_paragraph', paragraph_index)
    print(' ', 'total_entry', entry_index)



