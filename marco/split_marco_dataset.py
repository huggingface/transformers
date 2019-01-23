
import torch
from torch.autograd import Variable
import numpy as np

from tqdm import tqdm

import json


#file="/home/david/work/MSMARCOV2/dataset/dev_v2.1.json"
file="/home/david/work/MSMARCOV2/dataset/eval_v2.1_public.json"
output ={}

split_count = 1000

import io, json
def save_to_file(file_name, data):
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


with open(file) as f_o:
    source = json.load(f_o)

    query_id = source['query_id']
    query = source['query']
    passages = source['passages']
    answers = source['answers']
    wellFormedAnswers = source['wellFormedAnswers']
    query_type = source['query_type']

    
    
    output['query_id'] = {}
    output['query'] = {}
    output['passages'] = {}
    output['answers'] = {}
    output['wellFormedAnswers'] = {}
    output['query_type'] = {}
    save_count = 0
    for index, qid in tqdm(enumerate(query_id)):
        output['query_id'][qid]=query_id[qid]
        output['query'][qid] = query[qid]
        output['answers'][qid] = answers[qid]
        output['wellFormedAnswers'][qid] = wellFormedAnswers[qid]
        output['query_type'][qid] = query_type[qid]
        output['passages'][qid] = passages[qid]
        #print(output)
        if index and index % split_count == 0:
            file_name = 'output_file_{0:05}.json'.format(save_count)
            save_count += 1
            save_to_file(file_name, output)

            output['query_id'] = {}
            output['query'] = {}
            output['passages'] = {}
            output['answers'] = {}
            output['wellFormedAnswers'] = {}
            output['query_type'] = {}


