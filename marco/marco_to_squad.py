import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import json
from collections import OrderedDict
import collections
import io, json

input_file='/home/david/work/marco_dataset/dev_v2.1.json'
output_file='/home/david/work/marco_dataset/marco2squad_dev_v2.1.json'
limited = 10000000
support_select_answer = False
support_no_answer = False
is_evaluation= False
def add_insert_squad(data, title, context, question, id, answer_pos, answer_text):

    cur_entry = None
    for entry_index, entry in enumerate(data['data']):
        if entry['title'] == title:
            cur_entry = entry
            break
    if cur_entry is None:
        cur_entry={}
        cur_entry['title'] = title
        cur_entry['paragraphs'] = []
        data['data'].append(cur_entry)

    cur_paragraph = None
    for paragraph_index, paragraph in enumerate(cur_entry['paragraphs']):
        pos = paragraph['context'].find(context)
        if pos >= 0:
            cur_paragraph = paragraph
            break;
    if cur_paragraph is None:
        cur_paragraph = {}
        cur_paragraph['context'] = context
        cur_paragraph['qas'] = []
        cur_entry['paragraphs'].append(cur_paragraph)

    qas = cur_paragraph['qas']

    qa = {}
    qa['answers']=[]
    answer = {}
    answer['answer_start'] = answer_pos
    answer['text'] = answer_text
    qa['answers'].append(answer)
    qa['question'] = question
    qa['id'] = str(id)

    qas.append(qa)

def save_to_file(file_name, data):
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

output = {'data':[], 'version':'1.1'}

with open(input_file) as f_o:
    source = json.load(f_o)
    f_o.close()
    query_id = source['query_id']
    query = source['query']
    passages = source['passages']
    answers = source['answers']
    wellFormedAnswers = source['wellFormedAnswers']
    query_type = source['query_type']
        

    save_count = 0
    for index, qid in tqdm(enumerate(query_id)):
        is_found = False
        if is_evaluation is True:
            sum_passage = ' '.join(passage['passage_text'] for passage in passages[qid])
            add_insert_squad(output, query_type[qid], sum_passage, query[qid], query_id[qid], 0, '')
        else:
            for passage in passages[qid]:
                pos = passage['passage_text'].find(answers[qid][0])
                if pos >= 0:
                    is_found = True
                    save_count += 1
                    add_insert_squad(output, query_type[qid], passage['passage_text'], query[qid], query_id[qid], pos, answers[qid][0])
                    break;
            if support_select_answer is True and is_found is False:
                for passage in passages[qid]:
                    if passage['is_selected'] is 1:
                        save_count += 1
                        add_insert_squad(output, query_type[qid], passage['passage_text'], query[qid], query_id[qid], 0, passages[qid][0]['passage_text'])
                        is_found = True
                        break;
            if support_no_answer is True and is_found is False:
                save_count += 1
                add_insert_squad(output, query_type[qid], passages[qid][0]['passage_text'], query[qid], query_id[qid], 0, '')
        if index > limited:
            print('stop limited:', limited)
            break
print('load dataset', len(query_id))
print('save dataset', save_count)

save_to_file(output_file, output)