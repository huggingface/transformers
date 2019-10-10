import argparse
import json
import re
import random
from tqdm import tqdm
import torch
from torch import sigmoid
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import numpy as np

def convert_squad_to_sentence_cls(input_file, version=1):
    out_file =  input_file + '.sent'
    sentence_examples = []
    data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)['data']
    ans_sent = 0
    not_sent = 0
    for article in tqdm(data):
        for paragraph in article['paragraphs']:
            context =paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers, is_impossible = qa['id'], qa['question'], qa['answers'], qa.get('is_impossible', None)

                sent_ends = [m.start() for m in re.finditer('\.', context)]
                sent_ends = list(map(lambda x:x+1, sent_ends))
                sent_starts = [0] + sent_ends
                sent_spans = list(zip(sent_starts, sent_ends))
                context_sent_examples = [{'question': question, 'sentence': context[sentence[0]:sentence[1]],
                                         'id': id_, 'answers': answers, 'is_impossible': is_impossible, 'sent_span': sentence}
                                         for sentence in sent_spans]
                for i in range(len(context_sent_examples)):
                    if context_sent_examples[i]['sent_span'][1] - context_sent_examples[i]['sent_span'][0]< 1:
                        print('sentent is too short!')
                        print('squad id:', id_, 'sent_span:', context_sent_examples[i]['sentence'])

                    if context_sent_examples[i]['is_impossible']:
                        context_sent_examples[i]['label'] = '0'
                        not_sent += 1
                    else:
                        is_ans = False
                        for answer in answers:
                            ans_start = answer['answer_start']
                            ans_end = ans_start + len(answer['text'])
                            if context_sent_examples[i]['sent_span'][0] <= ans_start and context_sent_examples[i]['sent_span'][1] >= ans_end:
                                context_sent_examples[i]['label'] = '1'
                                is_ans = True
                                ans_sent += 1
                                break
                        if not is_ans:
                            not_sent += 1
                            context_sent_examples[i]['label'] = '0'
                sentence_examples.extend(context_sent_examples)
    print('can answer:', ans_sent)
    print('cannot answer:', not_sent)
    with open(out_file, 'w', encoding='utf-8') as fout:
        for example in sentence_examples:
            fout.write(json.dumps(example) + '\n')

def write_questions(eval_file, questions):
    questions_list = []
    remain_sents = 0
    for id_, question in tqdm(questions.items()):
        span_min = 99999999999
        span_max = -99999999999
        if question['answers'][0]['nq_input_text']=='long':
            continue
        for sent_span in question['sent_spans']:
            if sent_span[0] < span_min:
                span_min = sent_span[0]
            if sent_span[1] > span_max:
                span_max = sent_span[1]

        context_space = [' ' for _ in range(span_max)]
        sent_spans = []
        for sent_span, sentence, label in zip(question['sent_spans'], question['context'], question['labels']):
            if label == '1':
                is_all_zero = True
                sent_spans.append(sent_span)
                remain_sents += 1
                context_space[sent_span[0]:sent_span[1]] = list(sentence)
        context = ''.join(context_space)
        # verify answer:
        for i, answer in enumerate(question['answers']):
            in_sent = False
            for sent_span in sent_spans:
                if sent_span[0] <= answer['answer_start'] and sent_span[1] >=  answer['answer_start'] + len(answer['text']):
                    in_sent = True
            if in_sent:
                if  answer['text'] != context[answer['answer_start']: answer['answer_start'] + len(answer['text'])]:
                    print(context)
                    print(answer['text'], answer['answer_start'])
                assert answer['text'] == context[answer['answer_start']: answer['answer_start'] + len(answer['text'])]
                question['answers'][i]['in_sent'] = True
            else:
                question['answers'][i]['in_sent'] = False
        # pop invalid answer not in sentence
        answers_new = []
        for answer in question['answers']:
            if answer['in_sent']:
                answer['text'] = ' '.join([tmp for tmp in answer['text'].split(' ') if tmp])
                answers_new.append(answer)


        # context = ' '.join([tmp for tmp in context.split(' ') if tmp])
        if not answers_new:
            question['is_impossible_pred'] = True
        else:
            question['is_impossible_pred'] = False
            new_context = ' '.join([tmp for tmp in context.split(' ') if tmp])
            for i, answer in enumerate(answers_new):
                if answer['text'] in new_context:
                    answer['answer_start'] = new_context.index(answer['text'])
                else:
                    print(new_context)
                    print(answer['text'])
                    print('error!!!!!')
                    assert 0 == 1


            # ans_reduce = 0
            # start = 0
            # while(start + 1< len(context)):
            #     k = start
            #     while(k + 1 < len(context) and context[k] == ' ' and context[k + 1] == ' '):
            #         for i, answer in enumerate(answers_new):
            #             if answer['answer_start'] > k:
            #                 ans_reduce += 1
            #             elif answer['answer_start'] <= k and (answer['answer_start'] + len(answer['text'])) > k:
            #                 answers_new[i]['text'] = ' '.join([tmp for tmp in answers_new[i]['text'].split(' ') if tmp])
            #         k += 1
            #
            #     new_context += context[k]
            #     start = k + 1

        # verify answer:
        for answer in answers_new:
            if answer['text'] != new_context[answer['answer_start']: answer['answer_start'] + len(answer['text'])]:
                print(new_context, answer['text'])
            assert answer['text'] == new_context[answer['answer_start']: answer['answer_start'] + len(answer['text'])]
        # question['answers'] =  question['answers']
        question['answers_pred'] = answers_new
        question['context'] = new_context
        question['id'] = id_
        questions_list.append(question)

    with open(eval_file + '.selected', 'w', encoding='utf-8') as fout:
        for question in questions_list:
            fout.write(json.dumps(question) + '\n')

    squad_data = {'version': 'v2-sent', 'data':[]}
    paragraphs = {}
    for question in questions_list:
        context = question['context']
        id_ = question['id']
        if context not in paragraphs:
            paragraphs[context] = [question]
        else:
            paragraphs[context].append(question)
    para_id = 0
    for context, questions in paragraphs.items():
        if not question['is_impossible'] and len(question['answers'])==0:
            continue
        title = str(para_id)
        para_id += 1
        article = {}
        article['title'] = title

        paragraph = {'context': context}
        paragraph['qas'] = []
        for question in questions:
            qa = {}
            qa['id'] = question['id']
            qa['question'] = question['question']
            if 'dev' in eval_file:
                # print('it is dev.')
                qa['is_impossible'] = question['is_impossible']
                qa['answers'] = question['answers']
                qa['is_impossible_pred'] = question['is_impossible_pred']
                qa['answers_pred'] = question['answers_pred']
            else:
                qa['is_impossible_origin'] = question['is_impossible']
                qa['is_impossible'] = question['is_impossible_pred']
                qa['answers_origin'] = question['answers']

                qa['answers'] = [question['answers_pred'][0]] if question['answers_pred'] else []
                qa['answers_pred'] = question['answers_pred']

                if not qa['answers']:
                    assert qa['is_impossible'] == True
                else:
                    assert len(qa['answers']) == 1
                qa['is_impossible_pred'] = question['is_impossible_pred']
            paragraph['qas'].append(qa)
        article['paragraphs'] = [paragraph]
        squad_data['data'].append(article)
    with open(eval_file + 'selected_squad', 'w', encoding='utf-8') as fout:
        print('save to ', eval_file+'selected_squad')
        json.dump(squad_data, fout)



    return remain_sents

def select_sentence(eval_result, eval_file, ans_th):
    results = None
    with open(eval_result, 'r', encoding='utf-8') as fin:
        results = json.load(fin)
    sentences = []
    with open(eval_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            sentences.append(json.loads(line.strip('\n')))

    preds = results['preds']
    labels = results['labels']
    logits = results['logits']
    total_sents = len(preds)
    assert len(sentences) == len(preds)
    assert len(sentences) == len(logits)
    assert len(sentences) == len(labels)
    preds_th = []
    preds_positive_logits = []
    sentences_classified = []
    max_logit = 0.0
    max_logit_index = 0
    no_answer = True
    for i, (logit, label) in enumerate(zip(logits, labels)):
        sentence = sentences[i]
        sent_label = sentence['label']
        assert sent_label == str(label)
        logit = torch.tensor(logit)
        logit_0, logit_1 = sigmoid(logit).cpu().tolist()
        if logit_1 > max_logit:
            max_logit = logit_1
            max_logit_index = i
        preds_positive_logits.append(logit_1)
        if logit_1 > ans_th:
            pred_th = 1
            no_answer = False
        else:
            pred_th = 0
        preds_th.append(pred_th)
        sentence['label'] = str(pred_th)
        sentences_classified.append(sentence)
    preds = np.array(preds)
    labels = np.array(labels)
    preds_th = np.array(preds_th)
    preds_positive_logits = np.array(preds_positive_logits)
    map = average_precision_score(y_true=labels, y_score=preds)
    map1 = average_precision_score(y_true=labels, y_score=preds_positive_logits)
    print('map1:', map1)
    precision, recall, f1, true_sum = precision_recall_fscore_support(y_true=labels, y_pred=preds_th)
    print('pred th:', 'precision:', precision, 'recall:', recall, 'f1:', f1, 'true_sum', true_sum)

    with open(eval_file + '.cls', 'w', encoding='utf-8') as fout:
        for sentence in sentences_classified:
            fout.write(json.dumps(sentence) + '\n')
    ###merge sentences:
    data = []
    questions = {}
    for sentence in tqdm(sentences_classified):
        id_ = sentence['id']
        if id_ not in questions:
            question = {}
            question['context'] = [sentence['sentence']]
            question['is_impossible'] = sentence['is_impossible']
            question['answers'] = sentence['answers']
            question['sent_spans'] = [sentence['sent_span']]
            question['question'] = sentence['question']
            question['labels'] = [sentence['label']]
            questions[id_] = question
        else:
            questions[id_]['context'].append(sentence['sentence'])
            assert questions[id_]['is_impossible'] == sentence['is_impossible']
            questions[id_]['sent_spans'].append(sentence['sent_span'])
            questions[id_]['labels'].append(sentence['label'])
        ## verify answers
    remain_sents = write_questions(eval_file, questions)
    print('total sents:', total_sents, 'remain sents:', remain_sents, 'remove percentage:', (total_sents - remain_sents) * 1.0 / total_sents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str)
    parser.add_argument('--convert_sent', action='store_true')

    parser.add_argument('--select_sent', action='store_true')
    parser.add_argument('--eval_result', default='', type=str)
    parser.add_argument('--eval_file', default='', type=str)
    parser.add_argument('--ans_th', default=0.6, type=float)

    args = parser.parse_args()
    if args.convert_sent:
        convert_squad_to_sentence_cls(args.input_file)
    if args.select_sent:
        select_sentence(args.eval_result, args.eval_file, args.ans_th)


