import pickle
import argparse
import gzip
import json
import collections
import os
import re
def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "r")
    else:
        print("wrong file")
        exit()
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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # # Required parameters
    # parser.add_argument("--input_gzip_dir", default=None, type=str, required=True)
    # parser.add_argument("--output_pklabel_file", default=None, type=str, required=True)
    # args = parser.parse_args()
    # # -------------------------------------generated labels-------------------------------------------
    # input_files = []
    # for i in range(0,5):
    #     input_files.append(os.path.join(args.input_gzip_dir,"nq-train-0{}.jsonl.gz".format(i)))
    # example_short_anno = {}
    # from tqdm import tqdm
    # for input_file in tqdm(input_files):
    #     with _open(input_file) as input_jsonl:
    #
    #         for line in input_jsonl:
    #             e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    #             eid = e["example_id"]
    #             anno = e["annotations"][0]
    #             question = " ".join(e["question_tokens"])
    #             short_ans = anno["short_answers"]
    #             if len(short_ans) == 0:
    #                 example_short_anno[eid] = [-1, -1, False,question]
    #             else:
    #                 answer = short_ans[0]
    #                 start_tok = answer["start_token"]
    #                 end_tok = answer["end_token"]
    #                 if start_tok == -1 or end_tok ==-1:
    #                     example_short_anno[eid] = [start_tok,end_tok,False,question]
    #                     print("!")
    #                 else:
    #                     example_short_anno[eid] = [start_tok,end_tok,True,question]
    # print(len(example_short_anno))
    # pickle.dump(example_short_anno,open(args.output_pklabel_file,"wb"))

    #----------------------------------modified pred_with_sent-------------------------------------
    label_file = "/data/nieping/pytorch-transformers/data/nq_sentence_selector/train_5_piece/train5piece_short_label.pk"
    label_file = "train5piece_short_label.pk"
    example_labels = pickle.load(open(label_file,"rb"))#30632

    pred_files = []
    for i in range(4):
        pred_files.append("/data/nieping/pytorch-transformers/data/nq_sentence_selector/train_5_piece/train5piece_nbest_predwithsent_{}.pk".format(i))

    all_sents_examples = []
    count_no_annotation_examples = 0
    count_has_annotation_examples = 0
    count_has_answer = 0
    count_no_answer = 0
    from tqdm import tqdm
    for file in tqdm(pred_files):
        file = "train5piece_nbest_predwithsent_0.pk"
        data = pickle.load(open(file,"rb"))
        for (eid,ans_list) in data.items():
            #----------generated distinguished sentens
            sents_dict = {}
            for ana in ans_list:
                if ana["sent"] in sents_dict:
                    continue
                else:
                    print(ana["sent"])
                    print(ana["text"])


            # if int(eid) in example_labels:
            #     label = example_labels[int(eid)]
            # else:
            #     label = [-1,-1,False]
            #     continue

