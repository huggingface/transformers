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
def read_annotation_for_traingzip(gzip_dir,number):
    input_files = []
    for i in range(0,number):
        input_files.append(os.path.join(gzip_dir,"nq-train-0{}.jsonl.gz".format(i)))
    example_short_anno = {}
    from tqdm import tqdm
    for input_file in tqdm(input_files):
        with _open(input_file) as input_jsonl:

            for line in input_jsonl:
                e = json.loads(line, object_pairs_hook=collections.OrderedDict)
                eid = e["example_id"]
                anno = e["annotations"][0]
                question = " ".join(e["question_tokens"])
                short_ans = anno["short_answers"]
                if len(short_ans) == 0:
                    example_short_anno[eid] = [-1, -1, False,question]
                else:
                    answer = short_ans[0]
                    start_tok = answer["start_token"]
                    end_tok = answer["end_token"]
                    if start_tok == -1 or end_tok ==-1:
                        example_short_anno[eid] = [start_tok,end_tok,False,question]
                        print("!")
                    else:
                        example_short_anno[eid] = [start_tok,end_tok,True,question]
    print("total examples",len(example_short_anno))
    return example_short_anno

def convert_predwithsent_cls_format(example_labels,predwithsent_files):
    '''

    :param example_labels: a dict{example_id:[start_nq_idx, end_nq_idx, False/True,question]}
    :param predwithsent_files: a dict {example_id:{generated from nq_posrank_Sentences_generation}}
    :return:
    '''
    all_sents_examples = []
    count_no_annotation_examples = 0
    count_has_annotation_examples = 0
    count_has_answer = 0
    count_no_answer = 0
    data = pickle.load(open(predwithsent_files,"rb"))
    for (eid,ans_list) in data.items():
        question = ans_list[0]["question"]
        #----------generated distinguished sentens
        sents_dict = {}
        if int(eid) in example_labels:
            label = example_labels[int(eid)]
            count_has_annotation_examples+=1
        else:
            print("an example without annotation")
            count_no_annotation_examples +=1
            continue

        for ana in ans_list:
            if ana["sent"] in sents_dict:
                continue
            else:
                sents_dict[ana["sent"]] = [ana["sent_start_nq_idx"],ana["sent_end_nq_idx"]]

        for (sent,span) in sents_dict.items():
            if label[-1] and label[0] >= span[0] and label[1] <= span[1]:
                sent_label = 1
                count_has_answer+=1
            else:
                sent_label = 0
                count_no_answer+=1
            all_sents_examples.append({
                'example_id':int(eid),
                'question': question,
                'sentence': sent,
                'sent_span': span,
                'sent_label':sent_label,
            })
    print("no annotation examples:",count_no_annotation_examples)
    print("has annotation examples:",count_has_annotation_examples)
    print("All sents:",len(all_sents_examples))
    print("\t positive sent:",count_has_answer)
    print("\t negative sent:",count_no_answer)
    return all_sents_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters



    # # -------------------------------------generated labels-------------------------------------------
    # parser.add_argument("--input_gzip_dir", default=None, type=str, required=True)
    # parser.add_argument("--output_label_file", default=None, type=str, required=True)
    # parser.add_argument("--output_pklabel_file", default=None, type=str, required=True)
    # annotated_label = read_annotation_for_traingzip(args.input_gzip_dir, 5, args.output_label_file)
    # pickle.dump(annotated_label, open(args.output_file, "wb"))
    # print("Dumpted annotations into {}".format(args.output_file))
    #----------------------------------modified pred_with_sent-------------------------------------
    # parser.add_argument("--label_file", default=None, type=str, required=True)
    # parser.add_argument("--predwithsent_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    args = parser.parse_args()
    # label_file = args.label_file
    label_file = "/data/nieping/pytorch-transformers/data/nq_sentence_selector/train_5_piece/train5piece_short_label.pk"
    example_labels = pickle.load(open(label_file,"rb"))#read_annotation_for_trainingzip
    all_sents = []
    for i in range(4):
        file = "/data/nieping/pytorch-transformers/data/nq_sentence_selector/train_5_piece/train5piece_nbest_predwithsent_{}.pk".format(i)
        print("pred_with_sent_file:",file)
        all_sents.extend(convert_predwithsent_cls_format(example_labels,file))
    pickle.dump(all_sents,open(args.output_file,"wb"))
    print("Total sents:",len(all_sents))
    count_has = 0
    count_no = 0
    for sent in all_sents:
        if sent["sent_label"]:
            count_has+=1
        else:
            count_no+=1
    print("\t positive sent:",count_has)
    print("\t negative sent:",count_no)
    print("Finised dump:",args.output_file)
