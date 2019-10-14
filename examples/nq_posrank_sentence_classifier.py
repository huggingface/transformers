import pickle
import argparse
import gzip
import json
import collections
import os
def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "r")
    else:
        print("wrong file")
        exit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_gzip_dir", default=None, type=str, required=True)
    parser.add_argument("--output_pklabel_file", default=None, type=str, required=True)
    args = parser.parse_args()

    input_files = []
    for i in range(0,5):
        input_files.append(os.path.join(args.input_gzip_dir,"nq-train-0{}.jsonl.gz".format(i)))
    example_short_anno = {}
    import tqdm
    for input_file in tqdm(input_files):
        with _open(input_file) as input_jsonl:

            for line in input_jsonl:
                e = json.loads(line, object_pairs_hook=collections.OrderedDict)
                eid = e["example_id"]
                anno = e["annotations"][0]
                short_ans = anno["short_answers"]
                if len(short_ans) == 0:
                    example_short_anno[eid] = [-1, -1, False]
                else:
                    answer = short_ans[0]
                    start_tok = answer["start_token"]
                    end_tok = answer["end_token"]
                    if start_tok == -1 or end_tok ==-1:
                        example_short_anno[eid] = [start_tok,end_tok,False]
                        print("!")
                    else:
                        example_short_anno[eid] = [start_tok,end_tok,True]
    print(len(example_short_anno))
    pickle.dump(example_short_anno,open(args.output_pklabel_file,"wb"))