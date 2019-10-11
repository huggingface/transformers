import gzip
import json
def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "r")
if __name__ == '__main__':
    # with gzip.open("/mnt/nq/sampledev/nq-dev-10.jsonl.gz", "w") as f:
    #     with _open("/mnt/nq/sampledev/nq-dev-sample.jsonl.gz") as input_jsonl:
    #         c = 0
    #         for line in input_jsonl:
    #             f.write(line)
    #             c += 1
    #             if c >= 10:
    #                 break

    json_file = "all_squadformat_nq_dev_withnqidx.json"
    split_file = "_split_squadformat_nq_dev_withnqidx.json"
    with open(json_file,"r") as fin:
        results = json.load(fin)
    # exit()
    print(results.keys())
    print("total examples:",len(results["data"]))

    num = int(len(results["data"])/4)
    idx = []
    for i in range(4):
        idx.append(i*num)
    idx.append(len(results["data"]))
    sum  = 0
    for i in range(len(idx)-1):
        data = results["data"][idx[i]:idx[i+1]]
        # print(len(data))
        split_json_path = str(i)+split_file
        with open(split_json_path, 'wb') as w:
            datas = json.dumps({'data': data, 'version': 'nq'})
            # print(type(datas))
            w.write(datas.encode())
        print("Split {}: {} examples, and saved to {}".format(i,len(data),split_json_path))