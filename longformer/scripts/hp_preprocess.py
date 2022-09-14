"""
Instrocutions for prepraing the hyperpartisan dataset:

1- Download the original data from PAN at SemEval 2019 Task 4 https://zenodo.org/record/1489920
   -  the training subset: `articles-training-byarticle-20181122.zip`
   -   labels: `ground-truth-training-byarticle-20181122.zip`
2- Decompress the files (the output should be a single .xml file)
3- run this script with appropriate file paths
"""

import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import os
import simplejson as json
import codecs
import re
import io
import jsonlines
from collections import defaultdict
import pathlib

fp = io.BytesIO()  # writable file-like object
writer = jsonlines.Writer(fp)

FLAGS = re.MULTILINE | re.DOTALL
def re_sub(pattern, repl, text, flags=None):
    if flags is None:
        return re.sub(pattern, repl, text, flags=FLAGS)
    else:
        return re.sub(pattern, repl, text, flags=(FLAGS | flags))


def clean_txt(text):

    text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"&#160;", "", text)

    # Remove URL
    text = re_sub(r"(http)\S+", "", text)
    text = re_sub(r"(www)\S+", "", text)
    text = re_sub(r"(href)\S+", "", text)
    # Remove multiple spaces
    text = re_sub(r"[ \s\t\n]+", " ", text)

    # remove repetition
    text = re_sub(r"([!?.]){2,}", r"\1", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

    return text.strip()


def write_jsonlist(list_of_json_objects, output_filename):
    with jsonlines.open(output_filename, mode='w') as writer:
        writer.write_all(list_of_json_objects)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='articles-training-byarticle-20181122.xml')
    parser.add_argument('--labels-file', default='ground-truth-training-byarticle-20181122.xml')
    parser.add_argument('--splits-file', default='hp-splits.json')
    parser.add_argument('--output-dir', help='path to write outfile files')
    args = parser.parse_args()

    print('loading articles...')
    articles_root = ET.parse(args.train_file).getroot()
    print('loading labels...')
    labels_root = ET.parse(args.labels_file).getroot()
    articles = articles_root.findall('article')
    labels = labels_root.findall('article')
    assert len(articles) == len(labels)

    data = {}
    for article, label in tqdm(zip(articles, labels), total=len(labels), desc="preprocessing"):
        text = ET.tostring(article, method='text', encoding="utf-8").decode('utf-8')
        text = clean_txt(text)
        id_ = int(label.attrib['id'])
        data[id_] = {'text': text, 'label': label.attrib['hyperpartisan'], 'id': id_}

    splits = defaultdict(list)
    with open(args.splits_file) as f_in:
        for split, ids in json.load(f_in).items():
            for id_ in ids:
                splits[split].append(data[id_])

    for subset, data_list in splits.items():
        output_filename = os.path.join(args.output_dir, subset + '.jsonl')
        pathlib.Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        write_jsonlist(data_list, output_filename)


if __name__ == '__main__':
    main()
