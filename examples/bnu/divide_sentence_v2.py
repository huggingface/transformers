from concurrent.futures import ThreadPoolExecutor
from nltk import sent_tokenize
from collections import OrderedDict
from typing import Any
from pathlib import Path
import numpy as np
import logging
import csv

logging.basicConfig(level=logging.INFO, format='[%(threadName)-9s] %(message)s')


def process_and_save_func(file_path: str) -> Any:
    sentence_list=[]
    try:
        with open(file_path, 'r+', encoding="utf-8") as f:
            document = f.read()
            sentence_list = sent_tokenize(document)
            f.seek(0)
            for item in sentence_list:
                f.write(item + '\n')
    except UnicodeError:
        logging.info('{} have unicodeError'.format(file_path))
        with open(file_path, 'r+', encoding="gkb") as f:
            document = f.read()
            sentence_list = sent_tokenize(document)
            f.seek(0)
            for item in sentence_list:
                f.write(item + '\n')
    finally:
        numpy_arr = np.array([len(length) for length in sentence_list])
        if numpy_arr.size != 0:
            static_info = [numpy_arr.min(), numpy_arr.max(), numpy_arr.mean(), numpy_arr[numpy_arr <= 128],
                           numpy_arr[numpy_arr > 128]]
            gol_info_dict[file_path] = static_info
        else:
            logging.info("{} length is zero".format(file_path))
        return file_path




def multip_process(task: list):
    executor = ThreadPoolExecutor()
    for result in executor.map(process_and_save_func, task):
        logging.info('{} already processed'.format(result))


def save_info_to_csv(info_dict: dict):
    with open('result.csv', 'w+') as f:
        w = csv.writer(f)
        w.writerows(info_dict.items())


if __name__ == '__main__':
    # root_path = r"G:\BTU\bert_corpus\bert_corpus"
    root_path = r"/home/wuyan/usr/material/bert_corpus"
    # save_root_path = r"G:\BTU\bert_corpus_update"
    # save_root_path = r"/home/wuyan/usr/material/bert_corpus_update"
    gol_info_dict = OrderedDict()
    path = [str(x) for x in Path(root_path).glob("**/*.txt")]
    # path = [r"G:\BTU\bert_corpus\bert_corpus\alloys\1\1.txt", r"G:\BTU\bert_corpus\bert_corpus\alloys\1\2.txt"]
    # path = ["./1.txt", "./10.txt"]
    multip_process(path)
    # process_and_save_func(path)
    save_info_to_csv(gol_info_dict)
