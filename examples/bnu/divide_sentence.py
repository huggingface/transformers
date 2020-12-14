# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 13:27
# @Author  : wuyan
# @Email   :aerolandX@163.com
# @FileName: divide_sentence.py
# @Function:

from concurrent.futures import ThreadPoolExecutor
from nltk import sent_tokenize
from collections import OrderedDict
from pathlib import PureWindowsPath
from pathlib import PurePosixPath
from typing import Any
from pathlib import Path
import numpy as np
import logging
import os
import csv

logging.basicConfig(level=logging.INFO, format='[%(threadName)-9s] %(message)s')


# root_path = ...
# save_root_path = ...
# gol_info_dict = OrderedDict()
# paths = [str(x) for x in Path(root_path).glob("**/*.txt")]


def process_and_save_func(path: str) -> Any:
    sentence_list = []
    with open(path, 'r+', encoding="utf-8") as f:
        document = f.read()
        sentence_list = sent_tokenize(document)

    save_path = path.replace(root_path, save_root_path)
    pure_path = ''
    if os.name == 'posix':
        pure_path = PurePosixPath(save_path).parents[0]
    else:
        pure_path = PureWindowsPath(save_path).parents[0]
    if not os.path.exists(pure_path):
        os.makedirs(pure_path)
        print(pure_path)
    with open(save_path, 'w+', encoding="utf-8") as f:
        for item in sentence_list:
            f.write(item + '\n')

    numpy_arr = np.array([len(length) for length in sentence_list])
    static_info = [numpy_arr.min(), numpy_arr.max(), numpy_arr.mean(), numpy_arr[numpy_arr <= 128],
                   numpy_arr[numpy_arr > 128]]
    gol_info_dict[path] = static_info
    return save_path


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
    save_root_path = r"/home/wuyan/usr/material/bert_corpus_update"
    gol_info_dict = OrderedDict()
    path = [str(x) for x in Path(root_path).glob("**/*.txt")]
    # path = [r"G:\BTU\bert_corpus\bert_corpus\alloys\1\1.txt", r"G:\BTU\bert_corpus\bert_corpus\alloys\1\2.txt"]
    multip_process(path)
    # process_and_save_func(path)
    save_info_to_csv(gol_info_dict)
