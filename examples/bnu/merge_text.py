# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 19:37
# @Author  : wuyan
# @Email   :aerolandX@163.com
# @FileName: merge_text.py
# @Function:

from concurrent.futures import ThreadPoolExecutor
from chardet.universaldetector import UniversalDetector
from pathlib import Path
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format='[%(threadName)-9s] %(message)s')


def get_encode_info(file):
    with open(file, 'rb') as f:
        detector = UniversalDetector()
        for line in f.readlines():
            detector.feed(line)
            if detector.done:
                break
        detector.close()
        return detector.result['encoding']


def read_file(file):
    with open(file, 'rb') as f:
        return f.read()


def write_file(content, file):
    with open(file, 'wb') as f:
        f.write(content)


def convert_encode2utf8(file, original_encode, des_encode):
    file_content = read_file(file)
    file_decode = file_content.decode(original_encode, 'ignore')
    file_encode = file_decode.encode(des_encode)
    write_file(file_encode, file)


def covert_all(file_path):
    file_content = read_file(file_path)
    encode_info = get_encode_info(file_path)
    if encode_info != 'utf-8':
        convert_encode2utf8(file_path, encode_info, 'utf-8')
        return file_path


def merge_text():
    ...


def multip_process(task: list):
    executor = ThreadPoolExecutor()
    for result in executor.map(covert_all, task):
        logging.info('{} already processed'.format(result))


if __name__ == '__main__':
    root_path = r"/home/wuyan/usr/material/bert_corpus"
    path = [str(x) for x in Path(root_path).glob("**/*.txt")]
    multip_process(path)
