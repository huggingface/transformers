import argparse
import pickle
import random
import time
import numpy as np
from pytorch_transformers import BertTokenizer

from ..utils import logger

def main():
    parser = argparse.ArgumentParser(description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids).")
    parser.add_argument('--file_path', type=str, default='data/dump.txt',
                        help='The path to the data.')
    parser.add_argument('--bert_tokenizer', type=str, default='bert-base-uncased',
                        help="The tokenizer to use.")
    parser.add_argument('--dump_file', type=str, default='data/dump',
                        help='The dump file prefix.')
    args = parser.parse_args()


    logger.info(f'Loading Tokenizer ({args.bert_tokenizer})')
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)


    logger.info(f'Loading text from {args.file_path}')
    with open(args.file_path, 'r', encoding='utf8') as fp:
        data = fp.readlines()


    logger.info(f'Start encoding')
    logger.info(f'{len(data)} examples to process.')

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()
    for text in data:
        text = f'[CLS] {text.strip()} [SEP]'
        token_ids = bert_tokenizer.encode(text)
        rslt.append(token_ids)

        iter += 1
        if iter % interval == 0:
            end = time.time()
            logger.info(f'{iter} examples processed. - {(end-start)/interval:.2f}s/expl')
            start = time.time()
    logger.info('Finished binarization')
    logger.info(f'{len(data)} examples processed.')


    dp_file = f'{args.dump_file}.{args.bert_tokenizer}.pickle'
    rslt_ = [np.uint16(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f'Dump to {dp_file}')
    with open(dp_file, 'wb') as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()