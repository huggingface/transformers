import json
import random
import os
import time
import tqdm
import copy
import sys
import multiprocessing as mp


from multiprocessing import Pool
from tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='./data')
rng = random.Random()

def get_wiki_docs(wiki_path):
    all_documents = list()
    with open(wiki_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)['text']
            all_documents.append(doc)
    return all_documents

def get_bookcorpus_docs(bookcorpus_dir):
    all_documents = list()
    for i, file in enumerate(os.listdir(bookcorpus_dir)):
        with open(os.path.join(bookcorpus_dir, file), 'r') as f:
            doc = f.read()
        all_documents.append(doc)
    return all_documents

def tokenize_document(document_index):
    document = all_documents[document_index]
    lines = [line for line in document.split('\n') if line]
    list_of_tokens = [tokenizer.tokenize(line) for line in lines]
    return list_of_tokens

def create_instances_from_document(document_index, max_seq_length=128, short_seq_prob=0.0):
    """Creates `TrainingInstance`s for a single document."""
    global tokenized_documents
    document = tokenized_documents[document_index]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    random_document_index = rng.randint(0, len(tokenized_documents) - 1)
                    for _ in range(10):
                        if random_document_index != document_index:
                            break
                        random_document_index = rng.randint(0, len(tokenized_documents) - 1)

                    random_document = tokenized_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                instance = dict()

                instance['tokens_a'] = tokens_a
                instance['tokens_b'] = tokens_b
                instance['is_random_next'] = is_random_next
                instance['target_seq_length'] = target_seq_length
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i+=1
    return instances


if __name__ == '__main__':
    mp.set_start_method('fork')
    n_cores = 30

    out_path = '/home/hdvries/data/prep_128.txt'
    all_documents = get_wiki_docs('/home/nathan/data/wiki/enwiki.txt')
    pool = Pool(n_cores)
    tokenized_documents = list()

    for doc in tqdm.tqdm(pool.imap_unordered(tokenize_document, range(len(all_documents))),
                         total=len(all_documents)):
        tokenized_documents.append(doc)
    pool.close()

    del all_documents

    pool = Pool(n_cores, maxtasksperchild=100000)
    with open(out_path, 'a', encoding='utf-8') as f_out:
        for results in tqdm.tqdm(pool.imap_unordered(create_instances_from_document, range(len(tokenized_documents))),
                                 total=len(tokenized_documents)):
            for r in results:
                r['dataset'] = 'wiki'
                f_out.write(json.dumps(r))
                f_out.write('\n')
    pool.close()

    del tokenized_documents

    all_documents = get_bookcorpus_docs('/home/hdvries/DeepLearningExamples/TensorFlow/LanguageModeling/BERT/data/bookcorpus/download/')
    pool = Pool(n_cores)
    tokenized_documents = list()

    for doc in tqdm.tqdm(pool.imap_unordered(tokenize_document, range(len(all_documents))),
                         total=len(all_documents)):
        tokenized_documents.append(doc)
    pool.close()

    del all_documents

    pool = Pool(n_cores, maxtasksperchild=10000)
    with open(out_path, 'a', encoding='utf-8') as f_out:
        for results in tqdm.tqdm(pool.imap_unordered(create_instances_from_document, range(len(tokenized_documents))),
                                 total=len(tokenized_documents)):
            for r in results:
                r['dataset'] = 'bookcorpus'
                f_out.write(json.dumps(r))
                f_out.write('\n')
    pool.close()
