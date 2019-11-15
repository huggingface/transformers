import pickle
import os
import argparse
import json
import spacy
import re

import torch
from tqdm import tqdm
from itertools import permutations 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse as sp

nlp = spacy.blank('en')

MAX_NUM_SENTENCES = 200
VOCAB_SIZE = 50000


# not used anymore?!
def vanilla_tokenize(sentence):
    tok_sentence = re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", sentence)
    return tok_sentence

def spacy_tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

# we dont talk about this anymore.
def normalize_adj(adj, sparsity=0.2):
    # zscore normalize adj
    mean = np.mean(adj)
    std = np.std(adj)
    shift = std * sparsity * 3
    adj = (adj - mean - shift) / std
    return 1 / (1 + np.exp(-adj))

def read_hotpotqa_data(args):

    data_file, output_dir, data_type = args.data_file, args.output_dir, args.data_type

    if args.tokenization_type == "spacy":
        tokenize_sentence = spacy_tokenize
        output_dir = output_dir + "_sp"
    else:
        tokenize_sentence = vanilla_tokenize
        output_dir = output_dir + "_vn"


    with open(data_file, "r", encoding='utf-8') as open_file:
        read_file = json.load(open_file)

    ##########
    # CONSTRUCT TEXT DATA
    ##########

    vocab = dict()
    vectorizer = TfidfVectorizer()

    for entry in tqdm(read_file):
        entry_id = entry["_id"]
        answer = tokenize_sentence(entry["answer"].lower())
        question = tokenize_sentence(entry["question"].lower())

        ### FLATTEN SENTENCE LIST # EDIT FROM FUTURE, WE DONT FLATTEN ANYMORE
        article_sentences = list() # list of sentences (NUM SENTENCES X SENTENCE LENGTH)
        article_sentence_tokens = list() # list of sentences (NUM SENTENCES X SENTENCE LENGTH)
        sentence_to_article = dict() # a way to get from sentence index, back to article index
        title_to_sentence = dict() # a way to get from title to first sentence index
        article_adjms = list() # what do you think this is lol

        total_len = 0
        sentence_idx = 0
        num_sentences = 0
        for article_idx, article in enumerate(entry['context']):
            title = article[0]
            title_to_sentence[title] = sentence_idx

            ind = np.diag_indices(len(article[1]))

            # tfidf = vectorizer.fit_transform(article[1])
            # article_adjm = (tfidf * tfidf.T).A
            article_adjm = torch.ones(len(article[1]), len(article[1]))
            article_adjm[ind[0], ind[1]] = 0 # zero out diag
            article_adjms.append(article_adjm)

            total_len += len(article[1])

            article_sentences.append(list())
            article_sentence_tokens.append(list())

            for sentence in article[1]:
                article_sentences[-1].append(sentence)
                # article_sentence_tokens[-1].append(tokenize_sentence(sentence.lower()))
                sentence_to_article[sentence_idx] = article_idx
                sentence_idx += 1
                num_sentences += 1

                # add to word id dict
                words = tokenize_sentence(sentence.lower())
                article_sentence_tokens[-1].append(words) # lowered now
                for word in words:
                    if word not in vocab:
                        vocab[word] = 0
                    vocab[word] += 1

        num_stns = sentence_idx

        ### SUPPORTING FACTS
        flatten_supporing_sentences = [0] * num_stns # zeros and ones (flags) say this sentence is supporting
        # print(flatten_supporing_sentences)
        try:
            for supporting_fact in entry["supporting_facts"]:
                title = supporting_fact[0]
                sentence_num = supporting_fact[1]
                flatten_index = title_to_sentence[title] + sentence_num
                flatten_supporing_sentences[flatten_index] = 1
        except:
            continue

        qa_pair = {
            'id': entry_id,
            'question': question,
            'answer': answer,
            'flatten_supporing_sentences': flatten_supporing_sentences,
            'article_sentences': article_sentences,
            'article_sentence_tokens': article_sentence_tokens
        }

        if not os.path.isdir(os.path.join(output_dir, data_type + "_text")):
            os.mkdir(os.path.join(output_dir, data_type + "_text"))
        text_file = os.path.join(output_dir, data_type + "_text", qa_pair['id'] + ".json")
        
        with open(text_file, 'w+') as open_file:
            open_file.write(json.dumps(qa_pair))

        ##########
        # CONSTRUCT ADJ MATRIX
        ##########
        # ind = np.diag_indices(num_sentences)

        # doc_adjm = torch.zeros(total_len, total_len)
        # start_idx = 0
        # end_idx = 0
        # for adjm_idx, adjm in enumerate(article_adjms):
        #     end_idx += len(adjm)
        #     doc_adjm[start_idx:end_idx, start_idx:end_idx] = adjm
        #     start_idx += len(adjm)

        # rot_ind = np.diag_indices(num_sentences - 1)
        # doc_adjm[rot_ind[0], rot_ind[1] + 1] = 1
        # doc_adjm[-1][0] = 1

        # qa_pair['adjm'] = doc_adjm
        # adjm_file = os.path.join(output_dir, data_type + "_adjm", qa_pair['id'] + ".adjm")

        # with open(adjm_file, 'w+') as open_file:
        #     open_file.write(json.dumps(doc_adjm.tolist()))

    vocab_file = os.path.join(output_dir, "hotpotqa_vocab.txt")
    vocab_list_of_list = list()
    for word, count in vocab.items():
        entry = [word, count]
        vocab_list_of_list.append(entry)
    vocab_list_of_list = sorted(vocab_list_of_list, key=lambda tup: tup[1], reverse=True)
    with open(vocab_file, 'w+', encoding='utf-8') as open_file:
        open_file.write(json.dumps(vocab_list_of_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None, type=str, help="data file")
    parser.add_argument("--data_type", default="train", type=str, help="train, valid, test")
    parser.add_argument("--output_dir", default=None, type=str, help="directory to output")
    parser.add_argument("--tokenization_type", default='spacy', choices=['vanilla','spacy'], type=str, help="tokenization type")
    args = parser.parse_args()

    read_hotpotqa_data(args)

if __name__ == "__main__":
    main()

