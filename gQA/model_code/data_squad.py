import argparse
import string
import json
import re
import random
import spacy
import torch
import numpy as np
import scipy.sparse as sp
import pickle
import torchtext
import os
import sys

from functools import reduce
from torch import nn

#Used for attention forcing. Soft cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.blank('en')

'''
final experiment
change to spacy tokenization (preprocess more)
change adjm from cos sim to
every sent connect to next sent, last sent in doc connect to first sent in doc
'''

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()

D_WORD_EMB = 300

vocab = ' ' + string.ascii_letters + string.digits + string.punctuation

character2id = {c:idx for idx, c in enumerate(vocab)}
UNK_CHAR = len(character2id)
CHAR_VOCAB_SIZE = len(vocab) + 1

#Pulled directly from PGMMR. Vocab size there was 50000
WORD_VOCAB_SIZE = 50000
SOS = 0
EOS = 1
PAD_WORD = 2
UNK_WORD = 3

word2id = {'SOS' : 0, 'EOS' : 1, 'PAD': PAD_WORD, 'UNK': UNK_WORD}
id2word = ['SOS', 'EOS', 'PAD', 'UNK']

embedding = torch.zeros((WORD_VOCAB_SIZE, D_WORD_EMB))
vocab = torchtext.vocab.GloVe(name='6B', dim=D_WORD_EMB)
word_count = WORD_VOCAB_SIZE

flatten = lambda l: [item for sublist in l for item in sublist]


def load_word_vocab(vocab_file):
    sorted_word = json.load(open(vocab_file))
    word_count = min(WORD_VOCAB_SIZE, len(sorted_word))
    words = [t[0] for t in sorted_word[:word_count]]
    global word2id
    i = 4
    for w in words:
        word2id[w.lower()] = i
        i += 1
        #id2word.append(w.lower())
        if i == WORD_VOCAB_SIZE:
            break
    cnt = 0
    for w,_id in word2id.items():
        if w not in vocab.stoi:
            cnt += 1
        embedding[_id] = vocab[w]
    global id2word
    id2word = {v: k for k, v in word2id.items()}
    print("%d words don't have pretrained embedding" % cnt)


def get_word_id(w):
    w = w.lower()
    return word2id[w] if w in word2id else UNK_WORD

def get_char_id(c):
    return character2id[c] if c in character2id else UNK_CHAR

def repeat(f, n):
    def rfun(p):
        return reduce(lambda x, _: f(x), range(n), p)
    return rfun

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.maximum(np.array(adj.sum(1)), 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_gold_embedding(arr):
    gold_embedding = np.zeros((len(arr[0]), D_WORD_EMB))
    for elem, _id in arr[0]:
        gold_embedding[_id] = vocab[elem]
    return gold_embedding

def flattenmax(inp_arr):
    to_process = [s for s in inp_arr]
    return(max(map(lambda x: len(flatten(x)), to_process)))

def vanilla_tokenize(sentence):
    tok_sentence = re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", sentence)
    return tok_sentence

def spacy_tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

def local_num2text_squad(inp, length, oov, test=False):
    to_return = []
    for indx in inp:
        int_indx = int(indx)
        if int_indx >= WORD_VOCAB_SIZE:
            to_return.append(oov[int_indx - WORD_VOCAB_SIZE])
        else:
            to_return.append(id2word[int_indx])
        if to_return[-1] == "EOS" and test:
            #Try just incase to_return is only one character. In that case we still want to see the EOS
            try:
                return to_return[0:len(to_return) - 1]
            except:
                continue
    if test:
        return to_return
    else:
        return to_return[0:int(length)]

def read_squad_files(path, files, args):
    count = 0
    batch_count = 0
    batch_id = 0

    batches = []
    cur_batch = []
    for squad in files:
        file = squad.name
        with open(os.path.join(path, file)) as json_file:
            data = json.load(json_file)
            cur_batch.append(SquadQA(data, args, True))
        count += 1
        batch_count += 1
        if batch_count >= args.batch:
            batch = Batch(cur_batch, batch_id)
            batches.append(Batch(cur_batch, batch_id))
            batch_id += 1
            batch_count = 0
            cur_batch = []
    return batches


class squad_short_hand():
    def __init__(self, name, id):
        self.name = name
        self.id = id


'''
need attention forcing -> add code -> read overleaf -> positive attention forcing with supporting sentences, overleaf describes negative attention forcing
word_data incorrect (trailing values should be pad)
char_mask good
char_data good -> get rid of 
answer_word_extended  (should have same value, if it has oov, has oov from original data)
    sos answer eos pad pad ... 15
answer_mask incorrect (just look at it)
answer_mask: (1 for all non pad terms)
answer_mask_unk: (1 if word in oov)


the thing with gQA, attention forcing
if gSUM gets accepted, we have gQA, they are helping each other.
'''

'''
answer_word_extended  (should have same value, if it has oov, has oov from original data)
    sos answer eos pad pad ... 15

change answer to gold
'''
class SquadQA():
    def __init__(self, obj, args=None, pointer_generation=False):

        self.id = obj['id']

        self.name = str(obj['id']) + ".json"

        self.gold = obj['answer_text'] # now tokenized and lower()'ed at preproc step

        self.question = obj['question'] # now tokenized and lower()'ed at preproc step

        self.articles = [obj['sentence_list']]

        self.article_tokens = obj['document']
        self.sentences_flat = list()
        for article in self.articles:
            self.sentences_flat.extend(article)
        self.sentences_flat.append(' '.join(self.question))

        self.sentence_tokens_flat = list()
        for article_token in self.article_tokens:
            self.sentence_tokens_flat.extend(article_token)
        self.sentence_tokens_flat.append(self.question)

        # gqa mark3, create adjm on the fly

        # with open(adjm_dir + str(self.id) + ".adjm", 'r') as open_file:
        #     sub_adjm = json.loads(open_file.read())
        # sub_adjm = torch.FloatTensor(sub_adjm)

        # number of sentences total in this data point
        self.sent_count = len(self.sentences_flat)
        # max number of words in a sentence
        self.max_sent_len = 0
        for sent_tokens in self.sentence_tokens_flat:
            self.max_sent_len = max(self.max_sent_len, len(sent_tokens))
        self.max_sent_len = min(self.max_sent_len, args.max_sentence_len)

            
        # max number of characters in a word
        self.max_word_len = 0
        for sent_tokens in self.sentence_tokens_flat:
            for token in sent_tokens:
                self.max_word_len = max(self.max_word_len, len(token))
        self.max_word_len = min(self.max_word_len, args.max_word_len)
        # question_length
        self.question_len = len(self.question)

        # Account for OOV words
        self.oov = list()

        #########################
        ### construct data points for question
        #########################
        question_char = np.full((self.question_len, self.max_word_len), 0, dtype=int)
        question_char_mask = np.zeros((self.question_len, self.max_word_len), dtype=int)
        question_word = np.full((self.question_len), PAD_WORD, dtype=int)
        question_word_mask = np.full((self.question_len), 0, dtype=int)
        question_word_extended = np.full((self.question_len), PAD_WORD, dtype=int)

        # test for num questions with > 64 words
        num_question_over_words = 0
        for i, q_token in enumerate(self.question):
            if i >= args.max_question_len:
                num_question_over_words += 1
                break
            question_word[i] = get_word_id(q_token)
            question_word_extended[i] = question_word[i]
            question_word_mask[i] = 1
            if question_word_extended[i] == UNK_WORD:
                if q_token not in self.oov:
                    self.oov.append(q_token)
                question_word_extended[i] = WORD_VOCAB_SIZE + self.oov.index(q_token)

            for j, q_char in enumerate(q_token):
                if j >= self.max_word_len:
                    break
                question_char[i][j] = get_char_id(q_char)
                question_char_mask[i, j] = 1

        length_q = np.array([self.question_len])
        self.question_char = torch.from_numpy(question_char)
        self.question_char_mask = torch.from_numpy(question_char_mask)
        self.question_word = torch.from_numpy(question_word)
        self.question_word_mask = torch.from_numpy(question_word_mask)
        self.question_word_extended = torch.from_numpy(question_word_extended)

        self.length_q = torch.from_numpy(np.array(len(self.question)))

        #########################
        ### construct data points documents
        #########################

        ### construct mask for character level (num sentences x num words x num characters 3d array)
        data_char = np.full((self.sent_count, self.max_sent_len, self.max_word_len), 0, dtype=int)
        data_char_mask = np.zeros((self.sent_count, self.max_sent_len, self.max_word_len), dtype=int)

        ### construct mask for word level (num sentences x num words 2d array) one/zeros
        data_word = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)
        data_word_mask = np.full((self.sent_count, self.max_sent_len), 0, dtype=int)
        data_word_extended = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)

        for i, sent_words in enumerate(self.sentence_tokens_flat):
            for j in range(len(sent_words)):
                if j >= self.max_sent_len:
                    break
                data_word_mask[i, j] = 1
                data_word[i, j] = get_word_id(sent_words[j])
                
                # copy extended vocab for PG
                data_word_extended[i, j] = data_word[i, j]
                if data_word_extended[i, j] == UNK_WORD:
                    if sent_words[j] not in self.oov:
                        self.oov.append(sent_words[j])
                    data_word_extended[i, j] = WORD_VOCAB_SIZE + self.oov.index(sent_words[j])

                for k, c in enumerate(sent_words[j]):
                    if k >= self.max_word_len:
                        continue
                    else:
                        data_char_mask[i][j][k] = 1
                        data_char[i, j, k] = get_char_id(c)

        #########################
        ### construct adjacency matrix
        #########################

        article_adjms = list() # what do you think this is lol
        total_len = 0

        for article_idx, article in enumerate(self.articles):
            total_len += len(article)
        
        #########################
        ### construct data points for answer
        #########################
        gold_word = np.zeros(len(self.gold) + 2, dtype = int) # literally answer word encodings
        gold_word_extended = np.zeros(len(self.gold)  + 2, dtype = int) # above, with oov's
        gold_word_mask = np.full(len(self.gold) + 2, 1, dtype = int) # mask for non-oov words
        gold_word_mask_unk = np.full(len(self.gold) + 2, 0, dtype = int) # mask for oov words

        gold_word[0] = SOS
        gold_word_extended[0] = SOS

        for i in range(len(self.gold)):
            gold_word[i + 1] = get_word_id(self.gold[i].lower())
            gold_word_extended[i + 1] = get_word_id(self.gold[i].lower())
            if pointer_generation:
                if gold_word_extended[i + 1] == UNK_WORD:
                    if self.gold[i].lower() in self.oov:
                        gold_word_mask_unk[i + 1] = 1
                        gold_word_extended[i + 1] = WORD_VOCAB_SIZE + self.oov.index(self.gold[i].lower())

        gold_word[-1] = EOS
        gold_word_extended[-1] = EOS

        #########################
        ### construct data points for similarity score
        #########################
        # sent_sim = np.zeros(self.max_sent_len, dtype = float)
        ##### experiment 2019/09/11
        # sentence_cat_questions = [s.lower() + " " + ' '.join(self.question) for s in self.sentences_flat]
        # sentence_cat_question_vectors = count_vectorizer.fit_transform(sentence_cat_questions)
        # sentence_vectors = count_vectorizer.transform(self.sentences_flat)
        # self.all_to_all = cosine_similarity(sentence_cat_question_vectors, sentence_vectors)
        ##### experiment is to use rotated diagonal, with end of doc -> begin of doc

        self.all_to_all = np.zeros((total_len + 1, total_len + 1))





        # print(f"article_vectors: {article_vectors}")
        # print(f"question_vectors: {question_vectors}")

        ### list of sentence lengths
        self.length_s = np.array(list(map(lambda sentence: min(args.max_sentence_len, sentence), data_word_mask.sum(axis=1))))
        self.max_sent_len = max(self.length_s)
        self.length_s = torch.from_numpy(self.length_s)

        word_arr = list(map(lambda i: local_num2text_squad(data_word_extended[i], self.length_s[i], self.oov), 
        range(len(data_word_extended))))
        word_arr = [x for x in word_arr if x]
        words = list(map(lambda strings: " ".join(strings), word_arr))

        documents_count = count_vectorizer.fit_transform(words)
        enrich = np.array(cosine_similarity(documents_count, documents_count))
        self.adjm = np.eye(total_len + 1, k = 1) + (0.25) * (enrich - np.eye(total_len + 1))
        self.adjm = torch.from_numpy(self.adjm)
        ### Attention forcing disabled for squad
        self.sent_sim = np.array([1] * self.sent_count)

        ss_temp = np.random.uniform(size=len(self.sent_sim))
        ss_index = np.argsort(ss_temp)

        for i in ss_index[0:min(len(ss_index), args.sample_size)]:
            self.sent_sim[i] += 1

        self.sent_sim = torch.from_numpy(self.sent_sim)
        self.sent_attn_mask = torch.full(self.sent_sim.size(), 1)

        # sent_attn_mask_pos = list(map(lambda x: 1 if x > (1. - s_mask) else 0, sent_sim))

        self.gold_word = torch.from_numpy(gold_word)
        self.gold_word_extended = torch.from_numpy(gold_word_extended)
        self.gold_mask = torch.from_numpy(gold_word_mask)
        self.gold_mask_unk = torch.from_numpy(gold_word_mask_unk)

        self.data_char = torch.from_numpy(data_char)
        self.data_word = torch.from_numpy(data_word)
        self.data_word_mask = torch.from_numpy(data_word_mask)
        self.data_word_extended = torch.from_numpy(data_word_extended)

        # self.all_to_all = np.full((self.sent_count, self.sent_count), 1., dtype=float) - np.eye(self.sent_count)
        # self.all_to_all = torch.from_numpy(self.all_to_all)

class Batch:
    def __init__(self, squad_batch, _id):
        self.id = _id
        self.squad_batch = squad_batch
        self.batch_size = len(squad_batch)
        self.max_num_sent = max([x.sent_count for x in self.squad_batch])
        self.max_question_len = max([x.question_len for x in self.squad_batch])
        self.max_sent_len = max([x.max_sent_len for x in self.squad_batch])
        self.max_word_len = max([x.max_word_len for x in self.squad_batch])
        self.max_gold_len = max([len(x.gold_word) for x in self.squad_batch])
        self.max_oov_len = max([len(x.oov) for x in self.squad_batch])
        self.max_length_q = max(x.question_len for x in self.squad_batch)
        self.oov = [x.oov for x in self.squad_batch]
        self.extra_zeros = None
        if self.max_oov_len > 0:
            self.extra_zeros = torch.zeros((self.batch_size, self.max_oov_len))

        self.question_word = torch.stack([
                torch.LongTensor(np.pad(x.question_word, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.question_word_ex = torch.stack([
                torch.LongTensor(np.pad(x.question_word_extended, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.question_word_mask = torch.stack([
                torch.LongTensor(np.pad(x.question_word_mask, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.question_char = torch.stack([
                torch.LongTensor(np.pad(x.question_char,
                    pad_width=[(0, self.max_question_len-x.question_len), (0, self.max_word_len-x.max_word_len)],
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.question_char_mask = torch.stack([
                torch.LongTensor(np.pad(x.question_char_mask, 
                    pad_width=[(0, self.max_question_len-x.question_len), (0, self.max_word_len-x.max_word_len)], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.data_word_ex = torch.stack([
                torch.LongTensor(np.pad(x.data_word_extended, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.data_char = torch.stack([
                torch.LongTensor(np.pad(x.data_char, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len), (0, self.max_word_len-x.max_word_len)],
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.data_mask = torch.stack([
                torch.LongTensor(np.pad(x.data_word_mask, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len)], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.gold_word = torch.stack([
                torch.LongTensor(np.pad(x.gold_word, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_word))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.gold_word_extended = torch.stack([
                torch.LongTensor(np.pad(x.gold_word_extended, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_word_extended))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.squad_batch])

        self.gold_mask = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_mask))], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.gold_mask_unk = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask_unk, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_mask_unk))], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.length_s = torch.stack([
                torch.from_numpy(np.pad(x.length_s,
                    pad_width=[(0, self.max_num_sent - len(x.length_s))], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.length_q = torch.stack([
                torch.from_numpy(np.pad(x.length_q,
                    pad_width=[(0, self.max_length_q-x.question_len)], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.adjm = torch.stack([
                torch.FloatTensor(np.pad(x.adjm,
                    pad_width=[(0,self.max_num_sent-x.adjm.size()[0]),(0,self.max_num_sent-x.adjm.size()[0])],
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.all_to_all = torch.stack([
                torch.FloatTensor(np.pad(x.all_to_all,
                    pad_width=[(0,self.max_num_sent-x.sent_count),(0,self.max_num_sent-x.sent_count)],
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.sent_sim = torch.stack([
                torch.FloatTensor(np.pad(x.sent_sim,
                    pad_width=[(0, self.max_num_sent - len(x.sent_sim))], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.sent_attn_mask = torch.stack([
                torch.FloatTensor(np.pad(x.sent_attn_mask,
                    pad_width=[(0, self.max_num_sent - len(x.sent_attn_mask))], 
                    mode='constant', constant_values=0))
                for x in self.squad_batch])

        self.gold = list(map(lambda topic: topic.gold, self.squad_batch))
        self.name = list(map(lambda topic: topic.name, self.squad_batch))
        self.word_embd = None

    def embed(self, embeddings):
        batch_size, docu_len, sent_len = self.data_word.size()
        self.word_embd = embeddings(self.data_word.view(batch_size*docu_len, -1))
        self.word_embd = self.word_embd.view(batch_size * docu_len, sent_len, -1)


class DataSet():

    def __init__(self, json_file, args):
        self.char_vocab_size = CHAR_VOCAB_SIZE
        self.args = args
        self.squad_file = json_file
        self.squad_file_list = os.listdir(self.squad_file)

        self.squad_files = list()
        self.squad_file_list.sort()
        for i in range(len(self.squad_file_list)):
            self.squad_files.append(squad_short_hand(self.squad_file_list[i], i))

        self.data = list()
        self.train = self.data
        self.valid = list()
        self.test = list()

        self.order = None

    # tested + works
    def split_train_valid_test(self, ratio):
        n = len(self.squad_files)
        if not self.order:
            self.order = list(range(n))
            random.shuffle(self.order)
        split_list = list()
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])

        self.train = [self.squad_files[i] for i in self.order[:train_size]]
        self.valid = [self.squad_files[i] for i in self.order[train_size:train_size+valid_size]]
        self.test = [self.squad_files[i] for i in self.order[train_size+valid_size:]]

        obj = {}
        obj['train'] = [t.id  for t in self.train]
        obj['valid'] = [t.id  for t in self.valid]
        obj['test']  = [t.id  for t in self.test]

        assert len(obj['train']) != 0
        assert len(obj['valid']) != 0
        assert len(obj['test']) != 0

        split_list.append(obj)

        with open('squad_data/squad_split.json', 'w') as open_file:
            json.dump(split_list, open_file)

    def split_dataset(self, filename, _id):
        f = open(filename)
        split_list = json.load(f)
        f.close()
        obj = split_list[_id]
        uid2topic = {t.id: t for t in self.squad_files}
        self.train = [uid2topic[uid] for uid in obj['train']]
        self.valid = [uid2topic[uid] for uid in obj['valid']]
        self.test  = [uid2topic[uid] for uid in obj['test']]

    def get_training_item(self, index, embedding, delta = 1, pgen = None):
        to_process = read_squad_files(self.squad_file, 
        self.train[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process

    def get_test_item(self, index, embedding, delta = 1, pgen = None):
        to_process = read_squad_files(self.squad_file, 
        self.test[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process

    def get_eval_item(self, index, embedding, delta = 1, pgen = None):
        to_process = read_squad_files(self.squad_file, 
        self.valid[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process

def read_squad_data(vocab_file, json_file, args):
    load_word_vocab(vocab_file)
    return DataSet(json_file, args)