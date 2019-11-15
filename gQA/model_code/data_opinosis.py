import string
import json
import random
from torch import nn
from functools import reduce
import torch
import numpy as np
import scipy.sparse as sp
import torchtext
import os
import sys

#Used for attention forcing. Soft cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pythonrouge.pythonrouge import Pythonrouge

#Convert data to states
from gQA.model_code.state import State


count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()

D_WORD_EMB = 300
MAX_SENTENCE_LEN = 64
MAX_GOLD_LEN = 16
def change_gold_len(val):
    global MAX_GOLD_LEN
    MAX_GOLD_LEN = val
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

vocab = ' ' + string.ascii_letters + string.digits + string.punctuation

character2id = {c:idx for idx, c in enumerate(vocab)}
UNK_CHAR = len(character2id)
CHAR_VOCAB_SIZE = len(vocab) + 1

def get_char_id(c):
    return character2id[c] if c in character2id else UNK_CHAR

#Pulled directly from PGMMR. Vocab size there was 50000
WORD_VOCAB_SIZE = 50000
SOS = 0
EOS = 1
PAD_WORD = 2
UNK_WORD = 3

word2id = {'SOS' : 0, 'EOS' : 1, 'PAD': PAD_WORD, 'UNK': UNK_WORD}
id2word = ['SOS', 'EOS', 'PAD', 'UNK']

#Load embeddings
embedding = torch.zeros((WORD_VOCAB_SIZE, D_WORD_EMB))
vocab = torchtext.vocab.GloVe(name='6B', dim=D_WORD_EMB)
word_count = WORD_VOCAB_SIZE

flatten = lambda l: [item for sublist in l for item in sublist]
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


def get_word_id(w):
    #w = w.lower()
    return word2id[w] if w in word2id else UNK_WORD
def get_gold_embedding(arr):
    gold_embedding = np.zeros((len(arr[0]), D_WORD_EMB))
    for elem, _id in arr[0]:
        gold_embedding[_id] = vocab[elem]
    return gold_embedding
MAX_WORD_LEN = 16
MAX_ARTICLE_LEN = 300
MAX_TOPICS = 32
def local_num2text(inp, length, oov, test=False):
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

def embed_sentence(inp):
    vec = torch.mean(torch.stack(list(map(lambda indx: embedding[indx], inp))), dim=0)

    norm = vec.norm(p=2, dim=0, keepdim=True)
    vec = vec.div(norm.expand_as(vec))
    return vec

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def flattenmax(inp_arr):
    to_process = [s for s in inp_arr]
    return(max(map(lambda x: len(flatten(x)), to_process)))

def calc_attn_score(sentence, gold):
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=[sentence], reference=[gold],
                            n_gram=1, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=True, stemming=True, stopwords=True,
                            word_level=True, length_limit=False, length=0,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=False, samples=10, favor=True, p=0.5)

        score = rouge.calc_score()
        return score['ROUGE-L']
class Topic():

    def __init__(self, obj, pointer_generation=False, s_mask = 0.05, gold_len = MAX_GOLD_LEN):
        full_stops = ['.', ' . ', '. ', ' .',
                    '?', ' ? ', '? ', ' ?',
                    '!', ' ! ', '! ', ' !']

        #id (just the index in the directory)
        self.id = obj['id']
        #Name of the topic (Just the file name)
        self.name = obj['name']
        #Gold standard for the summaries
        self.gold = obj['gold']

        self.articles = []

        #Used for positional embeddings
        i = 0
        j = 0
        wc = 0
        charc = 0

        #Load adj matrix
        self.adjm = np.loadtxt(open("/mnt/raid/gits/Graphical-Summarization/sum_data/opinosis_graphs/" + self.name + ".csv", "rb"), delimiter=",")
        self.sent_count = self.adjm.shape[0]
        #For backwards sequential connections
        self.adjm += np.eye(self.sent_count, k = -1)
        #Make sy
        self.adjm = torch.FloatTensor(1/2 * (self.adjm + np.transpose(self.adjm)))
        self.sent_count = self.adjm.size()[0]

        self.all_to_all = np.full((self.sent_count, self.sent_count), 1., dtype=float) - np.eye(self.sent_count)
        self.all_to_all = torch.from_numpy(self.all_to_all)

        sent_i = 0
        self.positions = np.zeros((self.sent_count, 4))
        for article in obj['articles']:
            self.articles.append([])
            #print(article)
            for word in article:
                self.articles[i].append(word)
                wc += 1
                charc += len(word)
                if word in full_stops:
                    j += 1
                    sent_i += 1
                    #One random sentence is filler, I don't know how to remove it besides either
                    #a) regenerating all of the data
                    #b) doing this
                    #doing this is easier
                    try:
                        self.positions[sent_i] = np.array([i, j, wc, charc])
                    except:
                        pass
            i += 1
            j = 0
            wc = 0
            charc = 0

        max_vec = np.max(self.positions, axis = 0) + [1, 1, 1, 1]
        #Construct positional embeddings for every sentence
        self.positions = self.positions/max_vec
        np_positions = list()
        for i in range(self.sent_count):
            np_positions.append(flatten([np.tile(np.sin(self.positions[i]), 4),
            np.tile(np.cos(self.positions[i]), 4)]))
        self.positions = np.array(np_positions)
        #self.sent_count = sent_i
        #self.positions = torch.from_numpy(self.positions)
        #Sentence to sample. Used in summarization
        #max_article_len is the maximum length of an article
        self.max_article_len = max([len(s) for s in self.articles])
        #We need to ravel the array and find the max length entry to determine the max word length
        self.max_word_len = max(list(map(lambda x: len(x), repeat(flatten, 1)(self.articles)))) - 1
        self.num_articles = len(self.articles)

        #Find the max length sentence
        sent_num = 0
        word_num = 0
        max_word_len = 0
        mask = np.zeros((self.sent_count, self.max_article_len), dtype=int)

        #print(self.articles)
        for article in self.articles:
            #print(article)
            sentences = split_into_sentences(" ".join(article))
            for i in range(self.sent_count):
                sent_words = sentences[i].split()
                for j in range(len(sent_words)):
                    mask[i, j] = 1
                    max_word_len = max(max_word_len, len(sent_words[j]))

        #Soft limit
        self.length_s = np.array(list(map(lambda sentence: min(MAX_SENTENCE_LEN, sentence), mask.sum(axis=1))))
        self.max_sent_len = max(self.length_s)
        self.length_s = torch.from_numpy(self.length_s)
        #We need to ravel the array and find the max length entry to determine the max word length
        self.max_word_len = min(max_word_len, MAX_WORD_LEN)
    
        data_char = np.full((self.sent_count, self.max_sent_len, self.max_word_len), PAD_WORD, dtype=int)
        data_word = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)
        data_word_extended = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)
        mask = np.full((self.sent_count, self.max_sent_len), 0, dtype=int)

        #Gold standard. + 2 is for SOS and trailing EOS
        summary_word = np.zeros((min(gold_len, len(self.gold)) + 2), dtype = int)
        summary_word_extended = np.zeros((min(gold_len, len(self.gold))  + 2), dtype = int)
        #The first mask is only nonzero for OOV words.
        summary_word_mask = np.full((min(gold_len, len(self.gold)) + 2), 1, dtype = int)
        #The second mask is one on alll nonOOV words
        summary_word_mask_unk = np.full((min(gold_len, len(self.gold)) + 2), 0, dtype = int)

        sent_num = 0
        word_num = 0
        #The following comment is outdated but was left in the code as a reminder.
        #Generate indexes for the sentences to be summarized
        #The current issue is that length is no longer being properly generated :(
        #For the third article, length determines that the maximum length of a sentence is 127 where as
        #our system determines that it is 552

        #Oh my god I am so dumb.... 552 is the max article length, not the max sentence length

        #RESOLVED
        #Account for OOV words
        self.oov = []
        words = [[""] * self.max_sent_len] * self.sent_count
        sent_num = 0
        word_num = 0
        #print(split_into_sentences(" ".join(self.articles[0])))
        #print(self.articles)
        for article in self.articles:
            #print(article)
            sentences = split_into_sentences(" ".join(article))
            for i in range(len(sentences)):
                sent_words = sentences[i].split()
                if i >= self.sent_count:
                    break
                for j in range(len(sent_words)):
                    if j >= self.max_sent_len:
                        break
                    mask[i, j] = 1
                    data_word[i, j] = get_word_id(sent_words[j])
                    words[i][j] = sent_words[j]

                    #Copy extended vocab for PG
                    data_word_extended[i, j] = data_word[i, j]
                    if data_word_extended[i, j] == UNK_WORD:
                        if sent_words[j] not in self.oov:
                            self.oov.append(sent_words[j])
                        data_word_extended[i, j] = WORD_VOCAB_SIZE + self.oov.index(sent_words[j])

                    for k, c in enumerate(sent_words[j]):
                        if k >= self.max_word_len:
                            continue
                        else:
                            data_char[i, j, k] = get_char_id(c)

        #Generate indexes for the gold standard. Assumes summary is only one sentence for now
        summary_word[0] = SOS
        summary_word_extended[0] = SOS
        for i in range(min(gold_len, len(self.gold))):
            summary_word[i+1] = get_word_id(self.gold[i])
            summary_word_extended[i+1] = summary_word[i+1]

            #Use extended vocab if PG is enabled
            if pointer_generation:
                if summary_word_extended[i+ 1] == UNK_WORD:
                    #Cross entropy does not careabout unknown words
                    if self.gold[i] in self.oov:
                        summary_word_mask_unk[i + 1] = 1
                        summary_word_extended[i+1] = WORD_VOCAB_SIZE + self.oov.index(self.gold[i])

        summary_word[-1] = EOS
        summary_word_extended[-1] = EOS

        self.data_char = torch.from_numpy(data_char)
        self.data_word = torch.from_numpy(data_word)
        self.data_word_extended = torch.from_numpy(data_word_extended)

        self.gold_word = torch.from_numpy(summary_word)
        self.gold_word_extended = torch.from_numpy(summary_word_extended)
        self.gold_mask = torch.from_numpy(summary_word_mask)
        self.gold_mask_unk = torch.from_numpy(summary_word_mask_unk)

        l = 1.0

        #Used for attention forcing. Compute redundency scores
        word_arr = list(map(lambda i: local_num2text(data_word_extended[i], self.length_s[i], self.oov), 
        range(len(data_word_extended))))

        word_arr.append(self.gold)
        words = list(map(lambda strings: " ".join(strings), word_arr))

        gold_sents = split_into_sentences(words[-1])
        gold_sent_count = len(gold_sents)
        words.pop(len(words) - 1)
        words.extend(gold_sents)

        documents_count = count_vectorizer.fit_transform(words)
        redundency_scores = np.transpose(np.array(cosine_similarity(documents_count, documents_count)[-gold_sent_count:]))
        redundency_scores = redundency_scores[0:self.sent_count]
        
        #Compute importance score
        word_arr = list(map(lambda i: local_num2text(data_word_extended[i], self.length_s[i], self.oov), 
        range(len(data_word_extended))))
        words = list(map(lambda strings: " ".join(strings), word_arr))
        #Append the entire document
        words.append(" ".join(words))
        documents_count = count_vectorizer.fit_transform(words)
        sent_sim = cosine_similarity(documents_count, documents_count)[-1]
        importance_scores = sent_sim[0:len(sent_sim) - 1]
        
        sent_scores = list()

        for i in range(self.sent_count):
            sent_score = (l) * importance_scores[i] - \
                (1 - l) * max(redundency_scores[i])
            sent_scores.append(sent_score)

        sent_scores = np.array(sent_scores)
        sent_scores = (sent_scores + min(0, min(sent_scores)))
        sent_scores = sent_scores / max(sent_scores)

        self.sent_sim = torch.from_numpy(sent_scores)
        sent_attn_mask_neg = list(map(lambda x: 1 if x < s_mask else 0, sent_scores))
        self.sent_attn_mask_neg = torch.from_numpy(np.array(sent_attn_mask_neg))
        
        sent_attn_mask_pos = list(map(lambda x: 1 if x > 1 - s_mask else 0, sent_scores))
        self.sent_attn_mask_pos = torch.from_numpy(np.array(sent_attn_mask_neg))
        
        self.mask = torch.from_numpy(mask)

class topic_short_hand():
    def __init__(self, name, id):
        self.name = name
        self.id = id

class Batch:
    def __init__(self, topics, id, gold_len = MAX_GOLD_LEN):
        self.id = id
        self.topics = topics
        self.batch_size = len(topics)
        self.max_num_sent = max([x.sent_count for x in self.topics])
        self.max_sent_len = max([x.max_sent_len for x in self.topics])
        max_word_len = max([x.max_word_len for x in self.topics])
        self.max_gold_len = max([min(gold_len, len(x.gold))  for x in self.topics])
        self.max_oov_len = max([len(x.oov) for x in self.topics])
        self.oov = [x.oov for x in self.topics]
        self.extra_zeros = None
        if self.max_oov_len > 0:
            self.extra_zeros = torch.zeros((self.batch_size, self.max_oov_len))

        self.data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.topics])
        self.data_word_ex = torch.stack([
                torch.LongTensor(np.pad(x.data_word_extended, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.topics])
        #Turns out positional embeddings didn't really help much
        '''

        '''
        self.data_char = torch.stack([
                torch.LongTensor(np.pad(x.data_char, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len), (0, max_word_len-x.max_word_len)],
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.data_mask = torch.stack([
                torch.LongTensor(np.pad(x.mask, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len)], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.gold_word = torch.stack([
                torch.LongTensor(np.pad(x.gold_word, 
                    pad_width=[(0, self.max_gold_len-min(gold_len, len(x.gold)))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.topics])

        self.gold_word_extended = torch.stack([
                torch.LongTensor(np.pad(x.gold_word_extended, 
                    pad_width=[(0, self.max_gold_len-min(gold_len, len(x.gold)))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.topics])

        self.gold_mask = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask, 
                    pad_width=[(0, self.max_gold_len-min(gold_len, len(x.gold)))], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.gold_mask_unk = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask_unk, 
                    pad_width=[(0, self.max_gold_len-min(gold_len, len(x.gold)))], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.length_s = torch.stack([
                torch.from_numpy(np.pad(x.length_s,
                    pad_width=[(0, self.max_num_sent-x.sent_count)], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.adjm = torch.stack([
                torch.FloatTensor(np.pad(x.adjm,
                    pad_width=[(0,self.max_num_sent-x.sent_count),(0,self.max_num_sent-x.sent_count)],
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.all_to_all = torch.stack([
                torch.FloatTensor(np.pad(x.all_to_all,
                    pad_width=[(0,self.max_num_sent-x.sent_count),(0,self.max_num_sent-x.sent_count)],
                    mode='constant', constant_values=0))
                for x in self.topics])  

        self.sent_sim = torch.stack([
                torch.FloatTensor(np.pad(x.sent_sim,
                    pad_width=[(0, self.max_num_sent-x.sent_count)], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.sent_attn_mask = torch.stack([
                torch.FloatTensor(np.pad(x.sent_attn_mask_pos,
                    pad_width=[(0, self.max_num_sent-x.sent_count)], 
                    mode='constant', constant_values=0))
                for x in self.topics])
                
        self.sent_attn_mask_neg = torch.stack([
                torch.FloatTensor(np.pad(x.sent_attn_mask_neg,
                    pad_width=[(0, self.max_num_sent-x.sent_count)], 
                    mode='constant', constant_values=0))
                for x in self.topics])

        self.gold = list(map(lambda topic: topic.gold, self.topics))
        self.name = list(map(lambda topic: topic.name, self.topics))
        self.word_embd = None
    def embed(self, embeddings):
        batch_size, docu_len, sent_len = self.data_word.size()
        self.word_embd = embeddings(self.data_word.view(batch_size*docu_len, -1))
        self.word_embd = self.word_embd.view(batch_size * docu_len, sent_len, -1)

    def convert_to_state(self):
        encoder_input = State(input=[self.data_word, self.data_mask], 
        other=[self.adjm, self.all_to_all, 
        self.length_s, None, self.data_word_ex, self.data_word.size()])

        return encoder_input

def read_topic_files(path, files, args, pgen=False):
    count = 0
    batch_count = 0
    batch_id = 0

    batches = []
    cur_batch = []
    for topic in files:
        file = topic.name
        with open(path + file) as json_file:
            data = json.load(json_file)
            cur_batch.append(Topic(data, pointer_generation=pgen, s_mask=args.s_mask, gold_len=args.gold_len))
        count += 1
        batch_count += 1
        if batch_count >= args.batch:
            batch = Batch(cur_batch, batch_id, gold_len=args.gold_len)
            #Reject articles that are too long
            if batch.max_num_sent < MAX_ARTICLE_LEN:
                batches.append(Batch(cur_batch, batch_id, gold_len=args.gold_len))

                batch_id += 1
                batch_count = 0
            cur_batch = []
    return(batches)


class DataSet():

    def __init__(self, topic_file, args):

        self.char_vocab_size = CHAR_VOCAB_SIZE
        self.args = args
        self.path = topic_file
        files = os.listdir(self.path)
        self.topic_files = []
        for i in range(len(files)):
            self.topic_files.append(topic_short_hand(files[i], i))

        self.data = []

        self.train = self.data
        self.valid = []
        self.test = []

        self.order = None

    def split_train_valid_test(self, ratio):
        n = len(self.topic_files)
        if not self.order:
            self.order = list(range(n))
            random.shuffle(self.order)

        f = open('/mnt/raid/gits/Graphical-Summarization/sum_data/opinosis_split.json', 'w')
        split_list = []

        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])
        self.train = [self.topic_files[i] for i in self.order[:train_size]]
        self.valid = [self.topic_files[i] for i in self.order[train_size:train_size+valid_size]]
        self.test = [self.topic_files[i] for i in self.order[train_size+valid_size:]]
        obj = {}
        obj['train'] = [t.id  for t in self.train]
        obj['valid'] = [t.id  for t in self.valid]
        obj['test']  = [t.id  for t in self.test]
        split_list.append(obj)
        json.dump(split_list, f)

    
    def split_dataset(self, filename, _id):
        f = open(filename)
        split_list = json.load(f)
        f.close()
        obj = split_list[_id]
        uid2topic = {t.id: t for t in self.topic_files}
        self.train = [uid2topic[uid] for uid in obj['train']]
        self.valid = [uid2topic[uid] for uid in obj['valid']]
        self.test  = [uid2topic[uid] for uid in obj['test']]

    def get_training_item(self, index, embedding, delta = 1, pgen=False):
        to_process = read_topic_files(self.path, 
        self.train[index * self.args.batch:(index + delta) * self.args.batch],
        self.args, pgen=pgen)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process
    def get_test_item(self, index, embedding, delta = 1, pgen=False):
        to_process = read_topic_files(self.path, 
        self.test[index * self.args.batch:(index + delta) * self.args.batch],
        self.args, pgen=pgen)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process
    def get_eval_item(self, index, embedding, delta = 1, pgen=False):
        to_process = read_topic_files(self.path, 
        self.valid[index * self.args.batch:(index + delta) * self.args.batch],
        self.args, pgen=pgen)
        for i in range(len(to_process)):
            to_process[i].embed(embedding)
        return to_process

def read_opinosis_data(topic_file, vocab_file, args):
    load_word_vocab(vocab_file)
    return DataSet(topic_file, args)
