import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from data_twitter import *
#from gnn_twitter import GNN_Twitter
import random
import json
import time
import math
import sys
#from tqdm import tqdm
import statistics
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import glob, os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

from gensim.models import FastText

import operator
import json

from bert_embedding import BertEmbedding
import mxnet as mx
import tqdm


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

flatten = lambda l: [item for sublist in l for item in sublist]

def load(file, direct = 'pre/opinosis/'):
    arr = []
    with open(direct + file) as opin:
        for line in opin:
            arr.append(line)
    return(arr)

def computesim(nparr):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(nparr)

    sim_matrix = (tfidf * tfidf.T).A
    return(sim_matrix)

#Exclude everything along the diagonal (and duplicates)
def simadj(sim_mat):
    nlen = sim_mat.shape[0]
    df = sim_mat[np.where(sim_mat != 1)]

    mu = statistics.mean(df.flatten())

    out_matrx = np.zeros((nlen, nlen))
    #Could not get np.where working for this, so I did it manually
    #The threshold for determining if an element is an outlier
    for i in range(nlen):
        for j in range(nlen):
            if i != j:
                if sim_mat[i,j] > mu:
                    out_matrx[i,j] = sim_mat[i,j]
            #No need for <, as we only care about points on the other end of the extreme spectrum
    return(out_matrx)

def lasttofirst(indx, out,nparr):
    
    paragraphs = indx
    #Make sure that the last sentence of our document points to 
    #the first sentence of every other document except
    #itself
    for i in paragraphs:
        for j in paragraphs:
            if i != j:
                #j is the first sentence and the last sentence 
                if(j - 1 in paragraphs or j == 0 or (j == len(nparr) - 1)):
                    out[i,j] = 1
                else:
                    out[i,j+1] = 1
    return(out)

def load_from_JSON(direct):
    full_stops = ['.', ' . ', '. ', ' .',
                  '?', ' ? ', '? ', ' ?',
                  '!', ' ! ', '! ', ' !']
    articles = []
    words = list()
    words.append(list())
    paragraphs = list()
    with open(direct) as json_file:
        data = json.load(json_file)
        obj = data['articles']
        sent_i = 0
        for i in range(len(obj)):
            articles.append([])
            for word in obj[i]:
                articles[i].append(word)
        #Compile everything into a set of sentences
        sent_count = sent_i
        num_articles = len(obj)
        sent_num = 0
        word_num = 0
        
        articles_split = list(map(lambda string: split_into_sentences(" ".join(string)), articles))

        par = 0
        for i  in range(len(articles_split)):
            par += len(articles_split[i])
            paragraphs.append(sent_count)
        #Get down to sentence level
        words = flatten(articles_split)
    return words, paragraphs
        


from tqdm import tnrange, tqdm_notebook


from pathlib import Path
def computeadjmatrices(path):
    adjmats = []
    files = os.listdir(path)
    max_size = 0
   
    for file in tqdm.tqdm(files):
        try:
            mFile = Path("../sum_data/opinosis_graphs/" + file.replace('.json','') + ".csv")
            #if mFile.is_file():
            #    continue
            words, paragraphs = load_from_JSON(path + file)
            out = computesim(words)
            out -= np.eye(out.shape[0])

            out = lasttofirst(paragraphs, out, words)
            size = out.shape[0]
            out = out + np.eye(size,size,1)

            #This is where we can make sure no edge has been counted for multiple times, or any
            #other error if I think of it later
            for i in range(size):
                for j in range(size):
                    #We made an oopsie
                    if out[i,j] > 1:
                        out[i,j] = 1
                    #Less than s_mask
                    if out[i,j] < .05:
                        out[i,j] = 0
            adjmats.append(out)

            np.savetxt("graphs/opinosis/" + file.replace('.json','') + ".csv", out, delimiter=",")
            #max_size = max(max_size, len(nfile))
        except:
            continue
        
    return(adjmats)

computeadjmatrices("../sum_data/final/")
