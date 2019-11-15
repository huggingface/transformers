#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# # Finish constructing the dataset

# In[3]:


flatten = lambda l: [item for sublist in l for item in sublist]

def load(file, direct = 'pre/opinosis/'):
    arr = []
    with open(direct + file) as opin:
        for line in opin:
            arr.append(line)
    return(arr)
def dirload(file, direct = 'pre/opinosis/'):
    arr = ""
    with open(direct + file) as opin:
        for line in opin:
            arr += line
    if arr == "":
        return None
    else:
        return arr

def fixdoubleindent(arr):      
    narr = np.array(arr)
    narr = narr[np.where(narr != '\n')]
    return narr


# In[4]:


def aggressive_fix_punc(string):
    string = string.replace('\n','')
    string = string.replace('.', '')
    string = string.replace('!', '')
    string = string.replace('?', '')
    string = string.replace(',', '')
    return(string)    
def fix_punc(string):
    string = string.replace('\n','')
    string = string.replace(' .', '')
    string = string.replace(' !', '')
    string = string.replace(' ?', '')
    string = string.replace(' ,', '')
    return(string)
def fix_punc2(string):
    string = string.replace('\n','')
    string = string.replace(' .', '. ')
    string = string.replace(' !', '! ')
    string = string.replace(' ?', '? ')
    string = string.replace(' ,', ', ')
    return(string)
def narr_fix_punc(narr):
    for i in range(len(narr)):
        narr[i] = fix_punc(narr[i])

    return(list(filter(("").__ne__, narr)))
def anarr_fix_punc(narr):
    for i in range(len(narr)):
        narr[i] = aggressive_fix_punc(narr[i])
    return(list(filter(("").__ne__, narr)))

# # Begin constructing the edges for our graph

# In[5]:



def computesim(nparr):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(nparr)

    sim_matrix = (tfidf * tfidf.T).A
    return(sim_matrix)

# In[5]:


#print(sim_matrix)

# In[6]:


#Exclude everything along the diagonal (and duplicates)
def simadj(sim_mat, thres = 1.5):
    nlen = sim_mat.shape[0]
    df = sim_mat[np.where(sim_mat != 1)]

    sd = statistics.stdev(df.flatten())
    mu = statistics.mean(df.flatten())

    out_matrx = np.zeros((nlen, nlen))
    #Could not get np.where working for this, so I did it manually
    #The threshold for determining if an element is an outlier
    for i in range(nlen):
        for j in range(nlen):
            if i != j:
                if sim_mat[i,j] > mu + thres * sd:
                    out_matrx[i,j] = 1
            #No need for <, as we only care about points on the other end of the extreme spectrum
    return(out_matrx)
#out_matrx is our adj matrix. I specifically avoided connecting the last sentence of a document
#to the first sentence of all others on this data set, since each documnt is only once sentence

#Actually I take the latter half of the above back, I figured out how the dataset is formatting different
#sentences of the same article. If the sentence does not begin with a space, then then it is from
#the same article
#print(np.where(out_matrx== 1))

# In[7]:


def advanced_punc_fix(string):
    return string 
    '''
    puncs = ['.','?','!']
    
    #Easiest case first
    if string[-1] not in puncs:
        string = string +' . '
    out = ''
    #Tokenize the text. If the first letter of a word is capital, insert a period before it
    for s in string.split():
        if s[0].isupper():
            out = out + '. ' + s + ' '
        else:
            out = out + s + ' '
    string = out[1:len(out)]
    return(string)
    '''
def lasttofirst_str(nparr):
    paragraphs = []
    string = ''
    for i in range(len(nparr)):
        string = string + nparr[i]
        if not nparr[i].startswith(' '):
            strs = aggressive_fix_punc(string).split()
            for j in range(len(strs)):
                strs[j] = strs[j].split('\'')
            strs = flatten(strs)
            paragraphs.append(strs)
            string = ''
    return paragraphs
def lasttofirst_indx(nparr):
    paragraphs = []
    for i in range(len(nparr)):
        if not nparr[i].startswith(' '):
            paragraphs.append(i)
    return paragraphs

def lasttofirst(nparr, out):
    
    paragraphs = lasttofirst_indx(nparr)
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

# # Now we can combine everything into a single function

# In[8]:


from pathlib import Path
def computeadjmatrices(path):
    adjmats = []
    files = os.listdir(path)
    max_size = 0
   
    for file in tqdm.tqdm(files):
        mFile = Path("graphs/opinosis/" + file.replace('.txt','') + ".csv")
        if mFile.is_file():
            continue
        nfile = load(file)
        if not nfile:
            continue
        try:
            nfile = narr_fix_punc(fixdoubleindent(nfile))
            out = simadj(computesim(nfile))
        
            out = lasttofirst(nfile, out)
            size = out.shape[0]
            out = out + np.eye(size,size,1)
        
            #This is where we can make sure no edge has been counted for multiple times, or any
            #other error if I think of it later
            for i in range(size):
                for j in range(size):
                    #We made an oopsie
                    if out[i,j] > 1:
                        out[i,j] = 1

            adjmats.append(out)

            np.savetxt("graphs/opinosis/" + file.replace('.txt','') + ".csv", out, delimiter=",")
        except:
            continue
            print(nfile)

        #max_size = max(max_size, len(nfile))
        
    return(adjmats)
#mats = computeadjmatrices("pre/opinosis")


# So in this dataset, the maximum adj size is 575. We can construct the GNN with this in mind.

# # We need to determine our vocabulary size

# In[12]:


import nltk
nltk.download('stopwords')
  
set(stopwords.words('english'))
#Go through all of the files and determine the vocabulary
def computeunq(path):
    words = []
    files = os.listdir(path)
    for file in tqdm.tqdm(files):
        nfile = load(file)
        if not nfile:
            continue
        try:
            nfile = narr_fix_punc(fixdoubleindent(nfile))
        
            for elem in nfile:
                words.append(word_tokenize(elem))
        except:
            continue
            
    flat_list = [item for sublist in words for item in sublist]
    words = list(set(flat_list))
    return(words)

#unq_words = computeunq("pre/opinosis")
#print(unq_words)

#print(len(unq_words))

# In[18]:

print(unq_words[:100])
count = dict()
for w in tqdm.tqdm(unq_words):
    count[w.lower()] = 1

def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

sorted_count = Reverse(sorted(count.items(), key=operator.itemgetter(1)))
with open("../sum_data/opinosis_vocab.json", 'w') as f:
    json.dump(sorted_count, f)
