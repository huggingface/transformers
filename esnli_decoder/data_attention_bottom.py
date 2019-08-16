import os
import numpy as np
import torch
import shutil
import sys
from mutils import makedirs

NLI_DIC_LABELS = {'entailment': 2,  'neutral': 1, 'contradiction': 0}


# batch  : list of sentences, where a sentences is a list of words
# output : tuple (torch.FloatTensor(T, bs, 300), array_sentence_lengths)
def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    for i in range(len(batch)):
        for j in range(min(len(batch[i]), max_len)):
            if batch[i][j] in word_vec:
                embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


# Doesn't include <s> because we don't output this
def get_target_expl_batch(batch, word_index):
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    batch_indexes = np.full((max_len - 1, len(batch)), word_index["<p>"])

    for i in range(len(batch)):
        # batch[i] from MultiNLI has length 2 because it contains only <s> and </s> so by ignoring it we let the target be <p> which will be ignored from backprop
        if len(batch[i]) > 2:
            for j in range(1, len(batch[i])): # exclude <s>
                if batch[i][j] in word_index:
                    batch_indexes[j - 1, i] = word_index[batch[i][j]]

    return torch.from_numpy(batch_indexes).long(), lengths - 1 


# sentences : array of strings
# output    : dictionary of all words + <s> + </s> + <p>   
def get_word_dict(sentences):
    # create vocab of words
    word_index = {}
    word_index['<p>'] = 0 
    word_index['</s>'] = 1
    word_index['<s>'] = 2
    word_index['<UNK>'] = 3
    i = 3
    for sent in sentences:
        for word in sent.split():
            if word not in word_index:
                i += 1
                word_index[word] = i
    return word_index


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))

    print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def copy_first_k_lines_txt(file, new_file, k):
    f = open(file)
    content = f.readlines()

    g = open(new_file, 'w')

    i = 0
    for line in content:
        i += 1
        if i <= k:
            g.write(line)
        else:
            break
    f.close()
    g.close()


def get_dev_test_original_expl(data_path, data_type):
    assert data_type in ['dev', 'test']

    expl_1, expl_2, expl_3 = {}, {}, {}
    expl_1['path'] = os.path.join(data_path, 'preproc1_expl_1.' + data_type)
    expl_2['path'] = os.path.join(data_path, 'preproc1_expl_2.' + data_type)
    expl_3['path'] = os.path.join(data_path, 'preproc1_expl_3.' + data_type)

    expl_1['sent'] = [line.rstrip() for line in open(expl_1['path'], 'r')]
    expl_2['sent'] = [line.rstrip() for line in open(expl_2['path'], 'r')]
    expl_3['sent'] = [line.rstrip() for line in open(expl_3['path'], 'r')]                         

    assert len(expl_1['sent']) == len(expl_2['sent']) == len(expl_3['sent'])
    print( data_path, data_type, len(expl_1['sent']))
   
    data = {'expl_1': expl_1['sent'], 'expl_2': expl_2['sent'], 'expl_3': expl_3['sent']}

    return data


def get_train(data_path, preproc, min_freq, n_train):
    #assert 'eSNLI' in data_path or 'ALLeNLI' in data_path

    data_path_train = data_path
    if n_train != -1:
        data_path_train = os.path.join(data_path, "train_" + str(n_train) + "_freq" + str(min_freq))
        if os.path.exists(data_path_train):
            shutil.rmtree(data_path_train)
        # create subset of n_train data from train only
        makedirs(data_path_train)
        for file in ["s1.train", "s2.train", "UNK_freq_" + str(min_freq) + "_preproc1_expl_1.train", "labels.train"]:
            copy_first_k_lines_txt(os.path.join(data_path, file), os.path.join(data_path_train, file), n_train)

    s1, s2, target_label, expl_1 = {}, {}, {}, {}

    freq_prefix = ""
    if min_freq > 0:
        freq_prefix = "UNK_freq_" + str(min_freq) + "_"


    data_type  = 'train'
    s1, s2, target_label, expl_1 = {}, {}, {}, {}
    s1['path'] = os.path.join(data_path_train, 's1.' + data_type)
    s2['path'] = os.path.join(data_path_train, 's2.' + data_type)
    expl_1['path'] = os.path.join(data_path_train, freq_prefix + preproc + 'expl_1.' + data_type)
    target_label['path'] = os.path.join(data_path_train, 'labels.' + data_type)

    s1['sent'] = [line.rstrip() for line in open(s1['path'], 'r')]
    s2['sent'] = [line.rstrip() for line in open(s2['path'], 'r')]
    expl_1['sent'] = [line.rstrip() for line in open(expl_1['path'], 'r')]
                  
    target_label['data'] = np.array([NLI_DIC_LABELS[line.rstrip('\n')] for line in open(target_label['path'], 'r')])

    assert len(s1['sent']) == len(s2['sent']) == len(target_label['data']) == len(expl_1['sent'])
    print( data_path, 'TRAIN ', len(s1['sent']))

    data = {'s1': s1['sent'], 's2': s2['sent'], 'label': target_label['data'], 'expl_1': expl_1['sent']}
    
    return data


def get_dev_test_with_expl(data_path, data_type, preproc, min_freq):
    assert data_type in ['dev', 'test']

    s1, s2, target_label, expl_1, expl_2, expl_3 = {}, {}, {}, {}, {}, {}

    freq_prefix = ""
    if min_freq > 0:
        freq_prefix = "UNK_freq_" + str(min_freq) + "_"


    s1['path'] = os.path.join(data_path, 's1.' + data_type)
    s2['path'] = os.path.join(data_path, 's2.' + data_type)
    expl_1['path'] = os.path.join(data_path, freq_prefix + preproc + 'expl_1.' + data_type)
    expl_2['path'] = os.path.join(data_path, freq_prefix + preproc + 'expl_2.' + data_type)
    expl_3['path'] = os.path.join(data_path, freq_prefix + preproc + 'expl_3.' + data_type)
    target_label['path'] = os.path.join(data_path, 'labels.' + data_type)

    s1['sent'] = [line.rstrip() for line in open(s1['path'], 'r')]
    s2['sent'] = [line.rstrip() for line in open(s2['path'], 'r')]
    expl_1['sent'] = [line.rstrip() for line in open(expl_1['path'], 'r')]
    expl_2['sent'] = [line.rstrip() for line in open(expl_2['path'], 'r')]
    expl_3['sent'] = [line.rstrip() for line in open(expl_3['path'], 'r')]                         
    target_label['data'] = np.array([NLI_DIC_LABELS[line.rstrip('\n')] for line in open(target_label['path'], 'r')])

    assert len(s1['sent']) == len(s2['sent']) == len(target_label['data']) == len(expl_1['sent']) == len(expl_2['sent']) == len(expl_3['sent'])
    print( data_path, data_type, len(s1['sent']))
   
    data = {'s1': s1['sent'], 's2': s2['sent'], 'label': target_label['data'], 'expl_1': expl_1['sent'], 'expl_2': expl_2['sent'], 'expl_3': expl_3['sent']}

    return data



def get_dev_or_test_without_expl(data_path, data_type, dic_labels):
    s1, s2, target_label = {}, {}, {}

    s1['path'] = os.path.join(data_path, 's1.' + data_type)
    s2['path'] = os.path.join(data_path, 's2.' + data_type)
    target_label['path'] = os.path.join(data_path, 'labels.' + data_type)

    s1['sent'] = [line.rstrip() for line in open(s1['path'], 'r')]
    s2['sent'] = [line.rstrip() for line in open(s2['path'], 'r')]
    target_label['data'] = np.array([dic_labels[line.rstrip('\n')] for line in open(target_label['path'], 'r')])

    assert len(s1['sent']) == len(s2['sent']) == len(target_label['data'])
    print( data_path, data_type.upper(), len(s1['sent']))
    
    data = {'s1': s1['sent'], 's2': s2['sent'], 'label': target_label['data']}
    
    return data
