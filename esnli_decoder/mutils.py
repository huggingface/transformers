# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import inspect
from torch import optim
import numpy as np
import os
import csv
from nltk.translate.bleu_score import corpus_bleu


##############################################################################
# from data_esnli
'''
NLI_DIC_LABELS = {'entailment': 2,  'neutral': 1, 'contradiction': 0}

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
    print data_path, data_type, len(s1['sent'])
   
    data = {'s1': s1['sent'], 's2': s2['sent'], 'label': target_label['data'], 'expl_1': expl_1['sent'], 'expl_2': expl_2['sent'], 'expl_3': expl_3['sent']}

    return data
'''
##############################################################################

def get_dir(file):
    directory = "."
    f = file.split("/")
    if len(f) == 1:
        return directory
    for i in range(len(f) - 1):
        if i == 0:
            directory = f[i]
        else:
            directory += "/" + f[i]
    return directory


def new_file_name(file, prefix):
    directory = "."
    f = file.split("/")
    if len(f) == 1:
        return directory
    for i in range(len(f) - 1):
        if i == 0:
            directory = f[i]
        else:
            directory += "/" + f[i]
    new_file = directory + "/" + prefix +"__" + file 
    return new_file



def permute(x, perm):
    perm_x = []
    for i in perm:
        perm_x.append(x[i])
    return perm_x
    

def bleu_prediction(pred_file, data):
    candidates = []
    references = []

    f = open(pred_file)
    pred_reader = csv.DictReader(f)
    k = -1
    for row in pred_reader:
        k += 1
        if k < len(data['expl_1']):
            prediction = row['pred_expl'].strip().split()
            candidates.append(prediction)
            current_refs = []
            for j in range(1, 3):
                current_refs.append(data['expl_' + str(j)][k].strip().split())
            if k % 5000 == 0:
                print 'refs: ', current_refs
            references.append(current_refs)
            if k == 3:
                print "candidates ", candidates
                print "references ", references, '\n\n\n'
    
    bleu_score = corpus_bleu(references, candidates)
    print 'bleu: ', bleu_score
    f.close()
    return bleu_score


def bleu_inter_annotations_expl3_wrt_12(data):
    candidates = []
    references = []

    for k in range(len(data['expl_1'])):
        candidates.append(data['expl_3'][k])
        current_refs = []
        for j in range(1, 3):
            current_refs.append(data['expl_' + str(j)][k])
        references.append(current_refs)
    
    bleu_score = 100 * corpus_bleu(references, candidates)
    print 'bleu: ', bleu_score
    return round(bleu_score, 2)


def remove_file(file):
    try:
        os.remove(file)
    except Exception as e:
        print("\n\nCouldn't remove " + file + " because ", e)
        pass


def n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""
    import os, errno
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def pretty_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def get_key_from_val(val, dic_labels):
    for k, v in dic_labels.iteritems():
        if v == val:
            return k
    raise NameError("invalid value " + str(val))


def get_keys_from_vals(vals, dic_labels):
    assert_sizes(vals, 2, [1, vals.size(1)])
    keys = []
    for i in range(vals.size(1)):
        val = vals[0][i]
        found = False
        for k, v in dic_labels.iteritems():
            if v == val:
                keys.append(k)
                found = True
                break
        if not found:
            print "vals", vals
            raise NameError("invalid value " + str(val))
    return keys


def get_sentence_from_indices(dictionary, tensor_indices):
    s = ''
    n = len(tensor_indices)
    for i in range(n):
        if i == 0:
            s = get_key_from_val(tensor_indices.data[i], dictionary)
        else:
            s = s + ' ' + get_key_from_val(tensor_indices.data[i], dictionary)
    return s


def get_bow_expl_from_indices(out_vocab, tensor_indices):
    s = ''
    for i in range(len(tensor_indices)):
        if tensor_indices[i].data[0] >= 0.5:
            s += ' ' + get_key_from_val(i, out_vocab)
    return s.strip()


def bow_correct(pred, tgt):
    assert_same_dims(pred, tgt)
    assert len(pred.size()) == 2, str(len(pred.size()))

    bool_pred = pred > 0.5
    bool_tgt = tgt > 0.5

    agree = (bool_tgt == bool_pred).float()
    sum_row_agree = agree.data.sum(1)
    assert_sizes(sum_row_agree, 1, [pred.size(0)])
    #print "agree", agree
    assert sum_row_agree.sum() == agree.sum().data[0], str(sum_row_agree.sum()) + " while " + str(agree.sum().data[0])
    return agree.sum().data[0]


def bow_correct_per_row(pred, tgt):
    assert_same_dims(pred, tgt)
    assert len(pred.size()) == 2, str(len(pred.size()))

    bool_pred = pred > 0.5
    bool_tgt = tgt > 0.5

    agree = (bool_tgt == bool_pred).float()
    sum_row_agree = agree.data.sum(1)
    assert_sizes(sum_row_agree, 1, [pred.size(0)])
    #print "pred", pred
    #print "tgt", tgt
    #print "sum_row_agree", sum_row_agree
    assert sum_row_agree.sum() == bow_correct(pred, tgt), str(sum_row_agree.sum()) + " while " + str(bow_correct(pred, tgt))
    return sum_row_agree


def bow_precision_recall_fscore_sum(pred, tgt):
    assert_same_dims(pred, tgt)
    assert len(pred.size()) == 2, str(len(pred.size()))

    precision_row, recall_row, fscore_row = bow_precision_recall_fscore_row(pred, tgt)

    return precision_row.sum(), recall_row.sum(), fscore_row.sum()


def bow_precision_recall_fscore_row(pred, tgt):
    assert_same_dims(pred, tgt)
    assert len(pred.size()) == 2, str(len(pred.size()))

    bool_pred = pred.data > 0.5
    bool_tgt = tgt.data > 0.5

    pred_correct = (bool_pred & bool_tgt).float()
    n_pred_correct = pred_correct.sum(1)

    n_pred = bool_pred.float().sum(1)
    n_correct = bool_tgt.float().sum(1)

    precision = n_pred_correct / (n_pred + 1e-6)
    recall = n_pred_correct / (n_correct + 1e-6)

    fscore = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, fscore


def assert_sizes(t, dims, sizes):
    assert len(t.size()) == dims, "input dim: " + str(len(t.size())) + " required dim " + str(dims)
    for i in range(dims):
        assert t.size(i) == sizes[i], "in size " + str(i) + " given " + str(t.size(i)) + " expected " + str(sizes[i])


def assert_same_dims(x, y):
    assert len(x.size()) == len(y.size()), str(len(x.size())) + " vs " + str(len(y.size()))
    for i in range(len(x.size())):
        assert x.size(i) == y.size(i), "for dim " + str(i) + " x has " + str(x.size(i)) + " y has " + str(y.size(i))


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":

    import torch
    from torch.autograd import Variable

    x = Variable(torch.zeros(2,3))
    x[0,0] = 0.7
    x[0,1] = 0.6
    x[1,2] = 0.51
    print "x", x
    y = Variable(torch.zeros(2,3))
    y[0,0] = 0.8
    y[1,1] = 0.9
    print "y", y
    print(bow_precision_recall_fscore_row(x, y))


    '''
    # compute BLEU inter-annotators
    preproc = "preproc1_"
    snli_test = get_dev_test_with_expl("../dataset/eSNLI/", 'test', preproc, 0)

    for split in ['expl_1', 'expl_2', 'expl_3']:
        for data_type in ['snli_test']:
            eval(data_type)[split] = np.array([[word for word in sent.split()] for sent in eval(data_type)[split]])

    #bleu_inter_annotations_expl3_wrt_12(snli_dev)
    bleu_inter_annotations_expl3_wrt_12(snli_test)
    '''



