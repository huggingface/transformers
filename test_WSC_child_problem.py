from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

import os
import json

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import tokenization, BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
from examples.extract_features import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

CONFIG_NAME = 'bert_config.json'
BERT_DIR = '/nas/pretrain-bert/pretrain-tensorflow/uncased_L-12_H-768_A-12/'
config_file = os.path.join(BERT_DIR, CONFIG_NAME)
config = BertConfig.from_json_file(config_file)
model = BertForPreTraining.from_pretrained(BERT_DIR)
model.eval()
class Args:
    def __init__(self):
        pass
    
args = Args()
args.no_cuda = False

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
model.to(device)

vis_attn_topk = 3


def has_chinese_label(labels):
    labels = [label.split('->')[0].strip() for label in labels]
    r = sum([len(label) > 1 for label in labels if label not in ['BOS', 'EOS']]) * 1. / (len(labels) - 1)
    return 0 < r < 0.5  # r == 0 means empty query labels used in self attention

def _plot_attn(ax1, attn_name, attn, key_labels, query_labels, col, color='b'):
    assert len(query_labels) == attn.size(0)
    assert len(key_labels) == attn.size(1)

    ax1.set_xlim([-1, 1])
    ax1.set_xticks([])
    ax2 = ax1.twinx()
    nlabels = max(len(key_labels), len(query_labels))
    pos = range(nlabels)
    
    if 'self' in attn_name and col < ncols - 1:
        query_labels = ['' for _ in query_labels]

    for ax, labels in [(ax1, key_labels), (ax2, query_labels)]:
        ax.set_yticks(pos)
        if has_chinese_label(labels):
            ax.set_yticklabels(labels, fontproperties=zhfont)
        else:
            ax.set_yticklabels(labels)
        ax.set_ylim([nlabels - 1, 0])
        ax.tick_params(width=0, labelsize='xx-large')

        for spine in ax.spines.values():
            spine.set_visible(False)

#     mask, attn = filter_attn(attn)
    for qi in range(attn.size(0)):
#         if not mask[qi]:
#             continue
#         for ki in range(attn.size(1)):
        for ki in attn[qi].topk(vis_attn_topk)[1]:
            a = attn[qi, ki]
            ax1.plot((-1, 1), (ki, qi), color, alpha=a)
#     print(attn.mean(dim=0).topk(5)[0])
#     ax1.barh(pos, attn.mean(dim=0).data.cpu().numpy())

def plot_layer_attn(result_tuple, attn_name='dec_self_attns', layer=0, heads=None):
    hypo, nheads, labels_dict = result_tuple
    key_labels, query_labels = labels_dict[attn_name]
    if heads is None:
        heads = range(nheads)
    else:
        nheads = len(heads)
    
    stride = 2 if attn_name == 'dec_enc_attns' else 1
    nlabels = max(len(key_labels), len(query_labels))
    rcParams['figure.figsize'] = 20, int(round(nlabels * stride * nheads / 8 * 1.0))
    
    rows = nheads // ncols * stride
    fig, axes = plt.subplots(rows, ncols)
    
    # for head in range(nheads):
    for head_i, head in enumerate(heads):
        row, col = head_i * stride // ncols, head_i * stride % ncols
        ax1 = axes[row, col]
        attn = hypo[attn_name][layer][head]
        _plot_attn(ax1, attn_name, attn, key_labels, query_labels, col)
        if attn_name == 'dec_enc_attns':
            col = col + 1
            axes[row, col].axis('off')  # next subfig acts as blank place holder
    # plt.suptitle('%s with %d heads, Layer %d' % (attn_name, nheads, layer), fontsize=20)
    plt.show()  
            
ncols = 4
import re
def convert_text_to_examples(text):
    examples = []
    unique_id = 0
    if True:
        for line in text:
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples

def convert_examples_to_features(examples, tokenizer, append_special_tokens=True, replace_mask=True, print_info=False):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        tokens = []
        input_type_ids = []
        if append_special_tokens:
            tokens.append("[CLS]")
            input_type_ids.append(0)
        for token in tokens_a:
            if replace_mask and token == '_':  # XD
                token = "[MASK]"
            tokens.append(token)
            input_type_ids.append(0)
        if append_special_tokens:
            tokens.append("[SEP]")
            input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                if replace_mask and token == '_':  # XD
                    token = "[MASK]"
                tokens.append(token)
                input_type_ids.append(1)
            if append_special_tokens:
                tokens.append("[SEP]")
                input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def copy_and_mask_features(features):
    import copy
    masked_feature_copies = []
    for feature in features:
        for masked_pos in range(len(feature.tokens)):
            feature_copy = copy.deepcopy(feature)
            feature_copy.input_ids[masked_pos] = tokenizer.vocab["[MASK]"]
            masked_feature_copies.append(feature_copy)
    return masked_feature_copies

def show_lm_probs(tokens, input_ids, probs, topk=5, firstk=20):
    def print_pair(token, prob, end_str='', hit_mark=' '):
        if i < firstk:
            # token = token.replace('</w>', '').replace('\n', '/n')
            print('{}{: >3} | {: <12}'.format(hit_mark, int(round(prob*100)), token), end=end_str)
    
    ret = None
    for i in range(len(tokens)):
        ind_ = input_ids[i].item() if input_ids is not None else tokenizer.vocab[tokens[i]]
        prob_ = probs[i][ind_].item()
        print_pair(tokens[i], prob_, end_str='\t')
        values, indices = probs[i].topk(topk)
        top_pairs = []
        for j in range(topk):
            ind, prob = indices[j].item(), values[j].item()
            hit_mark = '*' if ind == ind_ else ' '
            token = tokenizer.ids_to_tokens[ind]
            print_pair(token, prob, hit_mark=hit_mark, end_str='' if j < topk - 1 else '\n')
            top_pairs.append((token, prob))
        if tokens[i] == "[MASK]":
            ret = top_pairs
    return ret

import colored
from colored import stylize

def show_abnormals(tokens, probs, show_suggestions=False):
    def gap2color(gap):
        if gap <= 5:
            return 'yellow_1'
        elif gap <= 10:
            return 'orange_1'
        else:
            return 'red_1'
        
    def print_token(token, suggestion, gap):
        if gap == 0:
            print(stylize(token + ' ', colored.fg('white') + colored.bg('black')), end='')
        else:
            print(stylize(token, colored.fg(gap2color(gap)) + colored.bg('black')), end='')
            if show_suggestions and gap > 5:
                print(stylize('/' + suggestion + ' ', colored.fg('green' if gap > 10 else 'cyan') + colored.bg('black')), end='')
            else:
                print(stylize(' ', colored.fg(gap2color(gap)) + colored.bg('black')), end='')
                # print('/' + suggestion, end=' ')
            # print('%.2f' % gap, end=' ')
        
    avg_gap = 0.
    for i in range(1, len(tokens) - 1):  # skip first [CLS] and last [SEP]
        ind_ = tokenizer.vocab[tokens[i]]
        prob_ = probs[i][ind_].item()
        top_prob = probs[i].max().item()
        top_ind = probs[i].argmax().item()
        gap = math.log(top_prob) - math.log(prob_)
        suggestion = tokenizer.ids_to_tokens[top_ind]
        print_token(tokens[i], suggestion, gap)
        avg_gap += gap
    avg_gap /= (len(tokens) - 2)
    print()
    print(avg_gap)

analyzed_cache = {}

def analyze_text(text, show_suggestions=False, show_firstk_probs=20):
    if text[0] in analyzed_cache:
        features, mlm_probs = analyzed_cache[text[0]]
        given_mask = "[MASK]" in features[0].tokens
    else:
        examples = convert_text_to_examples(text)
        features = convert_examples_to_features(examples, tokenizer, print_info=False)
        given_mask = "[MASK]" in features[0].tokens
        if not given_mask:
            features = copy_and_mask_features(features)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
        input_ids = input_ids.to(device)
        input_type_ids = input_type_ids.to(device)

        mlm_logits, _ = model(input_ids, input_type_ids)
        mlm_probs = F.softmax(mlm_logits, dim=-1)

        if not given_mask:
            seq_len, _, vocab_size = mlm_probs.size()
            reduced_mlm_probs = torch.Tensor(1, seq_len, vocab_size)
            for i in range(seq_len):
                reduced_mlm_probs[0, i] = mlm_probs[i, i]
            mlm_probs = reduced_mlm_probs
        
        analyzed_cache[text[0]] = (features, mlm_probs)
        
    top_pairs = show_lm_probs(features[0].tokens, None, mlm_probs[0], firstk=show_firstk_probs)
    if not given_mask:
        show_abnormals(features[0].tokens, mlm_probs[0], show_suggestions=show_suggestions)
    return top_pairs


def test_by_WSC_child_problem():
    import json
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WSC_child_problem.json') 
    with open(path, 'r') as f:
        data_l = json.load(f)
    f.close()
    result = []
    for data in data_l:
        if data['sentences'] != []:
            for s in data['sentences']:
                s['predict_answer'] = []
                res = analyze_text([s['sentence']], show_firstk_probs=-1)
                for r in res:
                    if re.findall(r[0], str(s['correct_answer']), flags=re.IGNORECASE):
                        s['predict_answer'].append(str(r))
        result.append(data)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WSC_child_problem_e.json') 
    with open(path, 'w') as f:
         json.dump(result, f, indent=4, separators=(',', ': '), ensure_ascii=False)
    f.close()


test_by_WSC_child_problem()
