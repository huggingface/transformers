# coding=utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import logging
import json
import random
import unicodedata
import sys
import regex as re

from pytorch_transformers import AutoTokenizer, PreTrainedTokenizer
from pytorch_transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_xlm import XLM_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_openai import OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_transfo_xl import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def ignore_variations(tokens):
    # xlnet can tokenize '111' as '1', '11' or '11', '1'
    # merge tokens that are all one character to not report this error
    merged_tokens = []
    prev_char_set = None
    for t in tokens:
        char_set = set(t)
        if prev_char_set and len(prev_char_set) == 1 and prev_char_set == char_set:
            merged_tokens[-1] = merged_tokens[-1] + t
        else:
            merged_tokens.append(t)
        prev_char_set = char_set
    return merged_tokens


class TokenizationCheck:
    def __init__(self, mask_token, equiv_required=True):
        self.equiv_required = equiv_required
        self.mask_token = mask_token
        self.limit_warn = 2
        self.show_diff = 2
        self.diffs_passed = 0
        self.diffs_failed = 0
        self.equiv_passed = 0
        self.equiv_failed = 0

    def tokens_vs_spans(self, tokenizer, tokens, offsets, text):
        def normalize(token):
            # normalize tokens, just to compare our spans in original text to the tokens
            token = unicodedata.normalize('NFKD', token)
            token = ''.join([c for c in token if not unicodedata.combining(c)])
            # unicode spaces
            token = re.sub(r'[\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', token)
            return token.lower().strip()

        def samey(tok, txt):
            return normalize(tok) == normalize(txt) or tok == '[UNK]' or tok == u'�'

        tok_strs = [tokenizer._detokenize_for_offsets(t) for t in tokens]
        text_strs = [text[offsets[i, 0]:offsets[i, 1]] for i in range(len(tokens))]
        if all([samey(tok, txt) for tok, txt in zip(tok_strs, text_strs)]):
            self.diffs_passed += 1
            return True
        if self.show_diff > 0:
            line_length_limit = 100
            tok_list = []
            txt_list = []
            cur_line_len = -1
            print('\nTokens and text spans have a difference:')
            print('=' * line_length_limit)
            print(text.replace('\n', ' '))
            print('=' * line_length_limit)
            for tok_str, text_str in zip(tok_strs, text_strs):
                if normalize(tok_str) == normalize(text_str):
                    tok_list.append(tok_str)
                    txt_list.append(' '*len(tok_str))
                else:
                    tlen = max(len(tok_str), len(text_str))
                    tok_list.append("'"+tok_str+' '*(tlen-len(tok_str))+"'")
                    txt_list.append("'"+text_str+' '*(tlen-len(text_str))+"'")
                cur_line_len += 1+len(tok_list[-1])
                if cur_line_len > line_length_limit:
                    print('-'*line_length_limit)
                    print(' '.join(tok_list[0:-1]))
                    print(' '.join(txt_list[0:-1]))
                    tok_list = tok_list[-1:]
                    txt_list = txt_list[-1:]
                    cur_line_len = len(tok_list[-1])
            print('-' * line_length_limit)
            print(' '.join(tok_list))
            print(' '.join(txt_list))
            print('-' * line_length_limit)
            self.show_diff -= 1
        self.diffs_failed += 1
        return False

    @staticmethod
    def mask_a_token(text, mask_token):
        if len(text) < 30:
            return text
        sndx = random.randint(0, len(text)-10)
        start = text.find(' ', sndx)
        while start < len(text) and text[start].isspace():
            start += 1
        end = text.find(' ', start+1)
        if start == -1 or end == -1:
            return text
        return text[:start] + mask_token + text[end:]

    def check_same_tokens(self, tokenizer, text, strict=False):
        # also check if we are handling special tokens ([MASK] etc) the same way
        if self.mask_token and random.random() < 0.5:
            text = self.mask_a_token(text, self.mask_token)
        tokens_orig = tokenizer.tokenize(text)
        tokens_offs, offsets = tokenizer.tokenize_with_offsets(text)
        if tokens_orig != tokens_offs and (strict or ignore_variations(tokens_orig) != ignore_variations(tokens_offs)):
            self.equiv_failed += 1
            if self.limit_warn > 0:
                linetext = text.replace('\n', ' ')
                print('tokenization is different for "'+linetext+'"\n  '+str(tokens_orig)+'\n  '+str(tokens_offs))
                self.limit_warn -= 1
            if self.equiv_required:
                raise ValueError
        else:
            self.equiv_passed += 1
        # we do expect some differences here, so this only shows a sample
        if random.random() < 0.01:
            self.tokens_vs_spans(tokenizer, tokens_offs, offsets, text)

    def show_stats(self):
        print('equivalence with tokenize %d passed and %d failed' % (self.equiv_passed, self.equiv_failed))
        print('subword tokens match text %d passed and %d failed' % (self.diffs_passed, self.diffs_failed))


def main(test_file):
    # python regression_test_tokenization_with_offsets.py tokenize_offset_test_data.jsonl
    # tokenize_offset_test_data.jsonl at https://ibm.box.com/s/228183fe95ptn8eb9n0zq4i2y7picq4r
    #   (Wikipedia paragraphs)
    # Note that GPT2 (and therefore RoBERTa do not work in python 2
    # also have issue with XLNet in python 2
    model_maps = [
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    ]
    still_need_work = [
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,  # omits '\n</w>' tokens
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,         # non-sentence final periods '... inc. reported ...'
    ]
    for model_map in model_maps:
        for model_name in list(model_map.keys())[:1]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # iterate over a jsonl test file, try tokenizing each document both with and without offsets
            # verify that they give the same tokens
            # also show differences in 'detokenized' tokens vs. the substring of text indicated by the offsets
            tokenizer_check = TokenizationCheck(tokenizer.mask_token, equiv_required=(model_map not in still_need_work))
            with open(test_file, 'r') as f:
                line_count = 0
                for line in f:
                    jobj = json.loads(line)
                    tokenizer_check.check_same_tokens(tokenizer, jobj['contents'])
                    line_count += 1
            tokenizer_check.show_stats()
            logger.info('tested %s on %d paragraphs' % (model_name, line_count))


if __name__ == "__main__":
    main(sys.argv[1])
