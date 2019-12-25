# coding=utf-8

import logging
import json
import random
import unicodedata
import sys
import re
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers.modeling_albert import ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_ctrl import CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlm import XLM_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_openai import OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_transfo_xl import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_MAP

logging.basicConfig(
    format="%(filename)s:%(lineno)d - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
            token = unicodedata.normalize("NFKD", token)
            token = "".join([c for c in token if not unicodedata.combining(c)])
            # unicode spaces
            token = re.sub(
                r"[\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000]",
                " ",
                token,
            )
            return token.lower().strip()

        def samey(tok, txt):
            return normalize(tok) == normalize(txt) or tok == "[UNK]" or tok == u"�"

        tok_strs = [tokenizer._detokenize_for_offsets(t) for t in tokens]
        text_strs = [text[offsets[i] : offsets[i + 1]] for i in range(len(tokens) - 1)] + [
            text[offsets[len(tokens) - 1] :]
        ]
        if all([samey(tok, txt) for tok, txt in zip(tok_strs, text_strs)]):
            self.diffs_passed += 1
            return True
        if self.show_diff > 0:
            line_length_limit = 100
            tok_list = []
            txt_list = []
            cur_line_len = -1
            print("\nTokens and text spans have a difference:")
            print("=" * line_length_limit)
            print(text.replace("\n", " "))
            print("=" * line_length_limit)
            for tok_str, text_str in zip(tok_strs, text_strs):
                if normalize(tok_str) == normalize(text_str):
                    tok_list.append(tok_str)
                    txt_list.append(" " * len(tok_str))
                else:
                    tlen = max(len(tok_str), len(text_str))
                    tok_list.append("'" + tok_str + " " * (tlen - len(tok_str)) + "'")
                    txt_list.append("'" + text_str + " " * (tlen - len(text_str)) + "'")
                cur_line_len += 1 + len(tok_list[-1])
                if cur_line_len > line_length_limit:
                    print("-" * line_length_limit)
                    print(" ".join(tok_list[0:-1]))
                    print(" ".join(txt_list[0:-1]))
                    tok_list = tok_list[-1:]
                    txt_list = txt_list[-1:]
                    cur_line_len = len(tok_list[-1])
            print("-" * line_length_limit)
            print(" ".join(tok_list))
            print(" ".join(txt_list))
            print("-" * line_length_limit)
            self.show_diff -= 1
        self.diffs_failed += 1
        return False

    @staticmethod
    def mask_a_token(text, mask_token):
        if len(text) < 30:
            return text
        sndx = random.randint(0, len(text) - 10)
        start = text.find(" ", sndx)
        while start < len(text) and text[start].isspace():
            start += 1
        end = text.find(" ", start + 1)
        if start == -1 or end == -1:
            return text
        return text[:start] + mask_token + text[end:]

    def check_same_tokens(self, tokenizer, text, strict=False):
        # also check if we are handling special tokens ([MASK] etc) the same way
        if self.mask_token and random.random() < 0.5:
            text = self.mask_a_token(text, self.mask_token)
        tokens_orig = tokenizer.tokenize(text)
        tokens_offs, offsets = tokenizer.tokenize_with_offsets(text)
        if tokens_orig != tokens_offs:
            self.equiv_failed += 1
            if self.limit_warn > 0:
                linetext = text.replace("\n", " ")
                print(
                    'tokenization is different for "'
                    + linetext
                    + '"\n  '
                    + str(tokens_orig)
                    + "\n  "
                    + str(tokens_offs)
                )
                self.limit_warn -= 1
            if self.equiv_required:
                raise ValueError
        else:
            self.equiv_passed += 1
        # we do expect some differences here, so this only shows a sample
        if random.random() < 0.01:
            self.tokens_vs_spans(tokenizer, tokens_offs, offsets, text)

    def show_stats(self):
        print("equivalence with tokenize %d passed and %d failed" % (self.equiv_passed, self.equiv_failed))
        print("subword tokens match text %d passed and %d failed" % (self.diffs_passed, self.diffs_failed))


def main(test_file):
    # python regression_test_tokenization_with_offsets.py drop_dataset_train.json
    # drop_dataset_train.json is downloadable from https://allennlp.org/drop
    model_maps = [
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
        T5_PRETRAINED_MODEL_ARCHIVE_MAP,
    ]
    # In XLM, "\"." is tokenized to [".", "\""] and similarly "\",",
    # so it's not possible to create contiguous offsets.
    # Also in XLM, "30\xa0000" (\xa0 is space) is tokenized to
    # ['3', '0.', '000</w>'] which yields offsets of [0, 1, 1].
    still_need_work = []
    for model_map in model_maps:
        for model_name in list(model_map.keys())[:1]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # iterate over a jsonl test file, try tokenizing each document both with and without offsets
            # verify that they give the same tokens
            # also show differences in 'detokenized' tokens vs. the substring of text indicated by the offsets
            tokenizer_check = TokenizationCheck(
                tokenizer.mask_token, equiv_required=(model_map not in still_need_work)
            )
            with open(test_file, encoding="utf8") as f:
                dataset = json.load(f)
                line_count = 0
                for passage_id, passage_info in tqdm(dataset.items()):
                    try:
                        tokenizer_check.check_same_tokens(tokenizer, passage_info["passage"])
                    except Exception as e:
                        logger.error(str(e))
                        print("passage_id=%s" % passage_id)
                        raise e
                    line_count += 1
                    for qa_pair in passage_info["qa_pairs"]:
                        try:
                            tokenizer_check.check_same_tokens(tokenizer, qa_pair["question"])
                        except Exception as e:
                            logger.error(str(e))
                            print("query_id=%s" % qa_pair["query_id"])
                            raise e
                        line_count += 1
            tokenizer_check.show_stats()
            logger.info("tested %s on %d paragraphs" % (model_name, line_count))


if __name__ == "__main__":
    main(sys.argv[1])
