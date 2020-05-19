import argparse
import functools
import json
import logging
import math
import numpy as np
import os
import pickle
import time
import sys

from os.path import join as pjoin
from random import choice, randint
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from time import time
from tqdm import tqdm, trange

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk

import faiss

import torch

from transformers import AutoModel, AutoModelWithLMHead, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import nlp

###############
### Defining nlp datasets for: ELI5, KILT-wikipedia original and snippets, Wiki40B snippets
###############
_CITATION_ELI5 = """\
@inproceedings{DBLP:conf/acl/FanJPGWA19,
  author    = {Angela Fan and
               Yacine Jernite and
               Ethan Perez and
               David Grangier and
               Jason Weston and
               Michael Auli},
  editor    = {Anna Korhonen and
               David R. Traum and
               Llu{\'{\i}}s M{\`{a}}rquez},
  title     = {{ELI5:} Long Form Question Answering},
  booktitle = {Proceedings of the 57th Conference of the Association for Computational
               Linguistics, {ACL} 2019, Florence, Italy, July 28- August 2, 2019,
               Volume 1: Long Papers},
  pages     = {3558--3567},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://doi.org/10.18653/v1/p19-1346},
  doi       = {10.18653/v1/p19-1346},
}
"""

_DESCRIPTION_ELI5 = """\
Explain Like I'm 5 long form QA dataset
"""

_CITATION_KILT = """\
@inproceedings{fb_kilt,
  author    = {Fabio Petroni and
               Angela Fan and
               Sebastian Riedel},
  title     = {{KILT:} Knowledge Intensive Language Tasks},
  booktitle = {ArXiv},
  year      = {2020},
}
"""

_DESCRIPTION_KILT = """\
Wiki-KILT Wikipedia pre-processed for KILT/ELI5
"""

BUILDER_CONFIGS_ALL = [
            nlp.BuilderConfig(
                    name="plain_text",
                    version=nlp.Version("1.0.0", "New split API"),
                    description="Plain text",
            ),
    ]

class ELI5NLP(nlp.GeneratorBasedBuilder):
    """ELI5: Explain Like I'm Five long form question answering dataset."""
    _FILE_PATH = "/home/yacine/Code/transformers/examples/eli5/eli5_full_qa_with_qreps.pth"
    name = "eli5"
    BUILDER_CONFIGS = BUILDER_CONFIGS_ALL

    def _info(self):
        return nlp.DatasetInfo(
                description=_DESCRIPTION_ELI5,
                features=nlp.Features({
                        "q_id": nlp.Value("string"),
                        "title": nlp.Value("string"),
                        "selftext": nlp.Value("string"),
                        "answers":
                                nlp.features.Sequence({
                                        "a_id": nlp.Value("string"),
                                        "text": nlp.Value("string"),
                                        "score": nlp.Value("int32"),
                                }),
                        "title_urls": nlp.features.Sequence({"url": nlp.Value("string")}),
                        "selftext_urls": nlp.features.Sequence({"url": nlp.Value("string")}),
                        "answers_urls":  nlp.features.Sequence({"url": nlp.Value("string")}),
                }),
                supervised_keys=None,
                homepage="https://facebookresearch.github.io/ELI5/explore.html",
                citation=_CITATION_ELI5,
        )

    def _split_generators(self, dl_manager):
        self.torch_data = torch.load(self._FILE_PATH)
        return [
                nlp.SplitGenerator(
                        name=nlp.Split.TRAIN,
                        gen_kwargs={"split": "train"}),
                nlp.SplitGenerator(
                        name=nlp.Split.VALIDATION,
                        gen_kwargs={"split": "valid"}),
                nlp.SplitGenerator(
                        name=nlp.Split.TEST,
                        gen_kwargs={"split": "test"}),
        ]

    def _generate_examples(self, split):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", split)
        data = self.torch_data[split]
        for example in data:
            id_ = example['id']
            title = example['title']
            title_urls = example['title_urls']
            selftext = example['selftext']
            selftext_urls = example['selftext_urls']
            answer_scores = [ans['score'] for ans in example['answers']]
            # flatten list of URL mappings
            url_maps = [(ul, i, j) for i, ans in enumerate(example['answers']) for j, ul in enumerate(ans['text_urls'])]
            answers_urls = [ul for ul, _, _ in url_maps]
            map_url_indices = dict([((i, j), k) for k, (_, i, j) in enumerate(url_maps)])
            answer_texts = []
            for i, ans in enumerate(example['answers']):
                txt = ' ' + ans['text'] + ' '
                for j, _ in enumerate(ans['text_urls']):
                    txt = txt.replace(" URL_{} ".format(j), " URL_{} ".format(map_url_indices[(i, j)]))
                answer_texts += [txt.strip()]
            yield id_, {
                "q_id": id_,
                "title": title,
                "selftext": selftext,
                "answers": {
                    "a_id": ['' for _ in answer_texts], # TODO: fix!
                    "text": answer_texts,
                    "score": answer_scores,
                },
                "title_urls": {"url": title_urls},
                "selftext_urls": {"url": selftext_urls},
                "answers_urls": {"url": answers_urls},
            }

class WikiKILT(nlp.GeneratorBasedBuilder):
    """Wiki-KILT: Wikipedia for Knowledge Intensive Language Tasks."""
    name = "wiki_kilt"
    BUILDER_CONFIGS = BUILDER_CONFIGS_ALL

    def _info(self):
        return nlp.DatasetInfo(
                description=_DESCRIPTION,
                features=nlp.Features({
                        "title": nlp.Value("string"),
                        "kilt_id": nlp.Value("string"),
                        "paragraphs":
                                nlp.features.Sequence({
                                        "paragraph": nlp.Value("string"),
                                }),
                        "anchors":
                                nlp.features.Sequence({
                                        "text": nlp.Value("string"),
                                        "href": nlp.Value("string"),
                                        "paragraph_id": nlp.Value("int32"),
                                        "start": nlp.Value("int32"),
                                        "end": nlp.Value("int32"),
                                }),
                }),
                supervised_keys=None,
                homepage="https://facebookresearch.github.io/KILT",
                citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
                nlp.SplitGenerator(
                        name=nlp.Split.TRAIN,
                        gen_kwargs={"split": "train"}),
        ]

    def _generate_examples(self, split):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", split)
        kilt_file = "/home/yacine/Data/Wikipedia/kilt_knowledgesource.json"
        f = open(kilt_file)
        for line in f:
            example = json.loads(line.strip())
            id_ = 'kilt_' + example['_id']
            title = example['wikipedia_title']
            paragraphs = [par for par in example['text']]
            anchor_text =[a['text'] for a in example['anchors']]
            anchor_href = [a['href'] for a in example['anchors']]
            anchor_pid = [a['paragraph_id'] for a in example['anchors']]
            anchor_start = [a['start'] for a in example['anchors']]
            anchor_end = [a['end'] for a in example['anchors']]
            res_dct = {
                "title": title,
                "kilt_id": id_,
                "paragraphs": {"paragraph": paragraphs},
                "anchors": {
                    "text": anchor_text,
                    "href": anchor_href,
                    "paragraph_id": anchor_pid,
                    "start": anchor_start,
                    "end": anchor_end,
                },
            }
            yield id_, res_dct
        f.close()

def kilt_article_snippets(article, passage_len=100, overlap=0):
    paragraphs = article['paragraphs']['paragraph']
    par_tabs = [par.split(' ') for par in paragraphs]
    word_map = [(i, len(' '.join(par[:j])), w)
                for i, par in enumerate(par_tabs) for j, w in enumerate(par)]
    step_size = passage_len - overlap
    passages = []
    for i in range(math.ceil(len(word_map) // step_size)):
        pre_toks = word_map[i * step_size: i * step_size + passage_len]
        passage_text = ' '.join([w.replace('Section::::', '').replace(':', ' ') if w.startswith('Section::::') else w.replace('BULLET::::', '')
                                 for p_id, s_id, w in pre_toks])
        start_section_id = max([0] + [j for j, par in enumerate(paragraphs) if i <= pre_toks[0][0] and par.startswith('Section::::')])
        passages += [{
            'section_title': paragraphs[start_section_id].replace('Section::::', '').replace(':', ' -- '),
            'start_paragraph': pre_toks[0][0],
            'start_char_id': pre_toks[0][1],
            'end_paragraph': pre_toks[-1][0],
            'end_char_id': pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
            'passage_text': passage_text,
                     }]
    return passages

def kilt_generate_snippets(wikipedia, passage_len=100, overlap=0):
    for i, article in enumerate(wikipedia):
        article_title = article['title']
        for doc in kilt_article_snippets(article, passage_len, overlap):
            part_id = json.dumps(
                {
                    'nlp_id': i,
                    'kilt_id': article['kilt_id'],
                    'sp': doc['start_paragraph'],
                    'sc': doc['start_char_id'],
                    'ep': doc['end_paragraph'],
                    'ec': doc['end_char_id'],
                }
            )
            doc['article_title'] = article_title
            doc['_id'] = part_id
            doc['nlp_id'] = i
            doc['kilt_id'] = article['kilt_id']
            yield doc

class KiltSnippets(nlp.GeneratorBasedBuilder):
    name = "wiki_kilt_snippets_100w"
    BUILDER_CONFIGS = BUILDER_CONFIGS_ALL

    def _info(self):
        return nlp.DatasetInfo(
                description="Wiki-KILT Wikipedia pre-processed for KILT/ELI5 and split into 100-words snippets",
                features=nlp.Features({
                        "nlp_id": nlp.Value("int32"),
                        "kilt_id": nlp.Value("string"),
                        "start_paragraph": nlp.Value("int32"),
                        "start_character": nlp.Value("int32"),
                        "end_paragraph": nlp.Value("int32"),
                        "end_character": nlp.Value("int32"),
                        "article_title": nlp.Value("string"),
                        "section_title": nlp.Value("string"),
                        "passage_text": nlp.Value("string"),
                }),
                supervised_keys=None,
                homepage="https://facebookresearch.github.io/KILT",
                citation=_CITATION_KILT,
        )

    def _split_generators(self, dl_manager):
        kilt_dbuilder = WikiKILT(data_dir='wiki_kilt')
        kilt_dbuilder.download_and_prepare(ignore_checksums=True)
        self.kilt_dataset = kilt_dbuilder.as_dataset(split=nlp.splits.Split.TRAIN)
        return [
                nlp.SplitGenerator(
                        name=nlp.Split.TRAIN,
                        gen_kwargs={"split": "train"}),
        ]

    def _generate_examples(self, split):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", split)
        for doc in kilt_generate_snippets(self.kilt_dataset, passage_len=100, overlap=0):
            id_ = doc['_id']
            res_dct = {
                    "nlp_id": doc["nlp_id"],
                    "kilt_id": doc["kilt_id"],
                    "start_paragraph": doc["start_paragraph"],
                    "start_character": doc["start_char_id"],
                    "end_paragraph": doc["end_paragraph"],
                    "end_character": doc["end_char_id"],
                    "article_title": doc["article_title"],
                    "section_title": doc["section_title"],
                    "passage_text": doc["passage_text"],
            }
            yield id_, res_dct

def wiki40b_article_snippets(article, passage_len=100, overlap=0):
    paragraphs = article['text'].split('\n')
    aticle_idx = paragraphs.index('_START_ARTICLE_')
    article_title = paragraphs[aticle_idx] if aticle_idx < len(paragraphs) else ''
    section_indices = [i+1 for i, par in enumerate(paragraphs[:-1]) if par == '_START_SECTION_']
    par_tabs = [par.split(' ') for par in paragraphs]
    word_map = [(i, len(' '.join(par[:j])), w)
                for i, par in enumerate(par_tabs) if not par[0].startswith('_START_')
                for j, w in enumerate(par) if i > 0]
    step_size = passage_len - overlap
    passages = []
    for i in range(math.ceil(len(word_map) // step_size)):
        pre_toks = word_map[i * step_size: i * step_size + passage_len]
        start_section_id = max([0] + [j for j in section_indices if j <= pre_toks[0][0]])
        passage_text = ' '.join([w for p_id, s_id, w in pre_toks])
        passages += [{
            'article_title': article_title,
            'section_title': paragraphs[start_section_id],
            'start_paragraph': pre_toks[0][0],
            'start_char_id': pre_toks[0][1],
            'end_paragraph': pre_toks[-1][0],
            'end_char_id': pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
            'passage_text': passage_text,
                     }]
    return passages

def wiki40b_generate_snippets(article, passage_len=100, overlap=0):
     for i, article in enumerate(wikipedia):
        article_title = article['title']
        for doc in kilt_article_snippets(article, passage_len, overlap):
            part_id = json.dumps(
                {
                    'nlp_id': i,
                    'kilt_id': article['wikidata_id'],
                    'sp': doc['start_paragraph'],
                    'sc': doc['start_char_id'],
                    'ep': doc['end_paragraph'],
                    'ec': doc['end_char_id'],
                }
            )
            doc['_id'] = part_id
            doc['nlp_id'] = i
            doc['kilt_id'] = article['wikidata_id']
            yield doc

class Wiki40bSnippets(nlp.GeneratorBasedBuilder):
    name = "wiki40b_snippets_100w"
    BUILDER_CONFIGS = BUILDER_CONFIGS_ALL

    def _info(self):
        return nlp.DatasetInfo(
                description="Wiki-KILT Wikipedia pre-processed for KILT/ELI5 and split into 100-words snippets",
                features=nlp.Features({
                        "nlp_id": nlp.Value("int32"),
                        "kilt_id": nlp.Value("string"),
                        "start_paragraph": nlp.Value("int32"),
                        "start_character": nlp.Value("int32"),
                        "end_paragraph": nlp.Value("int32"),
                        "end_character": nlp.Value("int32"),
                        "article_title": nlp.Value("string"),
                        "section_title": nlp.Value("string"),
                        "passage_text": nlp.Value("string"),
                }),
                supervised_keys=None,
                homepage="https://facebookresearch.github.io/KILT",
                citation=_CITATION_KILT,
        )

    def _split_generators(self, dl_manager):
        self.wiki40b = nlp.load_dataset('wiki40b')
        return [
                nlp.SplitGenerator(
                        name=nlp.Split.TRAIN,
                        gen_kwargs={"split": "train"}),
        ]

    def _generate_examples(self, split):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", split)
        for k in ['train', 'validation', 'test']:
            for doc in wiki40b_generate_snippets(self.wiki40b[k], passage_len=100, overlap=0):
                id_ = doc['_id']
                res_dct = {
                        "nlp_id": doc["nlp_id"],
                        "kilt_id": doc["kilt_id"],
                        "start_paragraph": doc["start_paragraph"],
                        "start_character": doc["start_char_id"],
                        "end_paragraph": doc["end_paragraph"],
                        "end_character": doc["end_char_id"],
                        "article_title": doc["article_title"],
                        "section_title": doc["section_title"],
                        "passage_text": doc["passage_text"],
                }
                yield id_, res_dct

###############
### Sparse index
###############
def make_es_index_snippets(es_client, passages_dset, index_name='english_wiki_kilt_snippets_100w'):
    index_config = {
      "settings": {
        "number_of_shards": 1,
        "analysis": {
          "analyzer": {
            "stop_standard": {"type": "standard", " stopwords": "_english_"}
          }
        }
      },
      "mappings": {
        "properties": {
          "article_title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
          "section_title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
          "passage_text": {"type": "text", "analyzer": "standard", "similarity": "BM25"}
        }
      }
    }
    es_client.indices.create(index = index_name, body = index_config)
    number_of_docs = 23309001
    progress = tqdm(unit="docs", total=number_of_docs)
    successes = 0
    def passage_generator():
        for passage in passages_dset:
            yield passage
    # create the ES index
    for ok, action in streaming_bulk(
            client=es_client, index=index_name, actions=passage_generator(),
    ):
        progress.update(1)
        successes += ok
    print("Indexed %d documents" % (successes,))

def query_es_index(question, es_client, index_name='english_wiki_kilt_snippets_100w', n_results=10):
    q = question.lower()
    banned = ['how', 'why', 'what', 'where', 'which', 'do', 'does', 'is', '?', 'eli5', 'eli5:']
    q = ' '.join([w for w in q.split() if w not in banned])
    response = es_client.search(
        index = index_name,
        body = {
            "query": {
                "multi_match": {
                    "query": q,
                    "fields": ["article_title", "section_title", "passage_text^2"],
                    "type": "cross_fields",
                }
            },
            "size": n_results,
        }
    )
    hits = response['hits']['hits']
    support_doc = '<P> ' + ' <P> '.join([hit['_source']['passage_text'] for hit in hits])
    res_list = [dict([(k, hit['_source'][k]) for k in hit['_source'] if k != 'passage_text']) for hit in hits]
    for r, hit in zip(res_list, hits):
        r['passage_id'] = hit['_id']
        r['score'] = hit['_score']
    return support_doc, res_list

###############
### ELI5 retriever training
###############
class ELI5DatasetQARetriver(Dataset):

    def __init__(self, examples_array, extra_answer_threshold=3, min_answer_length=64, training=True):
        self.data = examples_array
        self.answer_thres = extra_answer_threshold
        self.min_length = min_answer_length
        self.training = training

    def __len__(self):
        return self.data.num_rows

    def make_example(self, idx):
        example = self.data[idx]
        question = example['title']
        if self.training:
            answers = [a for i, (a, sc) in enumerate(zip(example['answers']['text'], example['answers']['score']))]
            answer_tab = choice(answers).split(' ')
            start_idx = randint(0, max(0, len(answer_tab) - self.min_length))
            answer_span = ' '.join(answer_tab[start_idx:])
        else:
            answer_span = example['answers']['text'][0]
        return (question, answer_span)

    def __getitem__(self, idx):
        return self.make_example(idx)

class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(RetrievalQAEmbedder, self).__init__()
        self.sent_encoder = sent_encoder
        self.project_q = torch.nn.Linear(dim, 128, bias=False)
        self.project_a = torch.nn.Linear(dim, 128, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def embed_questions(self, q_ids, q_mask):
        _, q_reps = self.sent_encoder(q_ids, attention_mask=q_mask)
        return self.project_q(q_reps)

    def embed_answers(self, a_ids, a_mask):
        _, a_reps = self.sent_encoder(a_ids, attention_mask=a_mask)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask):
        device = next(self.parameters()).device
        q_reps = self.embed_questions(q_ids, q_mask)
        a_reps = self.embed_answers(a_ids, a_mask)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        loss = (loss_qa + loss_aq) / 2
        return loss

def make_qa_retriever_model(model_name="google/bert_uncased_L-8_H-512_A-8", from_file=None, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    qa_embedder = RetrievalQAEmbedder(bert_model, 512).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file) # has model weights, optimizer, and scheduler states
        qa_embedder.load_state_dict(param_dict['model'])
    return tokenizer, qa_embedder

def make_qa_retriever_batch(qa_list, tokenizer, max_len=64, device="cuda:0"):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=max_len, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    return (q_ids, q_mask, a_ids, a_mask)

def train_qa_retriever_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0):
    model.train()
    # make iterator
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_retriever_batch,
        tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch in enumerate(epoch_iterator):
        q_ids, q_mask, a_ids, a_mask = batch
        loss = model(q_ids, q_mask, a_ids, a_mask)
        # optimizer
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e, step,
                    len(dataset) // args.batch_size,
                    loc_loss / loc_steps,
                    time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0

def evaluate_qa_retriever(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    eval_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_retriever_batch,
        tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=eval_sampler, collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    tot_loss = 0.
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            q_ids, q_mask, a_ids, a_mask = batch
            loss = model(q_ids, q_mask, a_ids, a_mask)
            tot_loss += loss.item()
        return tot_loss / (step + 1)

###############
### ELI5 seq2seq model training
###############
class ELI5DatasetS2S(Dataset):

    def __init__(self, examples_array, make_doc_fun=None, extra_answer_threshold=3, document_cache=None, training=True):
        self.training = training
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        if self.training:
            self.qa_id_list = [(i, j)
                               for i, qa in enumerate(self.data) 
                               for j, (a, sc) in enumerate(zip(qa['answers']['text'], qa['answers']['score']))
                               if j == 0 or sc >= extra_answer_threshold]
        else:
            self.qa_id_list = [(i, 0) for i in range(self.data.num_rows)] 

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example['title'] + ' ' + example['selftext']
        answer = example['answers']['text'][j]
        q_id = example['q_id']
        if self.make_doc_function is not None:
            self.document_cache[q_id] = self.document_cache.get(q_id, self.make_doc_function(example['title']))
        document = self.document_cache[q_id]
        in_st = "question: {} context: {}".format(
            question.lower().replace(' --t--', '').strip(),
            document.lower().strip(),
        )
        out_st = answer
        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(idx)

def make_s2s_model(model_name="facebook/bart-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return tokenizer, model

def make_s2s_batch(qa_list, tokenizer, max_len=64, device="cuda:0"):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=64, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=64, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    return (q_ids, q_mask, a_ids, a_mask)



###############
### ELI5-trained retrieval model usage
###############
def embed_passages_for_retrieval(passage_list, tokenizer, qa_embedder, device='cuda:0'):
    a_ls = [p for p in passage_list['passage_text']]
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=128, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    with torch.no_grad():
        a_reps = qa_embedder.embed_answers(a_ids, a_mask).cpu().type(torch.float)
    return a_reps.numpy()

def embed_question_for_retrieval(q_ls, tokenizer, qa_embedder, device='cuda:0'):
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=128, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float)
    return q_reps.numpy()

def make_qa_dense_index(qa_embedder, tokenizer, passages_dset,
                        batch_size=512, index_name='kilt_passages_reps.dat',
                        device='cuda:0'):
    st_time = time()
    fp = np.memmap(index_name, dtype='float16', mode='w+', shape=(wiki_passages.num_rows, 128))
    n_batches = math.ceil(wiki_passages.num_rows / batch_size)
    for i in range(n_batches):
        reps = embed_passages_for_retrieval(passages_dset[i * batch_size:(i+1) * batch_size], tokenizer, qa_embedder, device)
        fp[i * batch_size:(i+1) * batch_size] = reps
        if i % 50 == 0:
            print(i, time() - st_time)

# build a support document for the question out of Wikipedia snippets
def query_qa_dense_index(question, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10):
    q_rep = embed_question_for_retrieval([question], tokenizer, qa_embedder)
    D, I = wiki_index.search(q_rep, n_results)
    res_passages = [wiki_passages[int(i)] for i in I[0]]
    support_doc = '<P> ' + ' <P> '.join([p['passage_text'] for p in res_passages])
    res_list = [dict([(k, p[k]) for k in wiki_passages.column_names if k != 'passage_text']) for p in res_passages]
    for r, sc in zip(res_list, D[0]):
        r['score'] = float(sc)
    return support_doc, res_list

def batch_query_qa_dense_index(questions, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10):
    q_rep = embed_question_for_retrieval(questions, tokenizer, qa_embedder)
    D, I = wiki_index.search(q_rep, n_results)
    res_passages_lst = [[wiki_passages[int(i)] for i in i_lst] for i_lst in I]
    support_doc_lst = ['<P> ' + ' <P> '.join([p['passage_text'] for p in res_passages])
                       for res_passages in res_passages_lst]
    all_res_lists = []
    for res_passages in res_passages_lst:
        res_list = [dict([(k, p[k]) for k in wiki_passages.column_names if k != 'passage_text'])
                    for p in res_passages]
        for r, sc in zip(res_list, D[0]):
            r['score'] = float(sc)
        all_res_lists += [res_list[:]]
    return support_doc_lst, all_res_lists

# find nearest neighbors of an answer or declarative text in Wikipedia snippets
def query_qa_dense_index_nn(passage, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10):
    a_rep = embed_passages_for_retrieval({'passage_text': [passage]}, tokenizer, qa_embedder)
    D, I = wiki_index.search(a_rep, n_results)
    res_passages = [wiki_passages[int(i)] for i in I[0]]
    support_doc = '<P> ' + ' <P> '.join([p['passage_text'] for p in res_passages])
    res_list = [dict([(k, p[k]) for k in wiki_passages.column_names if k != 'passage_text']) for p in res_passages]
    for r, sc, i in zip(res_list, D[0], I[0]):
        r['passage_id'] = int(i)
        r['score'] = float(sc)
    return support_doc, res_list

