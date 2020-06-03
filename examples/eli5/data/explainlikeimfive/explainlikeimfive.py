# coding=utf-8
# Copyright 2020 Facebook, Inc. and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ELI5: Long Form Question Answering dataset"""
from __future__ import absolute_import, division, print_function

import bz2
import io
import json
import logging
import lzma
import os
import re
from os.path import isfile
from os.path import join as pjoin
from time import time

import zstandard as zstd
from bs4 import BeautifulSoup

import nlp


_SUB_REDDITS = ["explainlikeimfive", "askscience", "AskHistorians"]
_REDDIT_URL = "https://files.pushshift.io/reddit/"

# pylint: disable=line-too-long
_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
# pylint: enable=line-too-long

_HTML_PAIRS = [
    ("&amp;", " & "),
    ("&quot", ' " '),
    ("&apos", " ' "),
    ("&gt;", " > "),
    ("&lt;", " < "),
]


# removes URLs (kept in separate list)
def _extract_urls_from_text(stp):
    url_list = list(set(re.findall(_URL_REGEX, stp)))
    for i, url in enumerate(url_list):
        stp = stp.replace(url, "_URL_%d_" % (i,))
    for a, b in _HTML_PAIRS:
        stp = stp.replace(a, b)
    return (stp, url_list)


# collects URLs for monthly dumps, has to be robust to file type changes
def _gather_dump_urls(base_url, mode, dl_manager):
    page_path = dl_manager.download(_REDDIT_URL + mode)
    page_f = open(page_path)
    page_content = page_f.read()
    page_f.close()
    soup = BeautifulSoup(page_content, "lxml")
    files = [it for it in soup.find_all(attrs={"class": "file"})]
    f_urls = [
        tg.find_all(lambda x: x.has_attr("href"))[0]["href"]
        for tg in files
        if len(tg.find_all(lambda x: x.has_attr("href"))) > 0
    ]
    date_to_url = {}
    for url_st in f_urls:
        ls = re.findall(r"20[0-9]{2}-[0-9]{2}", url_st)
        if len(ls) > 0:
            yr, mt = ls[0].split("-")
            date_to_url[(int(yr), int(mt))] = base_url + mode + url_st[1:]
    return date_to_url


# select valid top-level comments
def _valid_line(dct, mode):
    top_level = (mode == "submissions") or (
        len(dct["body"].split()) > 2
        and not dct["body"].startswith("Your submission has been removed")
        and dct["author"] != "AutoModerator"
        and dct["parent_id"] == dct["link_id"]
    )
    res = dct.get("num_comments", 1) > 0 and dct.get("score", 0) and dct.get("score", 0) >= 2 and top_level
    return res


def _open_compressed_file(f_name, f_type):
    fh = None
    if f_type == "xz":
        f = lzma.open(f_name, "rt")
    elif f_type == "bz2":
        f = bz2.open(f_name, "rt")
    elif f_type == "zst":
        fh = open(f_name, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        f = io.TextIOWrapper(stream_reader, encoding="utf-8")
    else:
        raise NotImplementedError
    return f, fh


# download a file, extract posts from desired subreddit, then remove from disk
def _download_and_select_lines(dl_manager, f_url, mode, st_time):
    # download and pre-process original posts
    print("downloading {} {:.2f}".format(f_url, time() - st_time))
    f_downloaded_path = dl_manager.download(f_url)
    print("decompressing and filtering {} {:.2f}".format(f_url, time() - st_time))
    f, fh = _open_compressed_file(f_downloaded_path, f_url.split(".")[-1])
    lines = dict([(name, []) for name in _SUB_REDDITS])
    for line in f:
        line_dct = json.loads(line)
        if any([line_dct.get("subreddit", "") == name for name in _SUB_REDDITS]):
            lines[line_dct["subreddit"]] += [line_dct]
    f.close()
    if f_url.split(".")[-1] == "zst":
        fh.close()
    os.remove(f_downloaded_path)
    os.remove(f_downloaded_path + '.json')
    os.remove(f_downloaded_path + '.lock')
    print("tokenizing and selecting {} {:.2f}".format(f_url, time() - st_time))
    processed_items = dict([(name, []) for name in _SUB_REDDITS])
    if mode == "submissions":
        key_list = ["id", "score", "url", "title", "selftext", "subreddit"]
    else:
        key_list = ["id", "link_id", "parent_id", "score", "body"]
    for name in _SUB_REDDITS:
        for line in lines[name]:
            if _valid_line(line, mode):
                reddit_res = {}
                for k in key_list:
                    if k in ["title", "selftext", "body"]:
                        reddit_res[k] = _extract_urls_from_text(line[k])
                    else:
                        reddit_res[k] = line[k]
                processed_items[name] += [reddit_res]
    print("Total found {} {} {:.2f}".format(sum([len(ls) for ls in processed_items.values()]), mode, time() - st_time))
    return processed_items


# post-process ELI5 questions and de-duplicate answers
def _post_process(reddit_dct, name=""):
    # remove the ELI5 at the start of explainlikeimfive questions
    start_re = re.compile(r"""\A[\[|\(]?[ ]?eli[5f][ ]?[\]|\)]?[]?[:,]?""", re.IGNORECASE)
    if name == "explainlikeimfive":
        title, uls = reddit_dct["title"]
        title = start_re.sub("", title.strip()).strip()
        reddit_dct["title"] = [title, uls]
    # dedupe and filter comments
    comments = [
        c
        for i, c in enumerate(reddit_dct["comments"])
        if len(c["body"][0].split()) >= 8 and c["id"] not in [x["id"] for x in reddit_dct["comments"][:i]]
    ]
    comments = sorted(comments, key=lambda c: (c["score"], len(c["body"][0].split()), c["id"]), reverse=True)
    reddit_dct["comments"] = comments
    return reddit_dct


def _download_and_filter_reddit(dl_manager, start_year=2011, start_month=7, end_year=2019, end_month=7):
    # collect submissions and comments monthly URLs
    date_to_url_submissions = _gather_dump_urls(_REDDIT_URL, "submissions", dl_manager)
    date_to_url_comments = _gather_dump_urls(_REDDIT_URL, "comments", dl_manager)
    # download, filter, process, remove
    st_time = time()
    qa_dict = dict([(name, {}) for name in _SUB_REDDITS])
    # first download all questions
    for year in range(start_year, end_year + 1):
        start_mth = start_month if year == start_year else 1
        end_mth = end_month if year == end_year else 12
        months = range(start_mth, end_mth + 1)
        for month in months:
            if (year, month) in date_to_url_submissions:
                f_url = date_to_url_submissions[(year, month)]
                processed_submissions = _download_and_select_lines(dl_manager, f_url, "submissions", st_time)
                for name in _SUB_REDDITS:
                    for dct in processed_submissions[name]:
                        qa_dict[name][dct["id"]] = dct
            else:
                print("Could not find submissions dump file for year {:4d} month {:2d}".format(year, month))
    # then all answers
    for year in range(start_year, end_year + 1):
        start_mth = start_month if year == start_year else 1
        end_mth = end_month if year == end_year else 12
        months = range(start_mth, end_mth + 1)
        for month in months:
            if (year, month) in date_to_url_comments:
                f_url = date_to_url_comments[(year, month)]
                processed_comments = _download_and_select_lines(dl_manager, f_url, "comments", st_time)
                # merge submissions and comments
                for name in _SUB_REDDITS:
                    merged_comments = 0
                    for dct in processed_comments[name]:
                        did = dct["parent_id"].split("_")[-1]
                        if did in qa_dict[name]:
                            merged_comments += 1
                            qa_dict[name][did]["comments"] = qa_dict[name][did].get("comments", []) + [dct]
            else:
                print("Could not find comments dump file for year {:4d} month {:2d}".format(year, month))
    # then post-process
    res = {}
    for name in _SUB_REDDITS:
        qa_dct_list = [(k, _post_process(rdct, name)) for k, rdct in qa_dict[name].items() if "comments" in rdct]
        qa_dct_list = [x for x in qa_dct_list if len(x[1]["comments"]) > 0 and name in x[1]["url"]]
        res[name] = dict(qa_dct_list[:])
    return res


_DESCRIPTION = """\
Explain Like I'm 5 long form QA dataset
"""

_CITATION = """\
@inproceedings{DBLP:conf/acl/FanJPGWA19,
  author    = {Angela Fan and
               Yacine Jernite and
               Ethan Perez and
               David Grangier and
               Jason Weston and
               Michael Auli},
  editor    = {Anna Korhonen and
               David R. Traum and
               Lluis Marquez},
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

class ExplainLikeImFiveConfig(nlp.BuilderConfig):
    """BuilderConfig for ExplainLikeImFive."""

    def __init__(self, **kwargs):
        """BuilderConfig for ExplainLikeImFive.
    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(ExplainLikeImFiveConfig, self).__init__(**kwargs)


class ExplainLikeImFive(nlp.GeneratorBasedBuilder):
    """ELI5: Explain Like I'm Five long form question answering dataset."""

    _DATA_SPLIT_URL = (
        "https://s3.amazonaws.com/datasets.huggingface.co/nlp/datasets/explainlikeimfive/reddit_data_split.json"
    )

    name = "ELI5_LFQA"
    BUILDER_CONFIGS = [
        ExplainLikeImFiveConfig(name="LFQA_reddit", version=nlp.Version("1.0.0"), description="long from QA subreddits"),
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "q_id": nlp.Value("string"),
                    "title": nlp.Value("string"),
                    "selftext": nlp.Value("string"),
                    "document": nlp.Value("string"),
                    "subreddit": nlp.Value("string"),
                    "answers": nlp.features.Sequence(
                        {"a_id": nlp.Value("string"), "text": nlp.Value("string"), "score": nlp.Value("int32")}
                    ),
                    "title_urls": nlp.features.Sequence({"url": nlp.Value("string")}),
                    "selftext_urls": nlp.features.Sequence({"url": nlp.Value("string")}),
                    "answers_urls": nlp.features.Sequence({"url": nlp.Value("string")}),
                }
            ),
            supervised_keys=None,
            homepage="https://facebookresearch.github.io/ELI5/explore.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        qa_data_file = pjoin(
            self._cache_dir_root, self._relative_data_dir(with_version=False), "reddit_downloaded_qa_lists.json"
        )
        if isfile(qa_data_file):
            logging.info("loading pre-computed QA list")
            self.filtered_reddit = json.load(open(qa_data_file))
        else:
            self.filtered_reddit = _download_and_filter_reddit(
                dl_manager, start_year=2011, start_month=7, end_year=2019, end_month=7
            )
            logging.info("saving pre-computed QA list")
            json.dump(self.filtered_reddit, open(qa_data_file, "w"))
        # download data splits from AWS
        fpath_splits = dl_manager.download(self._DATA_SPLIT_URL)
        self.data_split = json.load(open(fpath_splits))
        return [
            nlp.SplitGenerator(
                name=nlp.Split('train_eli5'),
                gen_kwargs={"split": "train", "subreddit_name": "explainlikeimfive"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('validation_eli5'),
                gen_kwargs={"split": "validation", "subreddit_name": "explainlikeimfive"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('test_eli5'),
                gen_kwargs={"split": "test", "subreddit_name": "explainlikeimfive"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('train_asks'),
                gen_kwargs={"split": "train", "subreddit_name": "askscience"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('validation_asks'),
                gen_kwargs={"split": "validation", "subreddit_name": "askscience"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('test_asks'),
                gen_kwargs={"split": "test", "subreddit_name": "askscience"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('train_askh'),
                gen_kwargs={"split": "train", "subreddit_name": "AskHistorians"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('validation_askh'),
                gen_kwargs={"split": "validation", "subreddit_name": "AskHistorians"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split('test_askh'),
                gen_kwargs={"split": "test", "subreddit_name": "AskHistorians"},
            ),
        ]

    def _generate_examples(self, split, subreddit_name):
        logging.info("generating examples from = {}, {} set".format(subreddit_name, split))
        if split in self.data_split.get(subreddit_name, []):
            id_list = self.data_split[subreddit_name][split]
            data = [self.filtered_reddit[subreddit_name][q_id] for q_id in id_list if q_id in self.filtered_reddit[subreddit_name]]
        elif split == "train":
            data = [
                self.filtered_reddit[subreddit_name][q_id]
                for subreddit_name in self.filtered_reddit
                for q_id in self.filtered_reddit[subreddit_name]
            ]
        else:
            data = []
        for example in data:
            id_ = example["id"]
            title = example["title"][0]
            title_urls = example["title"][1]
            selftext = example["selftext"][0]
            selftext_urls = example["selftext"][1]
            answer_scores = [ans["score"] for ans in example["comments"]]
            answer_ids = [ans["id"] for ans in example["comments"]]
            # flatten list of URL mappings
            url_maps = [(ul, i, j) for i, ans in enumerate(example["comments"]) for j, ul in enumerate(ans["body"][1])]
            answers_urls = [ul for ul, _, _ in url_maps]
            map_url_indices = dict([((i, j), k) for k, (_, i, j) in enumerate(url_maps)])
            answer_texts = []
            for i, ans in enumerate(example["comments"]):
                txt = ans["body"][0]
                for j, _ in enumerate(ans["body"][1]):
                    txt = txt.replace("_URL_{}_".format(j), "_URL_{}_".format(map_url_indices[(i, j)]))
                answer_texts += [txt.strip()]
            yield id_, {
                "q_id": id_,
                "title": title,
                "selftext": selftext,
                "document": "",
                "subreddit": example.get("subreddit", subreddit_name),
                "answers": {"a_id": answer_ids, "text": answer_texts, "score": answer_scores},
                "title_urls": {"url": title_urls},
                "selftext_urls": {"url": selftext_urls},
                "answers_urls": {"url": answers_urls},
            }
