import json
import logging
import math
import nlp

_CITATION = """\
@ONLINE {wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}
"""

_DESCRIPTION = """\
Wikipedia version split into plain text snippets for dense semantic indexing.
"""

_LICENSE = (
    "This work is licensed under the Creative Commons Attribution-ShareAlike "
    "3.0 Unported License. To view a copy of this license, visit "
    "http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to "
    "Creative Commons, PO Box 1866, Mountain View, CA 94042, USA."
)


def kilt_article_snippets(article, passage_len=100, overlap=0):
    paragraphs = article['paragraphs']['paragraph']
    section_indices = [i for i, par in enumerate(paragraphs) if par.startswith('Section::::')]
    word_map = []
    for i, par in enumerate(paragraphs):
        par_tab = par.split(' ')
        if par.startswith('Section::::'):
            word_map += [(i, len(' '.join(par[:j])), w.replace('Section::::', '').replace(':', ' '))
                         for j, w in enumerate(par_tab)]
        else:
            word_map += [(i, len(' '.join(par[:j])), w.replace('BULLET::::', ''))
                         for j, w in enumerate(par_tab)]
    step_size = passage_len - overlap
    passages = []
    for i in range(math.ceil(len(word_map) / step_size)):
        pre_toks = word_map[i * step_size: i * step_size + passage_len]
        passage_text = ' '.join([w for p_id, s_id, w in pre_toks])
        start_section_id = max([0] + [j for j in section_indices if j <= pre_toks[0][0]])
        section_ids = [j for j in section_indices if j >= start_section_id and j <= pre_toks[-1][0]]
        section_ids = section_ids if len(section_ids) > 0 else [0]
        passages += [{
            'article_title': article['title'],
            'section_title': ' & '.join([paragraphs[j].replace('Section::::', '').replace(':', ' -- ').strip() for j in section_ids]),
            'wiki_id': article['kilt_id'],
            'start_paragraph': pre_toks[0][0],
            'start_character': pre_toks[0][1],
            'end_paragraph': pre_toks[-1][0],
            'end_character': pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
            'passage_text': passage_text,
                     }]
    return passages


def wiki40b_article_snippets(article, passage_len=100, overlap=0):
    paragraphs = article['text'].split('\n')
    aticle_idx = paragraphs.index('_START_ARTICLE_') + 1
    article_title = paragraphs[aticle_idx] if aticle_idx < len(paragraphs) else ''
    section_indices = [i+1 for i, par in enumerate(paragraphs[:-1]) if par == '_START_SECTION_']
    par_tabs = [par.split(' ') for par in paragraphs]
    word_map = [(i, len(' '.join(par[:j])), w)
                for i, par in enumerate(par_tabs) if not par[0].startswith('_START_')
                for j, w in enumerate(par) if i > 0]
    step_size = passage_len - overlap
    passages = []
    for i in range(math.ceil(len(word_map) / step_size)):
        pre_toks = word_map[i * step_size: i * step_size + passage_len]
        start_section_id = max([0] + [j for j in section_indices if j <= pre_toks[0][0]])
        section_ids = [j for j in section_indices if j >= start_section_id and j <= pre_toks[-1][0]]
        section_ids = section_ids if len(section_ids) > 0 else [0]
        passage_text = ' '.join([w for p_id, s_id, w in pre_toks])
        passages += [{
            'article_title': article_title,
            'section_title': ' & '.join([paragraphs[j] for j in section_ids]),
            'wiki_id': article['wikidata_id'],
            'start_paragraph': pre_toks[0][0],
            'start_character': pre_toks[0][1],
            'end_paragraph': pre_toks[-1][0],
            'end_character': pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
            'passage_text': passage_text.replace('_NEWLINE_', '\n'),
                     }]
    return passages


def wikipedia_article_snippets(article, passage_len=100, overlap=0):
    paragraphs = [par for par in article['text'].split('\n') if not par.startswith('Category:')]
    if 'References' in paragraphs:
        paragraphs = paragraphs[:paragraphs.index('References')]
    article_title = article['title']
    section_indices = [i+1 for i, par in enumerate(paragraphs[:-2]) if paragraphs[i] == '' and paragraphs[i+1] != '' and paragraphs[i+2] != '']
    par_tabs = [par.split(' ') for par in paragraphs]
    word_map = [(i, len(' '.join(par[:j])), w)
                for i, par in enumerate(par_tabs)
                for j, w in enumerate(par)]
    step_size = passage_len - overlap
    passages = []
    for i in range(math.ceil(len(word_map) / step_size)):
        pre_toks = word_map[i * step_size: i * step_size + passage_len]
        start_section_id = max([0] + [j for j in section_indices if j <= pre_toks[0][0]])
        section_ids = [j for j in section_indices if j >= start_section_id and j <= pre_toks[-1][0]]
        section_ids = section_ids if len(section_ids) > 0 else [-1]
        passage_text = ' '.join([w for p_id, s_id, w in pre_toks])
        passages += [{
            'article_title': article_title,
            'section_title': ' & '.join(['Start' if j == -1 else paragraphs[j].strip() for j in section_ids]),
            'wiki_id': article_title.replace(' ', '_'),
            'start_paragraph': pre_toks[0][0],
            'start_character': pre_toks[0][1],
            'end_paragraph': pre_toks[-1][0],
            'end_character': pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
            'passage_text': passage_text,
                     }]
    return passages

_WIKI_KILT_DSET_LOCATION = "/home/yacine/Code/nlp/datasets/wiki_kilt"

_SPLIT_FUCNTION_MAP = {
    "wikipedia": wikipedia_article_snippets,
    "wiki40b": wiki40b_article_snippets,
    _WIKI_KILT_DSET_LOCATION: kilt_article_snippets,
}


def generate_snippets(wikipedia, split_funtion, passage_len=100, overlap=0):
    for i, article in enumerate(wikipedia):
        for doc in split_funtion(article, passage_len, overlap):
            part_id = json.dumps(
                {
                    'nlp_id': i,
                    'wiki_id': doc['wiki_id'],
                    'sp': doc['start_paragraph'],
                    'sc': doc['start_character'],
                    'ep': doc['end_paragraph'],
                    'ec': doc['end_character'],
                }
            )
            doc['_id'] = part_id
            doc['nlp_id'] = i
            yield doc


class WikiSnippetsConfig(nlp.BuilderConfig):
    """BuilderConfig for WikiSnippets."""

    def __init__(self,
                 wikipedia_name='wiki40b', wikipedia_version_name='en',
                 snippets_length=100, snippets_overlap=0,
                 **kwargs):
        """BuilderConfig for WikiSnippets.
    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(WikiSnippetsConfig, self).__init__(**kwargs)
        self.wikipedia_name = wikipedia_name
        self.wikipedia_version_name = wikipedia_version_name
        self.snippets_length = snippets_length
        self.snippets_overlap = snippets_overlap


class WikiSnippets(nlp.GeneratorBasedBuilder):
    name = "wiki_snippets"
    BUILDER_CONFIGS = [
        WikiSnippetsConfig(
            name="wiki40b_en_100_0", version=nlp.Version("1.0.0"),
            wikipedia_name='wiki40b', wikipedia_version_name='en',
            snippets_length=100, snippets_overlap=0,
        ),
        WikiSnippetsConfig(
            name="wikipedia_en_100_0", version=nlp.Version("1.0.0"),
            wikipedia_name='wikipedia', wikipedia_version_name='20200501.en',
            snippets_length=100, snippets_overlap=0,
        ),
        WikiSnippetsConfig(
            name="wiki_kilt_en_100_0", version=nlp.Version("1.0.0"),
            wikipedia_name=_WIKI_KILT_DSET_LOCATION, wikipedia_version_name='wikipedia_kilt',
            snippets_length=100, snippets_overlap=0,
        ),
    ]

    def _info(self):
        return nlp.DatasetInfo(
                description=_DESCRIPTION,
                features=nlp.Features({
                        "_id": nlp.Value("string"),
                        "nlp_id": nlp.Value("int32"),
                        "wiki_id": nlp.Value("string"),
                        "start_paragraph": nlp.Value("int32"),
                        "start_character": nlp.Value("int32"),
                        "end_paragraph": nlp.Value("int32"),
                        "end_character": nlp.Value("int32"),
                        "article_title": nlp.Value("string"),
                        "section_title": nlp.Value("string"),
                        "passage_text": nlp.Value("string"),
                }),
                supervised_keys=None,
                homepage="https://dumps.wikimedia.org",
                citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        wikipedia = nlp.load_dataset(
            path=self.config.wikipedia_name,
            name=self.config.wikipedia_version_name,
        )
        return [
                nlp.SplitGenerator(
                        name=nlp.Split.TRAIN,
                        gen_kwargs={"wikipedia": wikipedia}),
        ]

    def _generate_examples(self, wikipedia):
        logging.info("generating examples from = {} {}".format(self.config.wikipedia_name, self.config.wikipedia_version_name))
        for split in wikipedia:
            dset = wikipedia[split]
            split_function = _SPLIT_FUCNTION_MAP[self.config.wikipedia_name]
            for doc in generate_snippets(
                dset, split_function, 
                passage_len=self.config.snippets_length,
                overlap=self.config.snippets_overlap
            ):
                id_ = doc['_id']
                yield id_, doc

