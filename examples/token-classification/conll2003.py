# coding=utf-8
# Copyright 2020 HuggingFace NLP Authors.
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
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import logging

import nlp


_CITATION = """\
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
"""

_DESCRIPTION = """\
See https://www.aclweb.org/anthology/W03-0419/ for more details
"""

_URL = "https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"
_TEST_FILE = "test.txt"


class Conll2003Config(nlp.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Conll2003Config, self).__init__(**kwargs)


class Conll2003(nlp.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        Conll2003Config(name="conll2003", version=nlp.Version("1.0.0"), description="Conll2003 dataset"),
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "id": nlp.Value("string"),
                    "word": nlp.Sequence(nlp.Value("string")),
                    "pos": nlp.Sequence(nlp.Value("string")),
                    "chunk": nlp.Sequence(nlp.Value("string")),
                    "ner": nlp.Sequence(nlp.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            words = []
            pos = []
            chunk = []
            ner = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        yield guid, {"id": str(guid), "word": words, "pos": pos, "chunk": chunk, "ner": ner}
                        guid += 1
                        words = []
                        pos = []
                        chunk = []
                        ner = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    words.append(splits[0])
                    pos.append(splits[1])
                    chunk.append(splits[2])
                    ner.append(splits[3].rstrip())
            # last example
            yield guid, {"id": str(guid), "word": words, "pos": pos, "chunk": chunk, "ner": ner}
