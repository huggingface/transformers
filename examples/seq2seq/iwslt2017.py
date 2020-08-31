# coding=utf-8
# Copyright 2020 The HuggingFace NLP Authors and the current dataset script contributor.
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
"""IWSLT 2017 dataset """

from __future__ import absolute_import, division, print_function

import os

import nlp


_CITATION = """\
@inproceedings{cettoloEtAl:EAMT2012,
Address = {Trento, Italy},
Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation (EAMT)},
Date = {28-30},
Month = {May},
Pages = {261--268},
Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
Year = {2012}}
"""

_DESCRIPTION = """\
The IWSLT 2017 Evaluation Campaign includes a multilingual TED Talks MT task. The languages involved are five:

  German, English, Italian, Dutch, Romanian.

For each language pair, training and development sets are available through the entry of the table below: by clicking, an archive will be downloaded which contains the sets and a README file. Numbers in the table refer to millions of units (untokenized words) of the target side of all parallel training sets.
"""

MULTI_URL = "https://wit3.fbk.eu/archive/2017-01-trnmted//texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz"


class IWSLT2017Config(nlp.BuilderConfig):
    """ BuilderConfig for NewDataset"""

    def __init__(self, pair, is_multilingual, **kwargs):
        """

        Args:
            pair: the language pair to consider
            **kwargs: keyword arguments forwarded to super.
        """
        self.pair = pair
        self.is_multilingual = is_multilingual
        super(IWSLT2017Config, self).__init__(**kwargs)


# XXX: Artificially removed DE from here, as it also exists within bilingual data
MULTI_LANGUAGES = ["en", "it", "nl", "ro"]
BI_LANGUAGES = ["ar", "de", "en", "fr", "ja", "ko", "zh"]
MULTI_PAIRS = [f"{source}-{target}" for source in MULTI_LANGUAGES for target in MULTI_LANGUAGES if source != target]
BI_PAIRS = [
    f"{source}-{target}"
    for source in BI_LANGUAGES
    for target in BI_LANGUAGES
    if source != target and (source == "en" or target == "en")
]

PAIRS = MULTI_PAIRS + BI_PAIRS


class IWSLT217(nlp.GeneratorBasedBuilder):
    """The IWSLT 2017 Evaluation Campaign includes a multilingual TED Talks MT task."""

    VERSION = nlp.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = IWSLT2017Config
    BUILDER_CONFIGS = [
        IWSLT2017Config(
            name=pair,
            description="A small dataset",
            pair=pair,
            is_multilingual=pair in MULTI_PAIRS,
        )
        for pair in PAIRS
    ]

    def _info(self):
        return nlp.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # nlp.features.FeatureConnectors
            features=nlp.Features(
                {
                    "source": nlp.Value("string"),
                    "target": nlp.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://sites.google.com/site/iwsltevaluation2017/TED-tasks",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        source, target = self.config.pair.split("-")
        if self.config.is_multilingual:
            dl_dir = dl_manager.download_and_extract(MULTI_URL)
            data_dir = os.path.join(dl_dir, "DeEnItNlRo-DeEnItNlRo")
            years = [2010]
        else:
            bi_url = f"https://wit3.fbk.eu/archive/2017-01-trnted//texts/{source}/{target}/{source}-{target}.tgz"
            dl_dir = dl_manager.download_and_extract(bi_url)
            data_dir = os.path.join(dl_dir, f"{source}-{target}")
            years = [2010, 2011, 2012, 2013, 2014, 2015]
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            "train.tags.{}.{}".format(self.config.pair, source),
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            "train.tags.{}.{}".format(self.config.pair, target),
                        )
                    ],
                    "split": "train",
                },
            ),
            nlp.SplitGenerator(
                name=nlp.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            "IWSLT17.TED.tst{}.{}.{}.xml".format(year, self.config.pair, source),
                        )
                        for year in years
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            "IWSLT17.TED.tst{}.{}.{}.xml".format(year, self.config.pair, target),
                        )
                        for year in years
                    ],
                    "split": "test",
                },
            ),
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            "IWSLT17.TED.dev2010.{}.{}.xml".format(self.config.pair, source),
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            "IWSLT17.TED.dev2010.{}.{}.xml".format(self.config.pair, target),
                        )
                    ],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, source_files, target_files, split):
        """ Yields examples. """
        id_ = 0
        for source_file, target_file in zip(source_files, target_files):
            with open(source_file) as sf:
                with open(target_file) as tf:
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()
                            else:
                                continue

                        yield id_, {
                            "source": source_row,
                            "target": target_row,
                        }
                        id_ += 1
