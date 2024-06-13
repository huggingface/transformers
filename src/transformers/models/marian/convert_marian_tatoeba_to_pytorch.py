# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple

import yaml
from tqdm import tqdm

from transformers.models.marian.convert_marian_to_pytorch import (
    FRONT_MATTER_TEMPLATE,
    convert,
    convert_opus_name_to_hf_name,
    download_and_unzip,
    get_system_metadata,
)


DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")
ISO_URL = "https://cdn-datasets.huggingface.co/language_codes/iso-639-3.csv"
ISO_PATH = "lang_code_data/iso-639-3.csv"
LANG_CODE_PATH = "lang_code_data/language-codes-3b2.csv"
TATOEBA_MODELS_URL = "https://object.pouta.csc.fi/Tatoeba-MT-models"


class TatoebaConverter:
    """
    Convert Tatoeba-Challenge models to huggingface format.

    Steps:

        1. Convert numpy state dict to hf format (same code as OPUS-MT-Train conversion).
        2. Rename opus model to huggingface format. This means replace each alpha3 code with an alpha2 code if a unique
           one exists. e.g. aav-eng -> aav-en, heb-eng -> he-en
        3. Select the best model for a particular pair, parse the yml for it and write a model card. By default the
           best model is the one listed first in released-model-results, but it's also possible to specify the most
           recent one.
    """

    def __init__(self, save_dir="marian_converted"):
        assert Path(DEFAULT_REPO).exists(), "need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git"
        self.download_lang_info()
        self.model_results = json.load(open("Tatoeba-Challenge/models/released-model-results.json"))
        self.alpha3_to_alpha2 = {}
        for line in open(ISO_PATH):
            parts = line.split("\t")
            if len(parts[0]) == 3 and len(parts[3]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[3]
        for line in LANG_CODE_PATH:
            parts = line.split(",")
            if len(parts[0]) == 3 and len(parts[1]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[1]
        self.model_card_dir = Path(save_dir)
        self.tag2name = {}
        for key, value in GROUP_MEMBERS.items():
            self.tag2name[key] = value[0]

    def convert_models(self, tatoeba_ids, dry_run=False):
        models_to_convert = [self.parse_metadata(x) for x in tatoeba_ids]
        save_dir = Path("marian_ckpt")
        dest_dir = Path(self.model_card_dir)
        dest_dir.mkdir(exist_ok=True)
        for model in tqdm(models_to_convert):  # k, prepro, download, test_set_url in tqdm(model_list):
            if "SentencePiece" not in model["pre-processing"]:
                print(f"Skipping {model['release']} because it doesn't appear to use SentencePiece")
                continue
            if not os.path.exists(save_dir / model["_name"]):
                download_and_unzip(f"{TATOEBA_MODELS_URL}/{model['release']}", save_dir / model["_name"])
            # from convert_marian_to_pytorch
            opus_language_groups_to_hf = convert_opus_name_to_hf_name
            pair_name = opus_language_groups_to_hf(model["_name"])
            convert(save_dir / model["_name"], dest_dir / f"opus-mt-{pair_name}")
            self.write_model_card(model, dry_run=dry_run)

    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.alpha3_to_alpha2.get(x, x) for x in GROUP_MEMBERS[grp_name][1]]

    def is_group(self, code, name):
        return "languages" in name or len(GROUP_MEMBERS.get(code, [])) > 1

    def get_tags(self, code, name):
        if len(code) == 2:
            assert "languages" not in name, f"{code}: {name}"
            return [code]
        elif self.is_group(code, name):
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group
        else:  # zho-> zh
            print(f"Three letter monolingual code: {code}")
            return [code]

    def resolve_lang_code(self, src, tgt) -> Tuple[str, str]:
        src_tags = self.get_tags(src, self.tag2name[src])
        tgt_tags = self.get_tags(tgt, self.tag2name[tgt])
        return src_tags, tgt_tags

    @staticmethod
    def model_type_info_from_model_name(name):
        info = {"_has_backtranslated_data": False}
        if "1m" in name:
            info["_data_per_pair"] = str(1e6)
        if "2m" in name:
            info["_data_per_pair"] = str(2e6)
        if "4m" in name:
            info["_data_per_pair"] = str(4e6)
        if "+bt" in name:
            info["_has_backtranslated_data"] = True
        if "tuned4" in name:
            info["_tuned"] = re.search(r"tuned4[^-]+", name).group()
        return info

    def write_model_card(self, model_dict, dry_run=False) -> str:
        """
        Construct card from data parsed from YAML and the model's name. upload command: aws s3 sync model_card_dir
        s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        """
        model_dir_url = f"{TATOEBA_MODELS_URL}/{model_dict['release']}"
        long_pair = model_dict["_name"].split("-")
        assert len(long_pair) == 2, f"got a translation pair {model_dict['_name']} that doesn't appear to be a pair"
        short_src = self.alpha3_to_alpha2.get(long_pair[0], long_pair[0])
        short_tgt = self.alpha3_to_alpha2.get(long_pair[1], long_pair[1])
        model_dict["_hf_model_id"] = f"opus-mt-{short_src}-{short_tgt}"

        a3_src, a3_tgt = model_dict["_name"].split("-")
        # opus_src_tags, opus_tgt_tags = a3_src.split("+"), a3_tgt.split("+")

        # This messy part tries to deal with language tags in multilingual models, possibly
        # not all having three-letter codes
        resolved_src_tags, resolved_tgt_tags = self.resolve_lang_code(a3_src, a3_tgt)
        a2_src_tags, a2_tgt_tags = [], []
        for tag in resolved_src_tags:
            if tag not in self.alpha3_to_alpha2:
                a2_src_tags.append(tag)
        for tag in resolved_tgt_tags:
            if tag not in self.alpha3_to_alpha2:
                a2_tgt_tags.append(tag)

        lang_tags = dedup(a2_src_tags + a2_tgt_tags)
        src_multilingual, tgt_multilingual = (len(a2_src_tags) > 1), (len(a2_tgt_tags) > 1)
        s, t = ",".join(a2_src_tags), ",".join(a2_tgt_tags)

        metadata = {
            "hf_name": model_dict["_name"],
            "source_languages": s,
            "target_languages": t,
            "opus_readme_url": f"{model_dir_url}/README.md",
            "original_repo": "Tatoeba-Challenge",
            "tags": ["translation"],
            "languages": lang_tags,
        }
        lang_tags = l2front_matter(lang_tags)

        metadata["src_constituents"] = list(GROUP_MEMBERS[a3_src][1])
        metadata["tgt_constituents"] = list(GROUP_MEMBERS[a3_tgt][1])
        metadata["src_multilingual"] = src_multilingual
        metadata["tgt_multilingual"] = tgt_multilingual

        backtranslated_data = ""
        if model_dict["_has_backtranslated_data"]:
            backtranslated_data = " with backtranslations"

        multilingual_data = ""
        if "_data_per_pair" in model_dict:
            multilingual_data = f"* data per pair in multilingual model: {model_dict['_data_per_pair']}\n"

        tuned = ""
        if "_tuned" in model_dict:
            tuned = f"* multilingual model tuned for: {model_dict['_tuned']}\n"

        model_base_filename = model_dict["release"].split("/")[-1]
        download = f"* download original weights: [{model_base_filename}]({model_dir_url}/{model_dict['release']})\n"

        langtoken = ""
        if tgt_multilingual:
            langtoken = (
                "* a sentence-initial language token is required in the form of >>id<<"
                "(id = valid, usually three-letter target language ID)\n"
            )

        metadata.update(get_system_metadata(DEFAULT_REPO))

        scorestable = ""
        for k, v in model_dict.items():
            if "scores" in k:
                this_score_table = f"* {k}\n|Test set|score|\n|---|---|\n"
                pairs = sorted(v.items(), key=lambda x: x[1], reverse=True)
                for pair in pairs:
                    this_score_table += f"|{pair[0]}|{pair[1]}|\n"
                scorestable += this_score_table

        datainfo = ""
        if "training-data" in model_dict:
            datainfo += "* Training data: \n"
            for k, v in model_dict["training-data"].items():
                datainfo += f"  * {str(k)}: {str(v)}\n"
        if "validation-data" in model_dict:
            datainfo += "* Validation data: \n"
            for k, v in model_dict["validation-data"].items():
                datainfo += f"  * {str(k)}: {str(v)}\n"
        if "test-data" in model_dict:
            datainfo += "* Test data: \n"
            for k, v in model_dict["test-data"].items():
                datainfo += f"  * {str(k)}: {str(v)}\n"

        testsetfilename = model_dict["release"].replace(".zip", ".test.txt")
        testscoresfilename = model_dict["release"].replace(".zip", ".eval.txt")
        testset = f"* test set translations file: [test.txt]({model_dir_url}/{testsetfilename})\n"
        testscores = f"* test set scores file: [eval.txt]({model_dir_url}/{testscoresfilename})\n"

        # combine with Tatoeba markdown
        readme_url = f"{TATOEBA_MODELS_URL}/{model_dict['_name']}/README.md"
        extra_markdown = f"""
### {model_dict['_name']}

* source language name: {self.tag2name[a3_src]}
* target language name: {self.tag2name[a3_tgt]}
* OPUS readme: [README.md]({readme_url})
"""

        content = (
            f"""
* model: {model_dict['modeltype']}
* source language code{src_multilingual*'s'}: {', '.join(a2_src_tags)}
* target language code{tgt_multilingual*'s'}: {', '.join(a2_tgt_tags)}
* dataset: opus {backtranslated_data}
* release date: {model_dict['release-date']}
* pre-processing: {model_dict['pre-processing']}
"""
            + multilingual_data
            + tuned
            + download
            + langtoken
            + datainfo
            + testset
            + testscores
            + scorestable
        )

        content = FRONT_MATTER_TEMPLATE.format(lang_tags) + extra_markdown + content

        items = "\n".join([f"* {k}: {v}" for k, v in metadata.items()])
        sec3 = "\n### System Info: \n" + items
        content += sec3
        if dry_run:
            print("CONTENT:")
            print(content)
            print("METADATA:")
            print(metadata)
            return
        sub_dir = self.model_card_dir / model_dict["_hf_model_id"]
        sub_dir.mkdir(exist_ok=True)
        dest = sub_dir / "README.md"
        dest.open("w").write(content)
        for k, v in metadata.items():
            if isinstance(v, datetime.date):
                metadata[k] = datetime.datetime.strftime(v, "%Y-%m-%d")
        with open(sub_dir / "metadata.json", "w", encoding="utf-8") as writeobj:
            json.dump(metadata, writeobj)

    def download_lang_info(self):
        global LANG_CODE_PATH
        Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
        import wget
        from huggingface_hub import hf_hub_download

        if not os.path.exists(ISO_PATH):
            wget.download(ISO_URL, ISO_PATH)
        if not os.path.exists(LANG_CODE_PATH):
            LANG_CODE_PATH = hf_hub_download(
                repo_id="huggingface/language_codes_marianMT", filename="language-codes-3b2.csv", repo_type="dataset"
            )

    def parse_metadata(self, model_name, repo_path=DEFAULT_MODEL_DIR, method="best"):
        p = Path(repo_path) / model_name

        def url_to_name(url):
            return url.split("/")[-1].split(".")[0]

        if model_name not in self.model_results:
            # This is not a language pair, so model results are ambiguous, go by newest
            method = "newest"

        if method == "best":
            # Sort by how early they appear in released-models-results
            results = [url_to_name(model["download"]) for model in self.model_results[model_name]]
            ymls = [f for f in os.listdir(p) if f.endswith(".yml") and f[:-4] in results]
            ymls.sort(key=lambda x: results.index(x[:-4]))
            metadata = yaml.safe_load(open(p / ymls[0]))
            metadata.update(self.model_type_info_from_model_name(ymls[0][:-4]))
        elif method == "newest":
            ymls = [f for f in os.listdir(p) if f.endswith(".yml")]
            # Sort by date
            ymls.sort(
                key=lambda x: datetime.datetime.strptime(re.search(r"\d\d\d\d-\d\d?-\d\d?", x).group(), "%Y-%m-%d")
            )
            metadata = yaml.safe_load(open(p / ymls[-1]))
            metadata.update(self.model_type_info_from_model_name(ymls[-1][:-4]))
        else:
            raise NotImplementedError(f"Don't know argument method='{method}' to parse_metadata()")
        metadata["_name"] = model_name
        return metadata


GROUP_MEMBERS = {
    # three letter code -> (group/language name, {constituents...}
    # if this language is on the target side the constituents can be used as target language codes.
    # if the language is on the source side they are supported natively without special codes.
    "aav": ("Austro-Asiatic languages", {"hoc", "hoc_Latn", "kha", "khm", "khm_Latn", "mnw", "vie", "vie_Hani"}),
    "afa": (
        "Afro-Asiatic languages",
        {
            "acm",
            "afb",
            "amh",
            "apc",
            "ara",
            "arq",
            "ary",
            "arz",
            "hau_Latn",
            "heb",
            "kab",
            "mlt",
            "rif_Latn",
            "shy_Latn",
            "som",
            "thv",
            "tir",
        },
    ),
    "afr": ("Afrikaans", {"afr"}),
    "alv": (
        "Atlantic-Congo languages",
        {
            "ewe",
            "fuc",
            "fuv",
            "ibo",
            "kin",
            "lin",
            "lug",
            "nya",
            "run",
            "sag",
            "sna",
            "swh",
            "toi_Latn",
            "tso",
            "umb",
            "wol",
            "xho",
            "yor",
            "zul",
        },
    ),
    "ara": ("Arabic", {"afb", "apc", "apc_Latn", "ara", "ara_Latn", "arq", "arq_Latn", "arz"}),
    "art": (
        "Artificial languages",
        {
            "afh_Latn",
            "avk_Latn",
            "dws_Latn",
            "epo",
            "ido",
            "ido_Latn",
            "ile_Latn",
            "ina_Latn",
            "jbo",
            "jbo_Cyrl",
            "jbo_Latn",
            "ldn_Latn",
            "lfn_Cyrl",
            "lfn_Latn",
            "nov_Latn",
            "qya",
            "qya_Latn",
            "sjn_Latn",
            "tlh_Latn",
            "tzl",
            "tzl_Latn",
            "vol_Latn",
        },
    ),
    "aze": ("Azerbaijani", {"aze_Latn"}),
    "bat": ("Baltic languages", {"lit", "lav", "prg_Latn", "ltg", "sgs"}),
    "bel": ("Belarusian", {"bel", "bel_Latn"}),
    "ben": ("Bengali", {"ben"}),
    "bnt": (
        "Bantu languages",
        {"kin", "lin", "lug", "nya", "run", "sna", "swh", "toi_Latn", "tso", "umb", "xho", "zul"},
    ),
    "bul": ("Bulgarian", {"bul", "bul_Latn"}),
    "cat": ("Catalan", {"cat"}),
    "cau": ("Caucasian languages", {"abk", "kat", "che", "ady"}),
    "ccs": ("South Caucasian languages", {"kat"}),
    "ceb": ("Cebuano", {"ceb"}),
    "cel": ("Celtic languages", {"gla", "gle", "bre", "cor", "glv", "cym"}),
    "ces": ("Czech", {"ces"}),
    "cpf": ("Creoles and pidgins, French‑based", {"gcf_Latn", "hat", "mfe"}),
    "cpp": (
        "Creoles and pidgins, Portuguese-based",
        {"zsm_Latn", "ind", "pap", "min", "tmw_Latn", "max_Latn", "zlm_Latn"},
    ),
    "cus": ("Cushitic languages", {"som"}),
    "dan": ("Danish", {"dan"}),
    "deu": ("German", {"deu"}),
    "dra": ("Dravidian languages", {"tam", "kan", "mal", "tel"}),
    "ell": ("Modern Greek (1453-)", {"ell"}),
    "eng": ("English", {"eng"}),
    "epo": ("Esperanto", {"epo"}),
    "est": ("Estonian", {"est"}),
    "euq": ("Basque (family)", {"eus"}),
    "eus": ("Basque", {"eus"}),
    "fin": ("Finnish", {"fin"}),
    "fiu": (
        "Finno-Ugrian languages",
        {
            "est",
            "fin",
            "fkv_Latn",
            "hun",
            "izh",
            "kpv",
            "krl",
            "liv_Latn",
            "mdf",
            "mhr",
            "myv",
            "sma",
            "sme",
            "udm",
            "vep",
            "vro",
        },
    ),
    "fra": ("French", {"fra"}),
    "gem": (
        "Germanic languages",
        {
            "afr",
            "ang_Latn",
            "dan",
            "deu",
            "eng",
            "enm_Latn",
            "fao",
            "frr",
            "fry",
            "gos",
            "got_Goth",
            "gsw",
            "isl",
            "ksh",
            "ltz",
            "nds",
            "nld",
            "nno",
            "nob",
            "nob_Hebr",
            "non_Latn",
            "pdc",
            "sco",
            "stq",
            "swe",
            "swg",
            "yid",
        },
    ),
    "gle": ("Irish", {"gle"}),
    "glg": ("Galician", {"glg"}),
    "gmq": ("North Germanic languages", {"dan", "nob", "nob_Hebr", "swe", "isl", "nno", "non_Latn", "fao"}),
    "gmw": (
        "West Germanic languages",
        {
            "afr",
            "ang_Latn",
            "deu",
            "eng",
            "enm_Latn",
            "frr",
            "fry",
            "gos",
            "gsw",
            "ksh",
            "ltz",
            "nds",
            "nld",
            "pdc",
            "sco",
            "stq",
            "swg",
            "yid",
        },
    ),
    "grk": ("Greek languages", {"grc_Grek", "ell"}),
    "hbs": ("Serbo-Croatian", {"hrv", "srp_Cyrl", "bos_Latn", "srp_Latn"}),
    "heb": ("Hebrew", {"heb"}),
    "hin": ("Hindi", {"hin"}),
    "hun": ("Hungarian", {"hun"}),
    "hye": ("Armenian", {"hye", "hye_Latn"}),
    "iir": (
        "Indo-Iranian languages",
        {
            "asm",
            "awa",
            "ben",
            "bho",
            "gom",
            "guj",
            "hif_Latn",
            "hin",
            "jdt_Cyrl",
            "kur_Arab",
            "kur_Latn",
            "mai",
            "mar",
            "npi",
            "ori",
            "oss",
            "pan_Guru",
            "pes",
            "pes_Latn",
            "pes_Thaa",
            "pnb",
            "pus",
            "rom",
            "san_Deva",
            "sin",
            "snd_Arab",
            "tgk_Cyrl",
            "tly_Latn",
            "urd",
            "zza",
        },
    ),
    "ilo": ("Iloko", {"ilo"}),
    "inc": (
        "Indic languages",
        {
            "asm",
            "awa",
            "ben",
            "bho",
            "gom",
            "guj",
            "hif_Latn",
            "hin",
            "mai",
            "mar",
            "npi",
            "ori",
            "pan_Guru",
            "pnb",
            "rom",
            "san_Deva",
            "sin",
            "snd_Arab",
            "urd",
        },
    ),
    "ine": (
        "Indo-European languages",
        {
            "afr",
            "afr_Arab",
            "aln",
            "ang_Latn",
            "arg",
            "asm",
            "ast",
            "awa",
            "bel",
            "bel_Latn",
            "ben",
            "bho",
            "bjn",
            "bos_Latn",
            "bre",
            "bul",
            "bul_Latn",
            "cat",
            "ces",
            "cor",
            "cos",
            "csb_Latn",
            "cym",
            "dan",
            "deu",
            "dsb",
            "egl",
            "ell",
            "eng",
            "enm_Latn",
            "ext",
            "fao",
            "fra",
            "frm_Latn",
            "frr",
            "fry",
            "gcf_Latn",
            "gla",
            "gle",
            "glg",
            "glv",
            "gom",
            "gos",
            "got_Goth",
            "grc_Grek",
            "gsw",
            "guj",
            "hat",
            "hif_Latn",
            "hin",
            "hrv",
            "hsb",
            "hye",
            "hye_Latn",
            "ind",
            "isl",
            "ita",
            "jdt_Cyrl",
            "ksh",
            "kur_Arab",
            "kur_Latn",
            "lad",
            "lad_Latn",
            "lat_Grek",
            "lat_Latn",
            "lav",
            "lij",
            "lit",
            "lld_Latn",
            "lmo",
            "ltg",
            "ltz",
            "mai",
            "mar",
            "max_Latn",
            "mfe",
            "min",
            "mkd",
            "mwl",
            "nds",
            "nld",
            "nno",
            "nob",
            "nob_Hebr",
            "non_Latn",
            "npi",
            "oci",
            "ori",
            "orv_Cyrl",
            "oss",
            "pan_Guru",
            "pap",
            "pcd",
            "pdc",
            "pes",
            "pes_Latn",
            "pes_Thaa",
            "pms",
            "pnb",
            "pol",
            "por",
            "prg_Latn",
            "pus",
            "roh",
            "rom",
            "ron",
            "rue",
            "rus",
            "rus_Latn",
            "san_Deva",
            "scn",
            "sco",
            "sgs",
            "sin",
            "slv",
            "snd_Arab",
            "spa",
            "sqi",
            "srd",
            "srp_Cyrl",
            "srp_Latn",
            "stq",
            "swe",
            "swg",
            "tgk_Cyrl",
            "tly_Latn",
            "tmw_Latn",
            "ukr",
            "urd",
            "vec",
            "wln",
            "yid",
            "zlm_Latn",
            "zsm_Latn",
            "zza",
        },
    ),
    "isl": ("Icelandic", {"isl"}),
    "ita": ("Italian", {"ita"}),
    "itc": (
        "Italic languages",
        {
            "arg",
            "ast",
            "bjn",
            "cat",
            "cos",
            "egl",
            "ext",
            "fra",
            "frm_Latn",
            "gcf_Latn",
            "glg",
            "hat",
            "ind",
            "ita",
            "lad",
            "lad_Latn",
            "lat_Grek",
            "lat_Latn",
            "lij",
            "lld_Latn",
            "lmo",
            "max_Latn",
            "mfe",
            "min",
            "mwl",
            "oci",
            "pap",
            "pcd",
            "pms",
            "por",
            "roh",
            "ron",
            "scn",
            "spa",
            "srd",
            "tmw_Latn",
            "vec",
            "wln",
            "zlm_Latn",
            "zsm_Latn",
        },
    ),
    "jpn": ("Japanese", {"jpn", "jpn_Bopo", "jpn_Hang", "jpn_Hani", "jpn_Hira", "jpn_Kana", "jpn_Latn", "jpn_Yiii"}),
    "jpx": ("Japanese (family)", {"jpn"}),
    "kat": ("Georgian", {"kat"}),
    "kor": ("Korean", {"kor_Hani", "kor_Hang", "kor_Latn", "kor"}),
    "lav": ("Latvian", {"lav"}),
    "lit": ("Lithuanian", {"lit"}),
    "mkd": ("Macedonian", {"mkd"}),
    "mkh": ("Mon-Khmer languages", {"vie_Hani", "mnw", "vie", "kha", "khm_Latn", "khm"}),
    "msa": ("Malay (macrolanguage)", {"zsm_Latn", "ind", "max_Latn", "zlm_Latn", "min"}),
    "mul": (
        "Multiple languages",
        {
            "abk",
            "acm",
            "ady",
            "afb",
            "afh_Latn",
            "afr",
            "akl_Latn",
            "aln",
            "amh",
            "ang_Latn",
            "apc",
            "ara",
            "arg",
            "arq",
            "ary",
            "arz",
            "asm",
            "ast",
            "avk_Latn",
            "awa",
            "aze_Latn",
            "bak",
            "bam_Latn",
            "bel",
            "bel_Latn",
            "ben",
            "bho",
            "bod",
            "bos_Latn",
            "bre",
            "brx",
            "brx_Latn",
            "bul",
            "bul_Latn",
            "cat",
            "ceb",
            "ces",
            "cha",
            "che",
            "chr",
            "chv",
            "cjy_Hans",
            "cjy_Hant",
            "cmn",
            "cmn_Hans",
            "cmn_Hant",
            "cor",
            "cos",
            "crh",
            "crh_Latn",
            "csb_Latn",
            "cym",
            "dan",
            "deu",
            "dsb",
            "dtp",
            "dws_Latn",
            "egl",
            "ell",
            "enm_Latn",
            "epo",
            "est",
            "eus",
            "ewe",
            "ext",
            "fao",
            "fij",
            "fin",
            "fkv_Latn",
            "fra",
            "frm_Latn",
            "frr",
            "fry",
            "fuc",
            "fuv",
            "gan",
            "gcf_Latn",
            "gil",
            "gla",
            "gle",
            "glg",
            "glv",
            "gom",
            "gos",
            "got_Goth",
            "grc_Grek",
            "grn",
            "gsw",
            "guj",
            "hat",
            "hau_Latn",
            "haw",
            "heb",
            "hif_Latn",
            "hil",
            "hin",
            "hnj_Latn",
            "hoc",
            "hoc_Latn",
            "hrv",
            "hsb",
            "hun",
            "hye",
            "iba",
            "ibo",
            "ido",
            "ido_Latn",
            "ike_Latn",
            "ile_Latn",
            "ilo",
            "ina_Latn",
            "ind",
            "isl",
            "ita",
            "izh",
            "jav",
            "jav_Java",
            "jbo",
            "jbo_Cyrl",
            "jbo_Latn",
            "jdt_Cyrl",
            "jpn",
            "kab",
            "kal",
            "kan",
            "kat",
            "kaz_Cyrl",
            "kaz_Latn",
            "kek_Latn",
            "kha",
            "khm",
            "khm_Latn",
            "kin",
            "kir_Cyrl",
            "kjh",
            "kpv",
            "krl",
            "ksh",
            "kum",
            "kur_Arab",
            "kur_Latn",
            "lad",
            "lad_Latn",
            "lao",
            "lat_Latn",
            "lav",
            "ldn_Latn",
            "lfn_Cyrl",
            "lfn_Latn",
            "lij",
            "lin",
            "lit",
            "liv_Latn",
            "lkt",
            "lld_Latn",
            "lmo",
            "ltg",
            "ltz",
            "lug",
            "lzh",
            "lzh_Hans",
            "mad",
            "mah",
            "mai",
            "mal",
            "mar",
            "max_Latn",
            "mdf",
            "mfe",
            "mhr",
            "mic",
            "min",
            "mkd",
            "mlg",
            "mlt",
            "mnw",
            "moh",
            "mon",
            "mri",
            "mwl",
            "mww",
            "mya",
            "myv",
            "nan",
            "nau",
            "nav",
            "nds",
            "niu",
            "nld",
            "nno",
            "nob",
            "nob_Hebr",
            "nog",
            "non_Latn",
            "nov_Latn",
            "npi",
            "nya",
            "oci",
            "ori",
            "orv_Cyrl",
            "oss",
            "ota_Arab",
            "ota_Latn",
            "pag",
            "pan_Guru",
            "pap",
            "pau",
            "pdc",
            "pes",
            "pes_Latn",
            "pes_Thaa",
            "pms",
            "pnb",
            "pol",
            "por",
            "ppl_Latn",
            "prg_Latn",
            "pus",
            "quc",
            "qya",
            "qya_Latn",
            "rap",
            "rif_Latn",
            "roh",
            "rom",
            "ron",
            "rue",
            "run",
            "rus",
            "sag",
            "sah",
            "san_Deva",
            "scn",
            "sco",
            "sgs",
            "shs_Latn",
            "shy_Latn",
            "sin",
            "sjn_Latn",
            "slv",
            "sma",
            "sme",
            "smo",
            "sna",
            "snd_Arab",
            "som",
            "spa",
            "sqi",
            "srp_Cyrl",
            "srp_Latn",
            "stq",
            "sun",
            "swe",
            "swg",
            "swh",
            "tah",
            "tam",
            "tat",
            "tat_Arab",
            "tat_Latn",
            "tel",
            "tet",
            "tgk_Cyrl",
            "tha",
            "tir",
            "tlh_Latn",
            "tly_Latn",
            "tmw_Latn",
            "toi_Latn",
            "ton",
            "tpw_Latn",
            "tso",
            "tuk",
            "tuk_Latn",
            "tur",
            "tvl",
            "tyv",
            "tzl",
            "tzl_Latn",
            "udm",
            "uig_Arab",
            "uig_Cyrl",
            "ukr",
            "umb",
            "urd",
            "uzb_Cyrl",
            "uzb_Latn",
            "vec",
            "vie",
            "vie_Hani",
            "vol_Latn",
            "vro",
            "war",
            "wln",
            "wol",
            "wuu",
            "xal",
            "xho",
            "yid",
            "yor",
            "yue",
            "yue_Hans",
            "yue_Hant",
            "zho",
            "zho_Hans",
            "zho_Hant",
            "zlm_Latn",
            "zsm_Latn",
            "zul",
            "zza",
        },
    ),
    "nic": (
        "Niger-Kordofanian languages",
        {
            "bam_Latn",
            "ewe",
            "fuc",
            "fuv",
            "ibo",
            "kin",
            "lin",
            "lug",
            "nya",
            "run",
            "sag",
            "sna",
            "swh",
            "toi_Latn",
            "tso",
            "umb",
            "wol",
            "xho",
            "yor",
            "zul",
        },
    ),
    "nld": ("Dutch", {"nld"}),
    "nor": ("Norwegian", {"nob", "nno"}),
    "phi": ("Philippine languages", {"ilo", "akl_Latn", "war", "hil", "pag", "ceb"}),
    "pol": ("Polish", {"pol"}),
    "por": ("Portuguese", {"por"}),
    "pqe": (
        "Eastern Malayo-Polynesian languages",
        {"fij", "gil", "haw", "mah", "mri", "nau", "niu", "rap", "smo", "tah", "ton", "tvl"},
    ),
    "roa": (
        "Romance languages",
        {
            "arg",
            "ast",
            "cat",
            "cos",
            "egl",
            "ext",
            "fra",
            "frm_Latn",
            "gcf_Latn",
            "glg",
            "hat",
            "ind",
            "ita",
            "lad",
            "lad_Latn",
            "lij",
            "lld_Latn",
            "lmo",
            "max_Latn",
            "mfe",
            "min",
            "mwl",
            "oci",
            "pap",
            "pms",
            "por",
            "roh",
            "ron",
            "scn",
            "spa",
            "tmw_Latn",
            "vec",
            "wln",
            "zlm_Latn",
            "zsm_Latn",
        },
    ),
    "ron": ("Romanian", {"ron"}),
    "run": ("Rundi", {"run"}),
    "rus": ("Russian", {"rus"}),
    "sal": ("Salishan languages", {"shs_Latn"}),
    "sem": ("Semitic languages", {"acm", "afb", "amh", "apc", "ara", "arq", "ary", "arz", "heb", "mlt", "tir"}),
    "sla": (
        "Slavic languages",
        {
            "bel",
            "bel_Latn",
            "bos_Latn",
            "bul",
            "bul_Latn",
            "ces",
            "csb_Latn",
            "dsb",
            "hrv",
            "hsb",
            "mkd",
            "orv_Cyrl",
            "pol",
            "rue",
            "rus",
            "slv",
            "srp_Cyrl",
            "srp_Latn",
            "ukr",
        },
    ),
    "slv": ("Slovenian", {"slv"}),
    "spa": ("Spanish", {"spa"}),
    "swe": ("Swedish", {"swe"}),
    "taw": ("Tai", {"lao", "tha"}),
    "tgl": ("Tagalog", {"tgl_Latn"}),
    "tha": ("Thai", {"tha"}),
    "trk": (
        "Turkic languages",
        {
            "aze_Latn",
            "bak",
            "chv",
            "crh",
            "crh_Latn",
            "kaz_Cyrl",
            "kaz_Latn",
            "kir_Cyrl",
            "kjh",
            "kum",
            "ota_Arab",
            "ota_Latn",
            "sah",
            "tat",
            "tat_Arab",
            "tat_Latn",
            "tuk",
            "tuk_Latn",
            "tur",
            "tyv",
            "uig_Arab",
            "uig_Cyrl",
            "uzb_Cyrl",
            "uzb_Latn",
        },
    ),
    "tur": ("Turkish", {"tur"}),
    "ukr": ("Ukrainian", {"ukr"}),
    "urd": ("Urdu", {"urd"}),
    "urj": (
        "Uralic languages",
        {
            "est",
            "fin",
            "fkv_Latn",
            "hun",
            "izh",
            "kpv",
            "krl",
            "liv_Latn",
            "mdf",
            "mhr",
            "myv",
            "sma",
            "sme",
            "udm",
            "vep",
            "vro",
        },
    ),
    "vie": ("Vietnamese", {"vie", "vie_Hani"}),
    "war": ("Waray (Philippines)", {"war"}),
    "zho": (
        "Chinese",
        {
            "cjy_Hans",
            "cjy_Hant",
            "cmn",
            "cmn_Bopo",
            "cmn_Hang",
            "cmn_Hani",
            "cmn_Hans",
            "cmn_Hant",
            "cmn_Hira",
            "cmn_Kana",
            "cmn_Latn",
            "cmn_Yiii",
            "gan",
            "hak_Hani",
            "lzh",
            "lzh_Bopo",
            "lzh_Hang",
            "lzh_Hani",
            "lzh_Hans",
            "lzh_Hira",
            "lzh_Kana",
            "lzh_Yiii",
            "nan",
            "nan_Hani",
            "wuu",
            "wuu_Bopo",
            "wuu_Hani",
            "wuu_Latn",
            "yue",
            "yue_Bopo",
            "yue_Hang",
            "yue_Hani",
            "yue_Hans",
            "yue_Hant",
            "yue_Hira",
            "yue_Kana",
            "zho",
            "zho_Hans",
            "zho_Hant",
        },
    ),
    "zle": ("East Slavic languages", {"bel", "orv_Cyrl", "bel_Latn", "rus", "ukr", "rue"}),
    "zls": ("South Slavic languages", {"bos_Latn", "bul", "bul_Latn", "hrv", "mkd", "slv", "srp_Cyrl", "srp_Latn"}),
    "zlw": ("West Slavic languages", {"csb_Latn", "dsb", "hsb", "pol", "ces"}),
}


def l2front_matter(langs):
    return "".join(f"- {l}\n" for l in langs)


def dedup(lst):
    """Preservers order"""
    new_lst = []
    for item in lst:
        if not item or item in new_lst:
            continue
        else:
            new_lst.append(item)
    return new_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--models", action="append", help="<Required> Set flag", required=True, nargs="+", dest="models"
    )
    parser.add_argument("-save_dir", "--save_dir", default="marian_converted", help="where to save converted models")
    args = parser.parse_args()
    resolver = TatoebaConverter(save_dir=args.save_dir)
    resolver.convert_models(args.models[0])
