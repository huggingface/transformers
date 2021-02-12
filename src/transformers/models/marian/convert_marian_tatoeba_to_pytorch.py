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
import os
from pathlib import Path
from typing import List, Tuple

from transformers.models.marian.convert_marian_to_pytorch import (
    FRONT_MATTER_TEMPLATE,
    _parse_readme,
    convert_all_sentencepiece_models,
    get_system_metadata,
    remove_prefix,
    remove_suffix,
)


try:
    import pandas as pd
except ImportError:
    pass

DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")
LANG_CODE_URL = "https://datahub.io/core/language-codes/r/language-codes-3b2.csv"
ISO_URL = "https://cdn-datasets.huggingface.co/language_codes/iso-639-3.csv"
ISO_PATH = "lang_code_data/iso-639-3.csv"
LANG_CODE_PATH = "lang_code_data/language-codes-3b2.csv"


class TatoebaConverter:
    """
    Convert Tatoeba-Challenge models to huggingface format.

    Steps:

        1. convert numpy state dict to hf format (same code as OPUS-MT-Train conversion).
        2. rename opus model to huggingface format. This means replace each alpha3 code with an alpha2 code if a unique
           one exists. e.g. aav-eng -> aav-en, heb-eng -> he-en
        3. write a model card containing the original Tatoeba-Challenge/README.md and extra info about alpha3 group
           members.
    """

    def __init__(self, save_dir="marian_converted"):
        assert Path(DEFAULT_REPO).exists(), "need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git"
        reg = self.make_tatoeba_registry()
        self.download_metadata()
        self.registry = reg
        reg_df = pd.DataFrame(reg, columns=["id", "prepro", "url_model", "url_test_set"])
        assert reg_df.id.value_counts().max() == 1
        reg_df = reg_df.set_index("id")
        reg_df["src"] = reg_df.reset_index().id.apply(lambda x: x.split("-")[0]).values
        reg_df["tgt"] = reg_df.reset_index().id.apply(lambda x: x.split("-")[1]).values

        released_cols = [
            "url_base",
            "pair",  # (ISO639-3/ISO639-5 codes),
            "short_pair",  # (reduced codes),
            "chrF2_score",
            "bleu",
            "brevity_penalty",
            "ref_len",
            "src_name",
            "tgt_name",
        ]

        released = pd.read_csv("Tatoeba-Challenge/models/released-models.txt", sep="\t", header=None).iloc[:-1]
        released.columns = released_cols
        released["fname"] = released["url_base"].apply(
            lambda x: remove_suffix(remove_prefix(x, "https://object.pouta.csc.fi/Tatoeba-Challenge/opus"), ".zip")
        )

        released["2m"] = released.fname.str.startswith("2m")
        released["date"] = pd.to_datetime(
            released["fname"].apply(lambda x: remove_prefix(remove_prefix(x, "2m-"), "-"))
        )

        released["base_ext"] = released.url_base.apply(lambda x: Path(x).name)
        reg_df["base_ext"] = reg_df.url_model.apply(lambda x: Path(x).name)

        metadata_new = reg_df.reset_index().merge(released.rename(columns={"pair": "id"}), on=["base_ext", "id"])

        metadata_renamer = {"src": "src_alpha3", "tgt": "tgt_alpha3", "id": "long_pair", "date": "train_date"}
        metadata_new = metadata_new.rename(columns=metadata_renamer)

        metadata_new["src_alpha2"] = metadata_new.short_pair.apply(lambda x: x.split("-")[0])
        metadata_new["tgt_alpha2"] = metadata_new.short_pair.apply(lambda x: x.split("-")[1])
        DROP_COLS_BOTH = ["url_base", "base_ext", "fname"]

        metadata_new = metadata_new.drop(DROP_COLS_BOTH, 1)
        metadata_new["prefer_old"] = metadata_new.long_pair.isin([])
        self.metadata = metadata_new
        assert self.metadata.short_pair.value_counts().max() == 1, "Multiple metadata entries for a short pair"
        self.metadata = self.metadata.set_index("short_pair")

        # wget.download(LANG_CODE_URL)
        mapper = pd.read_csv(LANG_CODE_PATH)
        mapper.columns = ["a3", "a2", "ref"]
        self.iso_table = pd.read_csv(ISO_PATH, sep="\t").rename(columns=lambda x: x.lower())
        more_3_to_2 = self.iso_table.set_index("id").part1.dropna().to_dict()
        more_3_to_2.update(mapper.set_index("a3").a2.to_dict())
        self.alpha3_to_alpha2 = more_3_to_2
        self.model_card_dir = Path(save_dir)
        self.constituents = GROUP_MEMBERS

    def convert_models(self, tatoeba_ids, dry_run=False):
        entries_to_convert = [x for x in self.registry if x[0] in tatoeba_ids]
        converted_paths = convert_all_sentencepiece_models(entries_to_convert, dest_dir=self.model_card_dir)

        for path in converted_paths:
            long_pair = remove_prefix(path.name, "opus-mt-").split("-")  # eg. heb-eng
            assert len(long_pair) == 2
            new_p_src = self.get_two_letter_code(long_pair[0])
            new_p_tgt = self.get_two_letter_code(long_pair[1])
            hf_model_id = f"opus-mt-{new_p_src}-{new_p_tgt}"
            new_path = path.parent.joinpath(hf_model_id)  # opus-mt-he-en
            os.rename(str(path), str(new_path))
            self.write_model_card(hf_model_id, dry_run=dry_run)

    def get_two_letter_code(self, three_letter_code):
        return self.alpha3_to_alpha2.get(three_letter_code, three_letter_code)

    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.get_two_letter_code(x) for x in self.constituents[grp_name]]

    def get_tags(self, code, ref_name):
        if len(code) == 2:
            assert "languages" not in ref_name, f"{code}: {ref_name}"
            return [code], False
        elif "languages" in ref_name or len(self.constituents.get(code, [])) > 1:
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group, True
        else:  # zho-> zh
            print(f"Three letter monolingual code: {code}")
            return [code], False

    def resolve_lang_code(self, r) -> Tuple[List[str], str, str]:
        """R is a row in ported"""
        short_pair = r.short_pair
        src, tgt = short_pair.split("-")
        src_tags, src_multilingual = self.get_tags(src, r.src_name)
        assert isinstance(src_tags, list)
        tgt_tags, tgt_multilingual = self.get_tags(tgt, r.tgt_name)
        assert isinstance(tgt_tags, list)

        return dedup(src_tags + tgt_tags), src_multilingual, tgt_multilingual

    def write_model_card(
        self,
        hf_model_id: str,
        repo_root=DEFAULT_REPO,
        dry_run=False,
    ) -> str:
        """
        Copy the most recent model's readme section from opus, and add metadata. upload command: aws s3 sync
        model_card_dir s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        """
        short_pair = remove_prefix(hf_model_id, "opus-mt-")
        extra_metadata = self.metadata.loc[short_pair].drop("2m")
        extra_metadata["short_pair"] = short_pair
        lang_tags, src_multilingual, tgt_multilingual = self.resolve_lang_code(extra_metadata)
        opus_name = f"{extra_metadata.src_alpha3}-{extra_metadata.tgt_alpha3}"
        # opus_name: str = self.convert_hf_name_to_opus_name(hf_model_name)

        assert repo_root in ("OPUS-MT-train", "Tatoeba-Challenge")
        opus_readme_path = Path(repo_root).joinpath("models", opus_name, "README.md")
        assert opus_readme_path.exists(), f"Readme file {opus_readme_path} not found"

        opus_src, opus_tgt = [x.split("+") for x in opus_name.split("-")]

        readme_url = f"https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md"

        s, t = ",".join(opus_src), ",".join(opus_tgt)

        metadata = {
            "hf_name": short_pair,
            "source_languages": s,
            "target_languages": t,
            "opus_readme_url": readme_url,
            "original_repo": repo_root,
            "tags": ["translation"],
            "languages": lang_tags,
        }
        lang_tags = l2front_matter(lang_tags)
        metadata["src_constituents"] = self.constituents[s]
        metadata["tgt_constituents"] = self.constituents[t]
        metadata["src_multilingual"] = src_multilingual
        metadata["tgt_multilingual"] = tgt_multilingual

        metadata.update(extra_metadata)
        metadata.update(get_system_metadata(repo_root))

        # combine with Tatoeba markdown

        extra_markdown = f"### {short_pair}\n\n* source group: {metadata['src_name']} \n* target group: {metadata['tgt_name']} \n*  OPUS readme: [{opus_name}]({readme_url})\n"

        content = opus_readme_path.open().read()
        content = content.split("\n# ")[-1]  # Get the lowest level 1 header in the README -- the most recent model.
        splat = content.split("*")[2:]

        content = "*".join(splat)
        # BETTER FRONT MATTER LOGIC

        content = (
            FRONT_MATTER_TEMPLATE.format(lang_tags)
            + extra_markdown
            + "\n* "
            + content.replace("download", "download original " "weights")
        )

        items = "\n\n".join([f"- {k}: {v}" for k, v in metadata.items()])
        sec3 = "\n### System Info: \n" + items
        content += sec3
        if dry_run:
            return content, metadata
        sub_dir = self.model_card_dir / hf_model_id
        sub_dir.mkdir(exist_ok=True)
        dest = sub_dir / "README.md"
        dest.open("w").write(content)
        pd.Series(metadata).to_json(sub_dir / "metadata.json")
        return content, metadata

    def download_metadata(self):
        Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
        import wget

        if not os.path.exists(ISO_PATH):
            wget.download(ISO_URL, ISO_PATH)
        if not os.path.exists(LANG_CODE_PATH):
            wget.download(LANG_CODE_URL, LANG_CODE_PATH)

    @staticmethod
    def make_tatoeba_registry(repo_path=DEFAULT_MODEL_DIR):
        if not (Path(repo_path) / "zho-eng" / "README.md").exists():
            raise ValueError(
                f"repo_path:{repo_path} does not exist: "
                "You must run: git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git before calling."
            )
        results = {}
        for p in Path(repo_path).iterdir():
            if len(p.name) != 7:
                continue
            lns = list(open(p / "README.md").readlines())
            results[p.name] = _parse_readme(lns)
        return [(k, v["pre-processing"], v["download"], v["download"][:-4] + ".test.txt") for k, v in results.items()]


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
        if not item:
            continue
        elif item in new_lst:
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
