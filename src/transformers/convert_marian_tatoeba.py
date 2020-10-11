from pathlib import Path
from typing import List, Tuple

import numpy as np
from transformers.convert_marian_to_pytorch import (
    remove_suffix, remove_prefix, convert_all_sentencepiece_models, ORG_NAME, get_system_metadata, lmap, make_registry, _parse_readme,
)
from transformers.marian_constituents import GROUP_MEMBERS
import os
try:
    import pandas as pd
except ImportError:
    pass

import shutil

DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, 'models')
LANG_CODE_URL = 'https://datahub.io/core/language-codes/r/language-codes-3b2.csv'

class TatoebaCodeResolver:
    def __init__(self, save_dir='marian_converted'):
        assert Path(DEFAULT_REPO).exists(), 'need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git'
        reg = make_tatoeba_registry()
        self.registry = reg
        reg_df = pd.DataFrame(reg, columns=['id', 'prepro', 'url_model', 'url_test_set'])
        assert reg_df.id.value_counts().max() == 1
        reg_df = reg_df.set_index('id')
        _get_src = lambda x: x.split('-')[0]
        _get_tgt = lambda x: x.split('-')[1]
        reg_df['src'] = reg_df.reset_index().id.apply(_get_src).values
        reg_df['tgt'] = reg_df.reset_index().id.apply(_get_tgt).values

        released_cols = ['url_base',
                         'pair',  # (ISO639-3/ISO639-5 codes),
                         'short_pair',  # (reduced codes),
                         'chrF2_score',
                         'bleu',
                         'brevity_penalty',
                         'ref_len',
                         'src_name', 'tgt_name']

        released = pd.read_csv('Tatoeba-Challenge/models/released-models.txt', sep='\t', header=None).iloc[:-1]
        released.columns = released_cols
        released['fname'] = released['url_base'].apply(
            lambda x: remove_suffix(remove_prefix(x, 'https://object.pouta.csc.fi/Tatoeba-Challenge/opus'), '.zip'))

        released['2m'] = released.fname.str.startswith('2m')
        released['date'] = pd.to_datetime(
            released['fname'].apply(lambda x: remove_prefix(remove_prefix(x, '2m-'), '-')))

        released['base_ext'] = released.url_base.apply(lambda x: Path(x).name)
        reg_df['base_ext'] = reg_df.url_model.apply(lambda x: Path(x).name)

        metadata_new = reg_df.reset_index().merge(released.rename(columns={'pair': 'id'}), on=['base_ext', 'id'])

        metadata_renamer = {'src': 'src_alpha3', 'tgt': 'tgt_alpha3', 'id': 'long_pair', 'date': 'train_date'}
        metadata_new = metadata_new.rename(columns=metadata_renamer)

        metadata_new['src_alpha2'] = metadata_new.short_pair.apply(lambda x: x.split('-')[0])
        metadata_new['tgt_alpha2'] = metadata_new.short_pair.apply(lambda x: x.split('-')[1])
        DROP_COLS_BOTH = ['url_base', 'base_ext', 'fname']

        metadata_new = metadata_new.drop(DROP_COLS_BOTH, 1)
        metadata_new['prefer_old'] = metadata_new.long_pair.isin([])
        self.metadata = metadata_new
        self.tab = pd.read_csv('iso-639-3.csv', sep='\t').rename(columns=lambda x: x.lower())

        # wget.download(LANG_CODE_URL)
        mapper = pd.read_csv('language-codes-3b2.csv')
        mapper.columns = ['a3', 'a2', 'ref']
        a3to2 = mapper.set_index('a3')
        a2to3 = mapper.set_index('a2')
        more_3_to_2 = self.tab.set_index('id').part1.dropna().to_dict()
        before = len(more_3_to_2)
        more_3_to_2.update(a3to2.a2.to_dict())
        self.more_3_to_2 = more_3_to_2
        self.model_card_dir = Path(save_dir)

    def convert_model(self, tatoeba_ids, dry_run=False):
        entries_to_convert = [x for x in self.registry if x[0] in tatoeba_ids]
        converted_paths = convert_all_sentencepiece_models(entries_to_convert)
        for path in converted_paths:
            long_pair = remove_prefix(path.name, 'opus-mt-').split('-')  # eg. heb-eng
            assert len(long_pair) == 2
            new_p_src = self.get_two_letter_code(long_pair[0])
            new_p_tgt = self.get_two_letter_code(long_pair[1])
            hf_model_id = f'opus-mt-{new_p_src}-{new_p_tgt}'
            new_path = path.parent.joinpath()  # opus-mt-he-en
            shutil.mv(path, new_path)
            metadata_row = self.metadata.loc[hf_model_id].drop('2m')
            content, mmeta = self.write_model_card(hf_model_id, repo_root=DEFAULT_REPO, save_dir=self.model_card_dir,
                                                   dry_run=dry_run, extra_metadata=metadata_row)

    def download_everything(self):
        raise NotImplementedError()

    def get_two_letter_code(self, three_letter_code):
        return self.more_3_to_2.get(three_letter_code, three_letter_code)

    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.get_two_letter_code(x) for x in self.constituents[grp_name]]

    def get_tags(self, code, ref_name):
        if len(code) == 2:
            assert 'languages' not in ref_name, f'{code}: {ref_name}'
            return [code], False
        elif 'languages' in ref_name or len(self.constituents.get(code, [])) > 1:
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group, True
        else:  # zho-> zh
            print(f'Three letter monolingual code: {code}')
            return [code], False

    def resolve_lang_code(self, r) -> Tuple[List[str], str, str]:
        """R is a row in ported"""
        short_pair = r.short_pair
        src, tgt = short_pair.split('-')
        src_tags, src_multilingual = self.get_tags(src, r.src_name)
        assert isinstance(src_tags, list)
        tgt_tags, tgt_multilingual = self.get_tags(tgt, r.tgt_name)
        assert isinstance(tgt_tags, list)

        return dedup(src_tags + tgt_tags), src_multilingual, tgt_multilingual

    def write_model_card(
            self,
            hf_model_id: str,
            repo_root="Tatoeba-Challenge",
            dry_run=False,
            extra_metadata={}
    ) -> str:
        """Copy the most recent model's readme section from opus, and add metadata.
        upload command: aws s3 sync model_card_dir s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        """
        short_pair = remove_prefix(hf_model_id, ORG_NAME)
        lang_tags, src_multilingual, tgt_multilingual = self.resolve_lang_code(extra_metadata)
        opus_name = f'{src_multilingual}-{tgt_multilingual}'
        #opus_name: str = self.convert_hf_name_to_opus_name(hf_model_name)

        assert repo_root in ('OPUS-MT-train', 'Tatoeba-Challenge')
        opus_readme_path = Path(repo_root).joinpath('models', opus_name, 'README.md')
        assert opus_readme_path.exists(), f"Readme file {opus_readme_path} not found"

        opus_src, opus_tgt = [x.split("+") for x in opus_name.split("-")]

        readme_url = f"https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md"

        s, t = ",".join(opus_src), ",".join(opus_tgt)

        metadata = {'hf_name': short_pair, 'source_languages': s, 'target_languages': t,
                    'opus_readme_url': readme_url,
                    'original_repo': repo_root, 'tags': ['translation'], 'languages': lang_tags,
                    }
        lang_tags = l2front_matter(lang_tags)
        metadata['src_constituents'] = self.constituents[s]
        metadata['tgt_constituents'] = self.constituents[t]
        metadata['src_multilingual'] = src_multilingual
        metadata['tgt_multilingual'] = tgt_multilingual

        metadata.update(extra_metadata)
        metadata.update(get_system_metadata(repo_root))

        # combine with opus markdown

        extra_markdown = f"### {short_pair}\n\n* source group: {metadata['src_name']} \n* target group: {metadata['tgt_name']} \n*  OPUS readme: [{opus_name}]({readme_url})\n"

        content = opus_readme_path.open().read()
        content = content.split("\n# ")[-1]  # Get the lowest level 1 header in the README -- the most recent model.
        splat = content.split("*")[2:]

        content = "*".join(splat)
        # BETTER FRONT MATTER LOGIC

        content = front_matter.format(lang_tags) + extra_markdown + "\n* " + content.replace("download",
                                                                                             "download original " \
                                                                                             "weights")

        items = '\n\n'.join([f'- {k}: {v}' for k, v in metadata.items()])
        sec3 = '\n### System Info: \n' + items
        content += sec3
        if dry_run:
            return content, metadata
        sub_dir = self.model_card_dir / hf_model_id
        sub_dir.mkdir(exist_ok=True)
        dest = sub_dir / "README.md"
        dest.open("w").write(content)
        pd.Series(metadata).to_json(sub_dir / 'metadata.json')
        return content, metadata


def _process_benchmark_table_row(x):
    fields = lmap(str.strip, x.replace("\t", "").split("|")[1:-1])
    assert len(fields) == 3
    return (fields[0], float(fields[1]), float(fields[2]))


def process_last_benchmark_table(readme_path) -> List[Tuple[str, float, float]]:
    md_content = Path(readme_path).open().read()
    entries = md_content.split("## Benchmarks")[-1].strip().split("\n")[2:]
    data = lmap(_process_benchmark_table_row, entries)
    return data


def check_if_models_are_dominated(old_repo_path="OPUS-MT-train/models", new_repo_path="Tatoeba-Challenge/models/"):
    """Make a blacklist for models where we have already ported the same language pair, and the ported model has
    higher BLEU score."""
    import pandas as pd

    newest_released, old_reg, released = get_released_df(new_repo_path, old_repo_path)

    short_to_new_bleu = newest_released.set_index("short_pair").bleu

    assert released.groupby("short_pair").pair.nunique().max() == 1

    short_to_long = released.groupby("short_pair").pair.first().to_dict()

    overlap_short = old_reg.index.intersection(released.short_pair.unique())
    overlap_long = [short_to_long[o] for o in overlap_short]
    new_reported_bleu = [short_to_new_bleu[o] for o in overlap_short]

    def get_old_bleu(o) -> float:
        pat = old_repo_path + "/{}/README.md"
        bm_data = process_last_benchmark_table(pat.format(o))
        tab = pd.DataFrame(bm_data, columns=["testset", "bleu", "chr-f"])
        tato_bleu = tab.loc[lambda x: x.testset.str.startswith("Tato")].bleu
        if tato_bleu.shape[0] > 0:
            return tato_bleu.iloc[0]
        else:
            return np.nan

    old_bleu = [get_old_bleu(o) for o in overlap_short]
    cmp_df = pd.DataFrame(
        dict(short=overlap_short, long=overlap_long, old_bleu=old_bleu, new_bleu=new_reported_bleu)
    ).fillna(-1)

    dominated = cmp_df[cmp_df.old_bleu > cmp_df.new_bleu]
    whitelist_df = cmp_df[cmp_df.old_bleu <= cmp_df.new_bleu]
    blacklist = dominated.long.unique().tolist()  # 3 letter codes
    return whitelist_df, dominated, blacklist


def get_released_df(new_repo_path, old_repo_path):
    import pandas as pd

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
    released = pd.read_csv(f"{new_repo_path}/released-models.txt", sep="\t", header=None).iloc[:-1]
    released.columns = released_cols
    old_reg = make_registry(repo_path=old_repo_path)
    old_reg = pd.DataFrame(old_reg, columns=["id", "prepro", "url_model", "url_test_set"])
    assert old_reg.id.value_counts().max() == 1
    old_reg = old_reg.set_index("id")
    released["fname"] = released["url_base"].apply(
        lambda x: remove_suffix(remove_prefix(x, "https://object.pouta.csc.fi/Tatoeba-Challenge/opus"), ".zip")
    )
    released["2m"] = released.fname.str.startswith("2m")
    released["date"] = pd.to_datetime(released["fname"].apply(lambda x: remove_prefix(remove_prefix(x, "2m-"), "-")))
    newest_released = released.dsort("date").drop_duplicates(["short_pair"], keep="first")
    return newest_released, old_reg, released


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


front_matter = """---
language: 
{}
tags:
- translation

license: apache-2.0
---

"""


def l2front_matter(langs):
    return ''.join(f'- {l}\n' for l in langs)


def dedup(lst):
    new_lst = []
    for item in lst:
        if not item:
            continue
        elif item in new_lst:
            continue
        else:
            new_lst.append(item)
    return new_lst


if __name__ == '__main__':
    resolver = TatoebaCodeResolver()
    resolver.convert_model(['heb-eng', 'eng-heb'])
