from pathlib import Path

import fire
from tqdm import tqdm


def download_wmt_dataset(src_lang="ro", tgt_lang="en", dataset="wmt16", save_dir=None) -> None:
    """Download a dataset using the nlp package and save it to the format expected by finetune.py
    Format of save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> source language
        tgt_lang: <str> target language
        dataset: <str> like wmt16 (if you don't know, try wmt16).
        save_dir: <str>, where to save the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> download_wmt_dataset('en', 'ru', dataset='wmt19') # saves to wmt19_en_ru
    """
    try:
        import nlp
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install nlp")
    pair = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{pair}")
    ds = nlp.load_dataset(dataset, pair)
    if save_dir is None:
        save_dir = f"{dataset}-{pair}"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for split in tqdm(ds.keys()):
        print(f"Splitting {split} with {ds[split].num_rows} records")

        # to save to val.source, val.target like summary datasets
        fn = "val" if split == "validation" else split
        src_path = save_dir.joinpath(f"{fn}.source")
        src_fp = src_path.open("w+")
        tgt_path = save_dir.joinpath(f"{fn}.target")
        tgt_fp = tgt_path.open("w+")

        src, tgt = [], []
        # read/write in chunks so not to consume potentially 100GB of RAM
        for x in tqdm(ds[split]):
            ex = x["translation"]
            src.append(ex[src_lang])
            tgt.append(ex[tgt_lang])
            if len(src) == 100000:
                src_fp.write("\n".join(src))
                tgt_fp.write("\n".join(src))
                src, tgt = [], []
        # any leftovers
        src_fp.write("\n".join(src))
        tgt_fp.write("\n".join(src))

    print(f"saved dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)
