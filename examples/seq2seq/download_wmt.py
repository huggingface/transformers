#!/usr/bin/env python

from pathlib import Path

import fire
from tqdm import tqdm


def download_wmt_dataset(src_lang="ro", tgt_lang="en", dataset="wmt16", save_dir=None) -> None:
    """Download a dataset using the datasets package and save it to the format expected by finetune.py
    Format of save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> source language
        tgt_lang: <str> target language
        dataset: <str> wmt16, wmt17, etc. wmt16 is a good start as it's small. To get the full list run `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        save_dir: <str>, where to save the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> download_wmt_dataset('ro', 'en', dataset='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    pair = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{pair}")
    ds = datasets.load_dataset(dataset, pair)
    if save_dir is None:
        save_dir = f"{dataset}-{pair}"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for split in ds.keys():
        print(f"Splitting {split} with {ds[split].num_rows} records")

        # to save to val.source, val.target like summary datasets
        fn = "val" if split == "validation" else split
        src_path = save_dir.joinpath(f"{fn}.source")
        tgt_path = save_dir.joinpath(f"{fn}.target")
        src_fp = src_path.open("w+")
        tgt_fp = tgt_path.open("w+")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        for x in tqdm(ds[split]):
            ex = x["translation"]
            src_fp.write(ex[src_lang] + "\n")
            tgt_fp.write(ex[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)
