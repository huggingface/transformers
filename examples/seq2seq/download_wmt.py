from pathlib import Path

import fire
from tqdm import tqdm


def download_wmt_dataset(src_lang, tgt_lang, dataset="wmt19", save_dir=None) -> None:
    """Download a dataset using the nlp package and save it to the format expected by finetune.py
    Format of save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> source language
        tgt_lang: <str> target language
        dataset: <str> like wmt19 (if you don't know, try wmt19).
        save_dir: <str>, where to save the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> download_wmt_dataset('en', 'ru', dataset='wmt19') # saves to wmt19_en_ru
    """
    try:
        import nlp
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install nlp")
    pair = f"{src_lang}-{tgt_lang}"
    ds = nlp.load_dataset(dataset, pair)
    if save_dir is None:
        save_dir = f"{dataset}-{pair}"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for split in tqdm(ds.keys()):
        tr_list = list(ds[split])
        data = [x["translation"] for x in tr_list]
        src, tgt = [], []
        for example in data:
            src.append(example[src_lang])
            tgt.append(example[tgt_lang])
        if split == "validation":
            split = "val"  # to save to val.source, val.target like summary datasets
        src_path = save_dir.joinpath(f"{split}.source")
        src_path.open("w+").write("\n".join(src))
        tgt_path = save_dir.joinpath(f"{split}.target")
        tgt_path.open("w+").write("\n".join(tgt))
    print(f"saved dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)
