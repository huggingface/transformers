from pathlib import Path

import fire
from tqdm import tqdm


def download_summarization_dataset(dataset, save_dir=None, split=None, **load_kwargs) -> None:
    """Download a dataset using the datasets package and save it to the format expected by finetune.py
    Format of save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        dataset: <str> xsum, aeslc etc.
        save_dir: <str>, where to save the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> download_summarization_dataset('xsum', split='test') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    #pair = f"{src_lang}-{tgt_lang}"
    #print(f"Converting {dataset}-{pair}")
    ds = datasets.load_dataset(dataset, split=split, **load_kwargs)
    if save_dir is None:
        save_dir = dataset
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
            src_fp.write(x['document'] + "\n")
            tgt_fp.write(x['summary'] + "\n")

    print(f"Saved {dataset} dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(download_summarization_dataset)
