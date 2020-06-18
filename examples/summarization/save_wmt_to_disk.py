from pathlib import Path

from tqdm import tqdm

import nlp


ds = nlp.load_dataset("./datasets/wmt16", "de-en")
save_dir = Path("wmt_de_en")
save_dir.mkdir(exist_ok=True)
for split in tqdm(ds.keys()):
    tr_list = list(ds[split])
    data = [x["translation"] for x in tr_list]
    src, tgt = [], []
    for example in data:
        src.append(example["en"])
        tgt.append(example["ro"])
    if split == "validation":
        split = "val"  # to save to val.source, val.target like summary datasets
    src_path = save_dir.joinpath(f"{split}.source")
    src_path.open("w+").write("\n".join(src))
    tgt_path = save_dir.joinpath(f"{split}.target")
    tgt_path.open("w+").write("\n".join(tgt))
