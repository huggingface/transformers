import fire

from utils import calculate_rouge, save_json, read, write_txt_file


def calculate_rouge_path(pred_path, tgt_path, save_path=None, **kwargs):
    """Kwargs will be passed to calculate_rouge"""
    pred_lns = [x.strip() for x in open(pred_path).readlines()]
    tgt_lns = [x.strip() for x in open(tgt_path).readlines()][: len(pred_lns)]
    metrics = calculate_rouge(pred_lns, tgt_lns, **kwargs)
    if save_path is not None:
        save_json(metrics, save_path, indent=None)
    return metrics  # these print nicely
from pathlib import Path
def read_txt_file(path) -> list:
    lns = Path(path).open().read().split('\n')
    return lns

def remove_empty_entries(data_dir, save_dir, split='test'):
    src = read_txt_file(f'{data_dir}/{split}.source')
    tgt = read_txt_file(f'{data_dir}/{split}.target')
    empty = []
    new_src, new_tgt = [], []
    for i in range(len(tgt)):
        if (len(tgt[i]) == 0) or (len(src[i]) == 0):
            empty.append(i)
            continue
        new_src.append(src[i])
        new_tgt.append(tgt[i])
    Path(save_dir).mkdir(exist_ok=True)
    write_txt_file(new_src, f'{save_dir}/{split}.source')
    write_txt_file(new_tgt, f'{save_dir}/{split}.target')


if __name__ == "__main__":
    fire.Fire(calculate_rouge_path)
