import torch
import argparse
from typing import Dict
from tqdm import tqdm
def convert(dct) -> Dict:
    for k,v in tqdm(dct.items()):
        try:
            dct[k] = v.half()
        except Exception:
            continue
        if isinstance(v, dict):
            dct[k] = convert(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            dct[k] = type(v)(convert(x) for x in v)
        else: # unconvertable
            dct[k] = v
    return dct

def fp16_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--map_location", default='cpu', type=str)
    args = parser.parse_args()
    dct = convert(torch.load(args.src_path, map_location=args.map_location))
    save_path = args.src_path if args.save_path is None else args.save_path
    torch.save(dct, save_path)


if __name__ == "__main__":
    fp16_cli()
