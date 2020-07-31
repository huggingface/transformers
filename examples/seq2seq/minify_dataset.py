from pathlib import Path

import fire


def minify(src_dir: str, dest_dir: str, n: int):
    """Write first n lines of each file f in src_dir to dest_dir/f """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    for path in src_dir.iterdir():
        try:
            new = [x.rstrip() for x in list(path.open().readlines())][:n]
        except Exception:
            print(f' cant minify {path}')
            continue
        dest_path = dest_dir.joinpath(path.name)
        print(dest_path)
        dest_path.open("w").write("\n".join(new))


if __name__ == "__main__":
    fire.Fire(minify)
