import os
from pathlib import Path


def get_file(lpath, outfile=None, **kwargs) -> None:
    # # Taken from https://github.com/fsspec/filesystem_spec/blob/47b445ae4c284a82dd15e0287b1ffc410e8fc470/fsspec/spec.py#L883
    # if isfilelike(lpath):
    #     outfile = lpath
    # elif self.isdir(rpath):
    #     os.makedirs(lpath, exist_ok=True)
    #     return None

    if isinstance(lpath, (str, Path)):  # otherwise, let's assume it's a file-like object
        os.makedirs(os.path.dirname(lpath), exist_ok=True)
        # import time
        # time.sleep(10)

    # Open file if not already open
    close_file = False
    if outfile is None:
        outfile = open(lpath, "wb")
        print(outfile)


get_file(lpath="/mnt/cache/datasets/downloads/0f3877f8b8ff8f882b5032f93535a4d21bd7475946970fbf306f79c2769f3d0d.incomplete")