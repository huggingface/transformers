import re
import os
import ujson


def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def load_doclens(directory, flatten=True):
    doclens_filenames = {}

    for filename in os.listdir(directory):
        match = re.match("doclens.(\d+).json", filename)

        if match is not None:
            doclens_filenames[int(match.group(1))] = filename

    doclens_filenames = [os.path.join(directory, doclens_filenames[i]) for i in sorted(doclens_filenames.keys())]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]
    
    if len(all_doclens) == 0:
        raise ValueError("Could not load doclens")

    return all_doclens


def get_deltas(directory):
    extension = '.residuals.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]

    return parts, parts_paths


# def load_compression_data(level, path):
#     with open(path, "r") as f:
#         for line in f:
#             line = line.split(',')
#             bits = int(line[0])

#             if bits == level:
#                 return [float(v) for v in line[1:]]

#     raise ValueError(f"No data found for {level}-bit compression")
