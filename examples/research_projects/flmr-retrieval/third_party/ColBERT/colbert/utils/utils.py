import os
import tqdm
import torch
import datetime
import itertools

from multiprocessing import Pool
from collections import OrderedDict, defaultdict


def print_message(*s, condition=True, pad=False):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f'\n{msg}\n'
        print(msg, flush=True)


    return msg


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm.tqdm(total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB") as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def torch_load_dnn(path):
    if path.startswith("http:") or path.startswith("https:"):
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        dnn = torch.load(path, map_location='cpu')
    
    return dnn

def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    print(f"#> Saving a checkpoint to {path} ..")

    if hasattr(model, 'module'):
        model = model.module  # extract model from a distributed/data-parallel wrapper

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['arguments'] = arguments

    torch.save(checkpoint, path)


def load_checkpoint(path, model, checkpoint=None, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path, "..")

    if checkpoint is None:
        checkpoint = load_checkpoint_raw(path)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint


def load_checkpoint_raw(path):
    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]
        new_state_dict[name] = v

    checkpoint['model_state_dict'] = new_state_dict

    return checkpoint


def create_directory(path):
    if os.path.exists(path):
        print('\n')
        print_message("#> Note: Output directory", path, 'already exists\n\n')
    else:
        print('\n')
        print_message("#> Creating directory", path, '\n\n')
        os.makedirs(path)

# def batch(file, bsize):
#     while True:
#         L = [ujson.loads(file.readline()) for _ in range(bsize)]
#         yield L
#     return


def f7(seq):
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Credit: derek73 @ https://stackoverflow.com/questions/2352181
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class dotdict_lax(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result


def zipstar(L, lazy=False):
    """
    A much faster A, B, C = zip(*[(a, b, c), (a, b, c), ...])
    May return lists or tuples.
    """

    if len(L) == 0:
        return L

    width = len(L[0])

    if width < 100:
        return [[elem[idx] for elem in L] for idx in range(width)]

    L = zip(*L)

    return L if lazy else list(L)


def zip_first(L1, L2):
    length = len(L1) if type(L1) in [tuple, list] else None

    L3 = list(zip(L1, L2))

    assert length in [None, len(L3)], "zip_first() failure: length differs!"

    return L3


def int_or_float(val):
    if '.' in val:
        return float(val)
        
    return int(val)

def load_ranking(path, types=None, lazy=False):
    print_message(f"#> Loading the ranked lists from {path} ..")

    try:
        lists = torch.load(path)
        lists = zipstar([l.tolist() for l in tqdm.tqdm(lists)], lazy=lazy)
    except:
        if types is None:
            types = itertools.cycle([int_or_float])

        with open(path) as f:
            lists = [[typ(x) for typ, x in zip_first(types, line.strip().split('\t'))]
                     for line in file_tqdm(f)]

    return lists


def save_ranking(ranking, path):
    lists = zipstar(ranking)
    lists = [torch.tensor(l) for l in lists]

    torch.save(lists, path)

    return lists


def groupby_first_item(lst):
    groups = defaultdict(list)

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(rest)

    return groups


def process_grouped_by_first_item(lst):
    """
        Requires items in list to already be grouped by first item.
    """

    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            yield (last_group, groups[last_group])
            assert first not in groups, f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True

    return groups


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def lengths2offsets(lengths):
    offset = 0

    for length in lengths:
        yield (offset, offset + length)
        offset += length

    return


# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def load_batch_backgrounds(args, qids):
    if args.qid2backgrounds is None:
        return None

    qbackgrounds = []

    for qid in qids:
        back = args.qid2backgrounds[qid]

        if len(back) and type(back[0]) == int:
            x = [args.collection[pid] for pid in back]
        else:
            x = [args.collectionX.get(pid, '') for pid in back]

        x = ' [SEP] '.join(x)
        qbackgrounds.append(x)
    
    return qbackgrounds
