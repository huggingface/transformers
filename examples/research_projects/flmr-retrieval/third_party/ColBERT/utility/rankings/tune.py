import os
import ujson
import random

from argparse import ArgumentParser
from colbert.utils.utils import print_message, create_directory
from utility.utils.save_metadata import save_metadata


def main(args):
    AllMetrics = {}
    Scores = {}

    for path in args.paths:
        with open(path) as f:
            metric = ujson.load(f)
            AllMetrics[path] = metric

            for k in args.metric:
                metric = metric[k]

            assert type(metric) is float
            Scores[path] = metric
    
    MaxKey = max(Scores, key=Scores.get)

    MaxCKPT = int(MaxKey.split('/')[-2].split('.')[-1])
    MaxARGS = os.path.join(os.path.dirname(MaxKey), 'logs', 'args.json')

    with open(MaxARGS) as f:
        logs = ujson.load(f)
        MaxCHECKPOINT = logs['checkpoint']

        assert MaxCHECKPOINT.endswith(f'colbert-{MaxCKPT}.dnn'), (MaxCHECKPOINT, MaxCKPT)

    with open(args.output, 'w') as f:
        f.write(MaxCHECKPOINT)

    args.Scores = Scores
    args.AllMetrics = AllMetrics

    save_metadata(f'{args.output}.meta', args)

    print('\n\n', args, '\n\n')
    print(args.output)
    print_message("#> Done.")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='.')

    # Input / Output Arguments
    parser.add_argument('--metric', dest='metric', required=True, type=str)  # e.g., success.20
    parser.add_argument('--paths', dest='paths', required=True, type=str, nargs='+')
    parser.add_argument('--output', dest='output', required=True, type=str)

    args = parser.parse_args()

    args.metric = args.metric.split('.')

    assert not os.path.exists(args.output), args.output
    create_directory(os.path.dirname(args.output))

    main(args)
