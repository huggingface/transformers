import argparse 

"""
Prints results in logfile from benchmark_bettertransformer.py to be easy to copy to an excel
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="",
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=5,
        help="",
    )
    return parser

def summarize_results(n_repeat, filename='log.csv'):
    """
    Deprecated. 
    """
    lst = [[] for _ in range(n_repeat)]
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            lst[i%n_repeat].append(line)
    
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            print(lst[i][j], end='')

def summarize_results2(n_repeat, filename='log.csv'):
    """
    Puts lines in each block (of size n_repeat) onto one line, with space delineation
    Easy to copy paste into google sheets
    """
    lst = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            lst.append(line.rstrip())
            if((i+1) % n_repeat == 0):
                print(' '.join(lst))
                lst = []

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    summarize_results2(n_repeat=args.n_repeat, filename=args.filename)