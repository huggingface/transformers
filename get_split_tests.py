import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--target", default=None, type=str, required=True, help="Path to the output Tensorflow dump file."
    )

    args = parser.parse_args()

    with open(args.target) as fp:
        tests = fp.read().split('\n')

    split_idx = int(os.environ["CIRCLE_NODE_INDEX"])
    n_splits = int(os.environ["CIRCLE_NODE_TOTAL"])



    n_tests_per_split = len(tests) // int(n_splits)
    r = len(tests) % int(n_splits)

    split_indices = []
    start = 0
    end = 0
    for idx in range(n_splits):
        n_tests = n_tests_per_split
        if idx < r:
            n_tests += 1
        end = start + n_tests
        split_indices.append((start, end))
        start = end

    stat_idx, end_idx = split_indices[split_idx]
    split_tests = tests[stat_idx:end_idx]

    split_tests = "\n".join(split_tests)

    with open("splitted_tests.txt", "w") as fp:
        fp.write(split_tests)
