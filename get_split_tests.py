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
    stat_idx, end_idx = n_tests_per_split * split_idx, n_tests_per_split * (split_idx + 1)
    split_tests = tests[stat_idx:end_idx]

    split_tests = "\n".join(split_tests)

    with open("splitted_tests.txt", "w") as fp:
        fp.write(split_tests)
