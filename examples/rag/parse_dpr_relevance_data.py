"""
This script reads DPR retriever training data and parses each datapoint. We save a line per datapoint.
Each line consists of the query followed by a tab-separated list of Wikipedia page titles constituting
positive contexts for a given query.
"""

import argparse
import json

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--src_path", type=str, default="biencoder-nq-dev.json", help="Path to raw DPR training data",
    )
    parser.add_argument(
        "--dst_path", type=str, help="where to store parsed file",
    )
    args = parser.parse_args()

    with open(args.src_path, "r") as src_file, open(args.dst_path, "w") as dst_file:
        dpr_records = json.load(src_file)
        for dpr_record in tqdm(dpr_records):
            question = dpr_record["question"]
            contexts = [context["title"] for context in dpr_record["positive_ctxs"]]
            dst_file.write("\t".join([question] + contexts) + "\n")


if __name__ == "__main__":
    main()
