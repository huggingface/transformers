import argparse
from pathlib import Path

import tensorflow_datasets as tfds


def main(input_path, reference_path, data_dir):
    cnn_ds = tfds.load("cnn_dailymail", split="test", shuffle_files=False, data_dir=data_dir)
    cnn_ds_iter = tfds.as_numpy(cnn_ds)

    test_articles_file = Path(input_path).open("w")
    test_summaries_file = Path(reference_path).open("w")

    for example in cnn_ds_iter:
        test_articles_file.write(example["article"].decode("utf-8") + "\n")
        test_articles_file.flush()
        test_summaries_file.write(example["highlights"].decode("utf-8").replace("\n", " ") + "\n")
        test_summaries_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="where to save the articles input data")
    parser.add_argument(
        "reference_path", type=str, help="where to save the reference summaries",
    )
    parser.add_argument(
        "--data_dir", type=str, default="~/tensorflow_datasets", help="where to save the tensorflow datasets.",
    )
    args = parser.parse_args()
    main(args.input_path, args.reference_path, args.data_dir)
